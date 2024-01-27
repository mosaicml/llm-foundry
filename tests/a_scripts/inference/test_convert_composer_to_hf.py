# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import math
import os
import pathlib
import shutil
from argparse import Namespace
from typing import Any, Callable, Optional, cast
from unittest.mock import ANY, MagicMock, patch

import pytest
import torch
import transformers
from composer import Trainer
from composer.loggers import MLFlowLogger
from composer.utils import dist, get_device
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from llmfoundry import COMPOSER_MODEL_REGISTRY
from llmfoundry.callbacks import HuggingFaceCheckpointer
from llmfoundry.callbacks.hf_checkpointer import _maybe_get_license_filename
from llmfoundry.data.finetuning import build_finetuning_dataloader
from llmfoundry.models.mpt.modeling_mpt import ComposerMPTCausalLM
from llmfoundry.utils.builders import build_optimizer, build_tokenizer
from scripts.inference.convert_composer_to_hf import convert_composer_to_hf
from tests.data_utils import make_tiny_ft_dataset

def _save_model_mock(*args: Any, path: str, **kwargs: Any):
    os.makedirs(path, exist_ok=True)

def check_hf_tokenizer_equivalence(tokenizer1: PreTrainedTokenizerBase,
                                   tokenizer2: PreTrainedTokenizerBase):
    """WARNING: Parameters are updated within the check so don't call check_hf_tokenizer_equivalence on the same

    params more than once

    This is a best effort attempt to compare two tokenizers for equivalence

    This is not a perfect test, but it should catch most issues. We first check that the vocab is identical
    and that a string is tokenized the same one. Then we compare the __dict__ of the tokenizers, but we remove
    some keys that are not important for equivalence. See the inline explanations for each one.
    """
    if hasattr(tokenizer1, 'vocab') or hasattr(tokenizer2, 'vocab'):
        assert tokenizer1.vocab == tokenizer2.vocab

    # we only care about the file and class name, not the full import path
    assert str(type(tokenizer1)).split('.')[-2:] == str(
        type(tokenizer2)).split('.')[-2:]

    expected_tokenizer_output = tokenizer2(
        'This is some text that should get tokenizer !? @ totallyarealtoken')
    actual_tokenizer_output = tokenizer1(
        'This is some text that should get tokenizer !? @ totallyarealtoken')
    assert expected_tokenizer_output == actual_tokenizer_output

    # we remove the actual _tokenizer object because it is an instantiated object and so does not pass equality
    # the tokenizers are not usable below these pops
    if hasattr(tokenizer1, '_tokenizer') or hasattr(tokenizer2, '_tokenizer'):
        tokenizer1.__dict__.pop('_tokenizer')
        tokenizer2.__dict__.pop('_tokenizer')

    # we remove a couple more objects because they are instantiated objects and so do not pass equality
    if hasattr(tokenizer1, 'sp_model') or hasattr(tokenizer2, 'sp_model'):
        tokenizer1.__dict__.pop('sp_model')
        tokenizer2.__dict__.pop('sp_model')

    if hasattr(tokenizer1, 'tokens_trie') or hasattr(tokenizer2, 'tokens_trie'):
        tokenizer1.__dict__.pop('tokens_trie')
        tokenizer2.__dict__.pop('tokens_trie')

    # extra key that is not important
    if hasattr(tokenizer1, 'deprecation_warnings') or hasattr(
            tokenizer2, 'deprecation_warnings'):
        tokenizer1.__dict__.pop('deprecation_warnings')
        tokenizer2.__dict__.pop('deprecation_warnings')

    # name_or_path will be the path that the tokenizer was loaded from, which will just be a temporary directory for
    # the reloaded tokenizer, so we remove it and don't compare it between the two tokenizers
    tokenizer1.__dict__.pop('name_or_path')
    tokenizer2.__dict__.pop('name_or_path')
    tokenizer1.init_kwargs.pop('name_or_path', None)
    tokenizer2.init_kwargs.pop('name_or_path', None)

    # The init_kwargs are not always the same between initial load and reload, even though the tokenizers are the same
    # and have the attributes set correctly. This section removes the keys that are different, only checking for equality if they
    # are present in both tokenizers
    model_max_length_1 = tokenizer1.init_kwargs.get('model_max_length', None)
    model_max_length_2 = tokenizer2.init_kwargs.get('model_max_length', None)
    if model_max_length_1 is not None and model_max_length_2 is not None:
        assert model_max_length_1 == model_max_length_2
    tokenizer1.__dict__['init_kwargs'].pop('model_max_length', None)
    tokenizer2.__dict__['init_kwargs'].pop('model_max_length', None)

    spaces_1 = tokenizer1.init_kwargs.get('clean_up_tokenization_spaces', None)
    spaces_2 = tokenizer2.init_kwargs.get('clean_up_tokenization_spaces', None)
    if spaces_1 is not None and spaces_2 is not None:
        assert spaces_1 == spaces_2
    tokenizer1.__dict__['init_kwargs'].pop('clean_up_tokenization_spaces', None)
    tokenizer2.__dict__['init_kwargs'].pop('clean_up_tokenization_spaces', None)

    tokenizer1.__dict__['init_kwargs'].pop('special_tokens_map_file', None)
    tokenizer2.__dict__['init_kwargs'].pop('special_tokens_map_file', None)

    # tokenizer.init_kwargs['tokenizer_file'] is unset when the tokenizer does not specify it, but is set to
    # None when you save and reload, so here we just check that its the same if it is present in both tokenizers.
    tokenizer_file_1 = tokenizer1.init_kwargs.get('tokenizer_file', None)
    tokenizer_file_2 = tokenizer2.init_kwargs.get('tokenizer_file', None)
    if tokenizer_file_1 is not None or tokenizer_file_2 is not None:
        assert tokenizer_file_1 == tokenizer_file_2

    tokenizer1.__dict__['init_kwargs'].pop('tokenizer_file', None)
    tokenizer2.__dict__['init_kwargs'].pop('tokenizer_file', None)
    tokenizer1.__dict__['init_kwargs'].pop('vocab_file', None)
    tokenizer2.__dict__['init_kwargs'].pop('vocab_file', None)

    # vocab_file will be the path that the tokenizer was loaded from, which will just be a temporary directory for
    # the reloaded tokenizer, so we remove it and don't compare it between the two tokenizers
    tokenizer1.__dict__.pop('vocab_file', None)
    tokenizer2.__dict__.pop('vocab_file', None)
    tokenizer1.__dict__.pop('special_tokens_map_file', None)
    tokenizer2.__dict__.pop('special_tokens_map_file', None)

    # The tokenizer name is changed in transformers 4.31 when changing the tokenizer mapping, so we remove it and compare
    # if necessary. Checks whether the names are subsets of each other.
    tokenizer1_name = tokenizer1.__dict__['init_kwargs'].get(
        'auto_map', {}).get('AutoTokenizer', [None])[0]
    tokenizer2_name = tokenizer2.__dict__['init_kwargs'].get(
        'auto_map', {}).get('AutoTokenizer', [None])[0]
    if tokenizer1_name is not None and tokenizer2_name is not None:
        assert tokenizer1_name in tokenizer2_name or tokenizer2_name in tokenizer1_name
    tokenizer1.__dict__['init_kwargs'].pop('auto_map', None)
    tokenizer2.__dict__['init_kwargs'].pop('auto_map', None)

    # Additional special tokens do not match between original tokenizer and loaded tokenizer due to transformers
    # constructor differences
    additional_special_tokens_1 = {
        t if isinstance(t, str) else t.content
        for t in tokenizer1.__dict__.pop('_additional_special_tokens', [])
    }
    additional_special_tokens_2 = {
        t if isinstance(t, str) else t.content
        for t in tokenizer2.__dict__.pop('_additional_special_tokens', [])
    }
    # Also pop it out of init_kwargs
    tokenizer1.__dict__['init_kwargs'].pop('additional_special_tokens', None)
    tokenizer2.__dict__['init_kwargs'].pop('additional_special_tokens', None)
    tokenizer1.__dict__['init_kwargs'].pop('added_tokens_decoder', None)
    tokenizer2.__dict__['init_kwargs'].pop('added_tokens_decoder', None)
    # If the additional special tokens are the same (or a subset of each other), or if one of them is empty, then we are good
    assert additional_special_tokens_1.issubset(
        additional_special_tokens_2) or additional_special_tokens_2.issubset(
            additional_special_tokens_1)

    # The special token attributes may be strings or they may be AddedToken objects, so we just check string values
    # First check that they have the same attrs
    assert tokenizer1.SPECIAL_TOKENS_ATTRIBUTES == tokenizer2.SPECIAL_TOKENS_ATTRIBUTES
    # Then check that the values are the same
    for special_token_attr in tokenizer1.SPECIAL_TOKENS_ATTRIBUTES:
        # Skip additional_special_tokens because we already checked it above
        if special_token_attr == 'additional_special_tokens':
            continue

        # The init_kwargs can change between the original tokenizer and the loaded tokenizer,
        # so we just pop them
        tokenizer1.__dict__['init_kwargs'].pop(special_token_attr, None)
        tokenizer2.__dict__['init_kwargs'].pop(special_token_attr, None)

        attr1 = tokenizer1.__dict__.pop('_' + special_token_attr, None)
        attr2 = tokenizer2.__dict__.pop('_' + special_token_attr, None)
        if attr1 is None and attr2 is None:
            continue

        attr_value1 = attr1 if isinstance(attr1, str) else attr1.content
        attr_value2 = attr2 if isinstance(attr2, str) else attr2.content
        assert attr_value1 == attr_value2

    assert tokenizer1.__dict__ == tokenizer2.__dict__


def check_hf_model_equivalence(model1: PreTrainedModel,
                               model2: PreTrainedModel):
    expected_model_config_dict = model1.config.to_dict()
    new_model_config_dict = model2.config.to_dict()

    # _name_or_path is different depending on whether the model was loaded from disk or the hub,
    # so we remove it
    expected_model_config_dict.pop('_name_or_path')
    new_model_config_dict.pop('_name_or_path')

    # Special case a couple of differences that correctly occur when saving MPT to huggingface format
    # checkpoint
    architectures_1 = expected_model_config_dict.pop('architectures', None)
    architectures_2 = new_model_config_dict.pop('architectures', None)
    if architectures_1 != architectures_2:
        assert architectures_1 is None and architectures_2 == ['MPTForCausalLM']

    auto_map_1 = expected_model_config_dict.pop('auto_map', None)
    auto_map_2 = new_model_config_dict.pop('auto_map', None)
    if auto_map_1 != auto_map_2:
        assert auto_map_1 == {'AutoConfig': 'configuration_mpt.MPTConfig'}
        assert auto_map_2 == {
            'AutoConfig': 'configuration_mpt.MPTConfig',
            'AutoModelForCausalLM': 'modeling_mpt.MPTForCausalLM'
        }

    assert expected_model_config_dict == new_model_config_dict
    assert all(
        torch.equal(p1.cpu(), p2.cpu())
        for p1, p2 in zip(model1.parameters(), model2.parameters()))


def delete_transformers_cache():
    # Only delete the files on local rank 0, otherwise race conditions are created
    if not dist.get_local_rank() == 0:
        return

    hf_cache_home = os.path.expanduser(
        os.getenv(
            'HF_HOME',
            os.path.join(os.getenv('XDG_CACHE_HOME', '~/.cache'),
                         'huggingface')))
    HF_MODULES_CACHE = os.getenv('HF_MODULES_CACHE',
                                 os.path.join(hf_cache_home, 'modules'))
    if os.path.exists(HF_MODULES_CACHE) and os.path.isdir(HF_MODULES_CACHE):
        shutil.rmtree(HF_MODULES_CACHE)


def get_config(
        conf_path: str = 'scripts/train/yamls/pretrain/testing.yaml'
) -> DictConfig:
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    with open(conf_path) as f:
        test_cfg = om.load(f)

    return cast(DictConfig, test_cfg)


def test_callback_inits():
    # test with defaults
    _ = HuggingFaceCheckpointer(save_folder='test', save_interval='1ba')

    # test default metatdata when mlflow registered name is given
    hf_checkpointer = HuggingFaceCheckpointer(
        save_folder='test',
        save_interval='1ba',
        mlflow_registered_model_name='test_model_name')
    assert hf_checkpointer.mlflow_logging_config == {
        'task': 'text-generation',
        'metadata': {
            'task': 'llm/v1/completions'
        }
    }


@pytest.mark.gpu
@pytest.mark.parametrize('log_to_mlflow', [True, False])
@pytest.mark.parametrize(
    'hf_save_interval,save_interval,max_duration,expected_hf_checkpoints,expected_normal_checkpoints',
    [('3ba', '2ba', '4ba', 2, 2), ('1dur', '2ba', '1ep', 1, 2)])
@patch('os.cpu_count', MagicMock(return_value=1))
def test_huggingface_conversion_callback_interval(
        tmp_path: pathlib.Path, log_to_mlflow: bool, hf_save_interval: str,
        save_interval: str, max_duration: str, expected_hf_checkpoints: int,
        expected_normal_checkpoints: int, tiny_ft_dataloader: DataLoader,
        mpt_tokenizer: PreTrainedTokenizerBase, build_tiny_mpt: Callable):
    delete_transformers_cache()

    dist.initialize_dist(get_device('gpu'))

    device_batch_size = 1
    dataset_size = 4
    precision_str = 'bfloat16'
    precision = torch.bfloat16
    batches_per_epoch = math.ceil(dataset_size / device_batch_size)

    checkpointer_callback = HuggingFaceCheckpointer(
        save_folder=os.path.join(tmp_path, 'checkpoints'),
        save_interval=hf_save_interval,
        precision=precision_str,
        mlflow_registered_model_name='dummy-registered-name'
        if log_to_mlflow else None,
    )

    original_model = build_tiny_mpt()

    optimizer_config = {
        'name': 'decoupled_adamw',
        'lr': 6e-4,
        'betas': [0.9, 0.95],
        'eps': 1e-8,
        'weight_decay': 0.0,
    }
    optimizer_name = optimizer_config.pop('name')
    optimizer = build_optimizer(original_model, optimizer_name,
                                optimizer_config)

    mlflow_logger_mock = MagicMock(spec=MLFlowLogger)
    mlflow_logger_mock.state_dict = lambda *args, **kwargs: {}
    mlflow_logger_mock.save_model = MagicMock(wraps=_save_model_mock)
    mlflow_logger_mock.register_model = MagicMock()
    mlflow_logger_mock.model_registry_prefix = ''
    mlflow_logger_mock._experiment_id = 'mlflow-experiment-id'
    mlflow_logger_mock._run_id = 'mlflow-run-id'
    trainer = Trainer(
        model=original_model,
        device='gpu',
        train_dataloader=tiny_ft_dataloader,
        save_folder=os.path.join(tmp_path, 'checkpoints'),
        save_interval=save_interval,
        max_duration=max_duration,
        callbacks=[checkpointer_callback],
        loggers=[mlflow_logger_mock] if log_to_mlflow else [],
        optimizers=optimizer,
        save_latest_filename=None,
    )
    trainer.fit()

    if log_to_mlflow:
        assert mlflow_logger_mock.save_model.call_count == 1
        mlflow_logger_mock.save_model.assert_called_with(
            flavor='transformers',
            transformers_model=ANY,
            path=ANY,
            task='text-generation',
            metadata={'task': 'llm/v1/completions'})
        assert mlflow_logger_mock.register_model.call_count == 1
    else:
        assert mlflow_logger_mock.save_model.call_count == 0
        assert mlflow_logger_mock.register_model.call_count == 0

    normal_checkpoints = [
        name for name in os.listdir(os.path.join(tmp_path, 'checkpoints'))
        if name != 'huggingface'
    ]
    huggingface_checkpoints = [
        name for name in os.listdir(
            os.path.join(tmp_path, 'checkpoints', 'huggingface'))
    ]
    assert len(normal_checkpoints) == expected_normal_checkpoints
    assert len(huggingface_checkpoints) == expected_hf_checkpoints

    # Load the last huggingface checkpoint
    loaded_model = transformers.AutoModelForCausalLM.from_pretrained(
        os.path.join(tmp_path, 'checkpoints', 'huggingface',
                     f'ba{batches_per_epoch}'),
        trust_remote_code=True,
    )

    # Check that the loaded model has the correct precision, and then set it back
    # to the original for the equivalence check
    assert loaded_model.config.torch_dtype == precision
    loaded_model.config.torch_dtype = original_model.model.config.torch_dtype

    # Check that we have correctly set these attributes, and then set them back
    # to the original for the equivalence check
    assert loaded_model.config.attn_config['attn_impl'] == 'torch'
    assert loaded_model.config.init_device == 'cpu'
    loaded_model.config.attn_config[
        'attn_impl'] = original_model.model.config.attn_config['attn_impl']
    loaded_model.config.init_device = original_model.model.config.init_device

    loaded_tokenizer = transformers.AutoTokenizer.from_pretrained(
        os.path.join(tmp_path, 'checkpoints', 'huggingface',
                     f'ba{batches_per_epoch}'),
        trust_remote_code=True,
    )

    check_hf_model_equivalence(trainer.state.model.model.to(precision),
                               loaded_model)
    check_hf_tokenizer_equivalence(mpt_tokenizer, loaded_tokenizer)

    delete_transformers_cache()


@pytest.mark.world_size(2)
@pytest.mark.gpu
@pytest.mark.parametrize(
    'model,tie_word_embeddings',
    [('mpt', True), ('mpt', False), ('neo', None), ('llama2', None)],
)
@pytest.mark.parametrize('fsdp_state_dict_type', ['full', 'sharded', None])
@pytest.mark.parametrize(
    'hf_save_interval,save_interval,max_duration,expected_hf_checkpoints,expected_normal_checkpoints',
    [('1ba', '1ba', '1ba', 1, 1)])
@patch('os.cpu_count', MagicMock(return_value=1))
def test_huggingface_conversion_callback(
    model: str,
    tmp_path: pathlib.Path,
    tie_word_embeddings: bool,
    fsdp_state_dict_type: Optional[str],
    hf_save_interval: str,
    save_interval: str,
    max_duration: str,
    expected_hf_checkpoints: int,
    expected_normal_checkpoints: int,
):
    delete_transformers_cache()

    dist.initialize_dist(get_device('gpu'))

    max_seq_len = 16
    device_batch_size = 1
    dataset_size = 2
    precision_str = 'bfloat16'
    precision = torch.bfloat16
    batches_per_epoch = math.ceil(dataset_size / (device_batch_size * 2))

    checkpointer_callback = HuggingFaceCheckpointer(
        save_folder=os.path.join(tmp_path, 'checkpoints'),
        save_interval=hf_save_interval,
        precision=precision_str,
        mlflow_registered_model_name='dummy-registered-name')

    # get small version of each model
    model_cfg = None
    tokenizer_name = None
    if model == 'mpt':
        model_cfg = {
            'name': 'mpt_causal_lm',
            'init_device': 'cpu',
            'd_model': 64,
            'n_heads': 2,
            'n_layers': 2,
            'expansion_ratio': 4,
            'max_seq_len': max_seq_len,
            'vocab_size': 50368,
            'attn_config': {
                'attn_impl': 'torch',
            },
            'loss_fn': 'torch_crossentropy',
            'tie_word_embeddings': tie_word_embeddings,
        }
        tokenizer_name = 'EleutherAI/gpt-neox-20b'
    elif model == 'neo':
        assert tie_word_embeddings is None
        model_cfg = {
            'name': 'hf_causal_lm',
            'pretrained_model_name_or_path': 'EleutherAI/gpt-neo-125M',
            'config_overrides': {
                'max_position_embeddings': max_seq_len,
                'hidden_size': 36,
            },
            'pretrained': False,
            'init_device': 'cpu',
        }
        tokenizer_name = 'EleutherAI/gpt-neo-125M'
    elif model == 'llama2':
        assert tie_word_embeddings is None
        if 'HUGGING_FACE_HUB_TOKEN' not in os.environ:
            pytest.skip(
                'The CI cluster does not have access to the Llama models, so skip this test.'
            )
        model_cfg = {
            'name': 'hf_causal_lm',
            'pretrained_model_name_or_path': 'meta-llama/Llama-2-7b-hf',
            'config_overrides': {
                'num_hidden_layers': 2,
                'hidden_size': 32,
                'intermediate_size': 64,
            },
            'use_auth_token': True,
            'pretrained': False,
            'init_device': 'cpu',
        }
        tokenizer_name = 'meta-llama/Llama-2-7b-hf'
    else:
        raise ValueError(f'Unknown model {model}')
    assert model_cfg is not None
    assert tokenizer_name is not None
    model_cfg = om.create(model_cfg)

    fsdp_config = {
        'sharding_strategy': 'FULL_SHARD',
        'mixed_precision': 'PURE',
        'activation_checkpointing': False,
        'activation_checkpointing_reentrant': False,
        'activation_cpu_offload': False,
        'limit_all_gathers': True,
        'state_dict_type': fsdp_state_dict_type,
    }

    tiny_dataset_folder_path = os.path.join(os.getcwd(), 'test-ift-data-small')
    tiny_dataset_path = os.path.join(tiny_dataset_folder_path, 'train.jsonl')
    if dist.get_global_rank() == 0:
        make_tiny_ft_dataset(path=tiny_dataset_path, size=dataset_size)

    dataloader_cfg = {
        'name': 'finetuning',
        'dataset': {
            'hf_name': tiny_dataset_folder_path,
            'split': 'train',
            'max_seq_len': max_seq_len,
            'decoder_only_format': True,
            'allow_pad_trimming': False,
            'packing_ratio': None,
            'shuffle': True,
        },
        'drop_last': False,
        'num_workers': 0,
        'pin_memory': False,
        'prefetch_factor': None,
        'persistent_workers': False,
        'timeout': 0
    }

    dataloader_cfg = om.create(dataloader_cfg)

    tokenizer = build_tokenizer(
        tokenizer_name=tokenizer_name,
        tokenizer_kwargs={'model_max_length': max_seq_len},
    )

    train_dataloader = build_finetuning_dataloader(
        dataloader_cfg,
        tokenizer,
        device_batch_size,
    )

    original_model = COMPOSER_MODEL_REGISTRY[model_cfg['name']](model_cfg,
                                                                tokenizer)

    optimizer_config = {
        'name': 'decoupled_adamw',
        'lr': 6e-4,
        'betas': [0.9, 0.95],
        'eps': 1e-8,
        'weight_decay': 0.0,
    }
    optimizer_name = optimizer_config.pop('name')
    optimizer = build_optimizer(original_model, optimizer_name,
                                optimizer_config)

    mlflow_logger_mock = MagicMock(spec=MLFlowLogger)
    mlflow_logger_mock.state_dict = lambda *args, **kwargs: {}
    mlflow_logger_mock.save_model = MagicMock(wraps=_save_model_mock)
    mlflow_logger_mock.register_model = MagicMock()
    mlflow_logger_mock.model_registry_prefix = ''
    mlflow_logger_mock._experiment_id = 'mlflow-experiment-id'
    mlflow_logger_mock._run_id = 'mlflow-run-id'
    trainer = Trainer(
        model=original_model,
        device='gpu',
        fsdp_config=fsdp_config if fsdp_state_dict_type is not None else None,
        train_dataloader=train_dataloader,
        save_folder=os.path.join(tmp_path, 'checkpoints'),
        save_interval=save_interval,
        max_duration=max_duration,
        callbacks=[checkpointer_callback],
        loggers=[mlflow_logger_mock],
        optimizers=optimizer,
        save_latest_filename=None,
    )
    trainer.fit()

    if dist.get_global_rank() == 0:
        assert mlflow_logger_mock.save_model.call_count == 1
        mlflow_logger_mock.save_model.assert_called_with(
            flavor='transformers',
            transformers_model=ANY,
            path=ANY,
            task='text-generation',
            metadata={'task': 'llm/v1/completions'})
        assert mlflow_logger_mock.register_model.call_count == 1
    else:
        assert mlflow_logger_mock.log_model.call_count == 0
        assert mlflow_logger_mock.register_model.call_count == 0

    # summon full params to check equivalence
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    with FSDP.summon_full_params(trainer.state.model,
                                 writeback=False,
                                 recurse=True):
        loaded_model = None
        loaded_tokenizer = None
        # Only rank zero is saving the huggingface checkpoints, so only check
        # for equivalence on rank zero
        if dist.get_global_rank() == 0:
            normal_checkpoints = [
                name
                for name in os.listdir(os.path.join(tmp_path, 'checkpoints'))
                if name != 'huggingface'
            ]
            huggingface_checkpoints = [
                name for name in os.listdir(
                    os.path.join(tmp_path, 'checkpoints', 'huggingface'))
            ]
            assert len(normal_checkpoints) == expected_normal_checkpoints
            assert len(huggingface_checkpoints) == expected_hf_checkpoints

            # Patch flash_attn package to be empty to simulate loading the model in
            # an environment without flash atttention installed
            with patch.dict('sys.modules', {'flash_attn': None}):
                # Load the last huggingface checkpoint
                loaded_model = transformers.AutoModelForCausalLM.from_pretrained(
                    os.path.join(tmp_path, 'checkpoints', 'huggingface',
                                 f'ba{batches_per_epoch}'),
                    trust_remote_code=True,
                )

            # Check that the loaded model has the correct precision, and then set it back
            # to the original for the equivalence check
            assert loaded_model.config.torch_dtype == precision
            loaded_model.config.torch_dtype = original_model.model.config.torch_dtype

            if model == 'mpt':
                # Check that we have correctly set these attributes, and then set them back
                # to the original for the equivalence check
                assert loaded_model.config.attn_config['attn_impl'] == 'torch'
                assert loaded_model.config.init_device == 'cpu'
                loaded_model.config.attn_config[
                    'attn_impl'] = original_model.model.config.attn_config[
                        'attn_impl']
                loaded_model.config.init_device = original_model.model.config.init_device

            loaded_tokenizer = transformers.AutoTokenizer.from_pretrained(
                os.path.join(tmp_path, 'checkpoints', 'huggingface',
                             f'ba{batches_per_epoch}'),
                trust_remote_code=True,
            )

            check_hf_model_equivalence(
                trainer.state.model.model.to(precision) if fsdp_state_dict_type
                is not None else trainer.state.model.module.model.to(precision),
                loaded_model)
            check_hf_tokenizer_equivalence(tokenizer, loaded_tokenizer)

    dist.barrier()
    delete_transformers_cache()


@pytest.mark.parametrize(
    'model,tie_word_embeddings',
    [('mpt', True), ('mpt', False), ('neo', None), ('llama2', None)],
)
def test_convert_and_generate(model: str, tie_word_embeddings: bool,
                              tmp_path: pathlib.Path):
    delete_transformers_cache()

    om_cfg = None
    if model == 'mpt':
        om_cfg = get_config(
            conf_path='scripts/train/yamls/pretrain/testing.yaml')
        om_cfg['tie_word_embeddings'] = tie_word_embeddings
    elif model == 'neo':
        assert tie_word_embeddings is None
        om_cfg = get_config(
            conf_path='scripts/train/yamls/pretrain/gpt-neo-125m.yaml')
        om_cfg['model']['config_overrides']['hidden_size'] = 36
    elif model == 'llama2':
        assert tie_word_embeddings is None
        if 'HUGGING_FACE_HUB_TOKEN' not in os.environ:
            pytest.skip(
                'The CI cluster does not have access to the Llama models, so skip this test.'
            )
        om_cfg = get_config(
            conf_path='scripts/train/yamls/pretrain/gpt-neo-125m.yaml')
        om_cfg['model'][
            'pretrained_model_name_or_path'] = 'meta-llama/Llama-2-7b-hf'
        om_cfg['model']['config_overrides']['num_hidden_layers'] = 2
        om_cfg['model']['use_auth_token'] = True
        om_cfg['tokenizer']['name'] = 'meta-llama/Llama-2-7b-hf'
    else:
        raise ValueError(f'Unknown model {model}')
    assert om_cfg is not None

    om_cfg['model']['init_device'] = 'cpu'
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        om_cfg.tokenizer.name, use_auth_token=model == 'llama2')
    original_model = COMPOSER_MODEL_REGISTRY[om_cfg['model'].name](
        om_cfg['model'], tokenizer)
    trainer = Trainer(model=original_model, device='cpu')
    trainer.save_checkpoint(os.path.join(tmp_path, 'checkpoint.pt'))

    args = Namespace(composer_path=os.path.join(tmp_path, 'checkpoint.pt'),
                     hf_output_path=os.path.join(tmp_path, 'hf-output-folder'),
                     output_precision='fp32',
                     local_checkpoint_save_location=None,
                     hf_repo_for_upload=None,
                     trust_remote_code=False,
                     test_uploaded_model=False)
    convert_composer_to_hf(args)

    loaded_config = transformers.AutoConfig.from_pretrained(
        os.path.join(tmp_path, 'hf-output-folder'), trust_remote_code=True)
    loaded_model = transformers.AutoModelForCausalLM.from_pretrained(
        os.path.join(tmp_path, 'hf-output-folder'),
        config=loaded_config,
        trust_remote_code=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        os.path.join(tmp_path, 'hf-output-folder'), trust_remote_code=True)

    output = loaded_model.generate(tokenizer('hello',
                                             return_tensors='pt')['input_ids'],
                                   max_new_tokens=1)
    assert output.shape == (1, 2 + (1 if model == 'llama2' else 0))

    assert sum(p.numel() for p in original_model.model.parameters()) == sum(
        p.numel() for p in loaded_model.parameters())
    assert all(
        str(type(module1)).split('.')[-1] == str(type(module2)).split('.')[-1]
        for module1, module2 in zip(original_model.model.modules(),
                                    loaded_model.modules()))
    for p1, p2 in zip(original_model.model.parameters(),
                      loaded_model.parameters()):
        assert torch.allclose(p1, p2)

    delete_transformers_cache()


@pytest.mark.gpu
@pytest.mark.parametrize('tie_word_embeddings', [True, False])
def test_convert_and_generate_triton(tie_word_embeddings: str,
                                     tmp_path: pathlib.Path):
    delete_transformers_cache()

    cfg = get_config()
    cfg['model']['init_device'] = 'cpu'
    cfg['tie_word_embeddings'] = tie_word_embeddings
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        'EleutherAI/gpt-neox-20b')
    model = ComposerMPTCausalLM(cfg['model'], tokenizer)
    trainer = Trainer(model=model)
    trainer.save_checkpoint(os.path.join(tmp_path, 'checkpoint.pt'))

    args = Namespace(composer_path=os.path.join(tmp_path, 'checkpoint.pt'),
                     hf_output_path=os.path.join(tmp_path, 'hf-output-folder'),
                     output_precision='fp32',
                     local_checkpoint_save_location=None,
                     hf_repo_for_upload=None,
                     trust_remote_code=False,
                     test_uploaded_model=False)
    convert_composer_to_hf(args)

    config = transformers.AutoConfig.from_pretrained(os.path.join(
        tmp_path, 'hf-output-folder'),
                                                     trust_remote_code=True)
    config.attn_config['attn_impl'] = 'triton'
    model = transformers.AutoModelForCausalLM.from_pretrained(
        os.path.join(tmp_path, 'hf-output-folder'),
        config=config,
        trust_remote_code=True)
    model.to(device='cuda', dtype=torch.bfloat16)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        os.path.join(tmp_path, 'hf-output-folder'), trust_remote_code=True)

    output = model.generate(tokenizer(
        'hello', return_tensors='pt')['input_ids'].to(device='cuda'),
                            max_new_tokens=1)
    assert output.shape == (1, 2)

    delete_transformers_cache()


@pytest.mark.parametrize('tie_word_embeddings', [True, False])
def test_convert_and_generate_meta(tie_word_embeddings: str,
                                   tmp_path: pathlib.Path):
    delete_transformers_cache()

    from composer.utils import dist
    gathered_paths = dist.all_gather_object(tmp_path)
    tmp_path_gathered = gathered_paths[0]

    om_cfg = get_config(conf_path='scripts/train/yamls/pretrain/testing.yaml')

    om_cfg['model']['init_device'] = 'cpu'
    om_cfg['tie_word_embeddings'] = tie_word_embeddings
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        om_cfg.tokenizer.name)
    original_model = COMPOSER_MODEL_REGISTRY[om_cfg['model'].name](
        om_cfg['model'], tokenizer)
    trainer = Trainer(model=original_model, device='cpu')
    trainer.save_checkpoint(os.path.join(tmp_path_gathered, 'checkpoint.pt'))

    # patch in the meta device for testing
    sd = torch.load(os.path.join(tmp_path_gathered, 'checkpoint.pt'),
                    map_location='cpu')
    sd['state']['integrations']['huggingface']['model']['config']['content'][
        'init_device'] = 'meta'
    torch.save(sd, os.path.join(tmp_path_gathered, 'checkpoint.pt'))

    args = Namespace(composer_path=os.path.join(tmp_path_gathered,
                                                'checkpoint.pt'),
                     hf_output_path=os.path.join(tmp_path_gathered,
                                                 'hf-output-folder'),
                     output_precision='fp32',
                     local_checkpoint_save_location=None,
                     hf_repo_for_upload=None,
                     trust_remote_code=False,
                     test_uploaded_model=False)
    convert_composer_to_hf(args)

    loaded_config = transformers.AutoConfig.from_pretrained(
        os.path.join(tmp_path_gathered, 'hf-output-folder'),
        trust_remote_code=True)
    loaded_model = transformers.AutoModelForCausalLM.from_pretrained(
        os.path.join(tmp_path_gathered, 'hf-output-folder'),
        config=loaded_config,
        trust_remote_code=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        os.path.join(tmp_path_gathered, 'hf-output-folder'),
        trust_remote_code=True)

    output = loaded_model.generate(tokenizer('hello',
                                             return_tensors='pt')['input_ids'],
                                   max_new_tokens=1)
    assert output.shape == (1, 2)

    assert sum(p.numel() for p in original_model.model.parameters()) == sum(
        p.numel() for p in loaded_model.parameters())
    assert all(
        str(type(module1)).split('.')[-1] == str(type(module2)).split('.')[-1]
        for module1, module2 in zip(original_model.model.modules(),
                                    loaded_model.modules()))
    for p1, p2 in zip(original_model.model.parameters(),
                      loaded_model.parameters()):
        assert torch.allclose(p1, p2)

    delete_transformers_cache()


@pytest.mark.parametrize(
    'license_file_name',
    ['LICENSE', 'LICENSE.txt', 'license', 'license.md', None])
def test_license_file_finder(tmp_path: pathlib.Path,
                             license_file_name: Optional[str]):
    if license_file_name is not None:
        with open(os.path.join(tmp_path, license_file_name), 'w') as f:
            f.write('test')

    found_path = _maybe_get_license_filename(str(tmp_path))
    assert (found_path == license_file_name
           ) if license_file_name is not None else (found_path is None)
