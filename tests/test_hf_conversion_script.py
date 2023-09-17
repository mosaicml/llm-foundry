# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import math
import os
import pathlib
import sys

from composer import Trainer
from composer.utils import dist, get_device

from llmfoundry.callbacks import HuggingFaceCheckpointer
from llmfoundry.models.mpt.modeling_mpt import ComposerMPTCausalLM

# Add repo root to path so we can import scripts and test it
repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(repo_dir)
import shutil
from argparse import Namespace
from typing import cast

import pytest
import torch
import transformers
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from llmfoundry import COMPOSER_MODEL_REGISTRY
from llmfoundry.data.finetuning import build_finetuning_dataloader
from llmfoundry.utils.builders import build_optimizer, build_tokenizer
from scripts.inference.convert_composer_to_hf import convert_composer_to_hf
from tests.data_utils import make_tiny_ft_dataset


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

    assert tokenizer1.__dict__ == tokenizer2.__dict__


def check_hf_model_equivalence(model1: PreTrainedModel,
                               model2: PreTrainedModel):
    expected_model_config_dict = model1.config.to_dict()
    new_model_config_dict = model2.config.to_dict()

    # _name_or_path is different depending on whether the model was loaded from disk or the hub,
    # so we remove it
    expected_model_config_dict.pop('_name_or_path')
    new_model_config_dict.pop('_name_or_path')
    assert expected_model_config_dict == new_model_config_dict
    assert all(
        torch.equal(p1.cpu(), p2.cpu())
        for p1, p2 in zip(model1.parameters(), model2.parameters()))


def delete_transformers_cache():
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


def test_callback_inits_with_defaults():
    _ = HuggingFaceCheckpointer(save_folder='test', save_interval='1ba')


@pytest.mark.world_size(2)
@pytest.mark.gpu
@pytest.mark.parametrize('model', ['mpt', 'neo', 'llama2'])
@pytest.mark.parametrize('fsdp_state_dict_type', ['full', 'sharded'])
def test_huggingface_conversion_callback(model: str, tmp_path: pathlib.Path,
                                         fsdp_state_dict_type: str):
    delete_transformers_cache()

    dist.initialize_dist(get_device('gpu'))

    max_seq_len = 16
    save_interval_batches = 2
    huggingface_save_interval_batches = 3
    device_batch_size = 1
    dataset_size = 14
    max_duration_batches = 7
    precision_str = 'bfloat16'
    precision = torch.bfloat16

    checkpointer_callback = HuggingFaceCheckpointer(
        save_folder=os.path.join(tmp_path, 'checkpoints'),
        save_interval=f'{huggingface_save_interval_batches}ba',
        precision=precision_str,
    )

    # get small version of each model
    model_cfg = None
    tokenizer_name = None
    if model == 'mpt':
        model_cfg = {
            'name': 'mpt_causal_lm',
            'init_device': 'cpu',
            'd_model': 128,
            'n_heads': 2,
            'n_layers': 2,
            'expansion_ratio': 4,
            'max_seq_len': max_seq_len,
            'vocab_size': 50368,
            'attn_config': {
                'attn_impl': 'torch',
            },
            'loss_fn': 'torch_crossentropy',
        }
        tokenizer_name = 'EleutherAI/gpt-neox-20b'
    elif model == 'neo':
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

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_name, use_auth_token=model == 'llama2')

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
        'num_workers': 4,
        'pin_memory': False,
        'prefetch_factor': 2,
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

    trainer = Trainer(
        model=original_model,
        device='gpu',
        fsdp_config=fsdp_config,
        train_dataloader=train_dataloader,
        save_folder=os.path.join(tmp_path, 'checkpoints'),
        save_interval=f'{save_interval_batches}ba',
        max_duration=f'{max_duration_batches}ba',
        callbacks=[checkpointer_callback],
        optimizers=optimizer,
        save_latest_filename=None,
    )
    trainer.fit()

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
            assert len(normal_checkpoints) == math.ceil(max_duration_batches /
                                                        save_interval_batches)
            assert len(huggingface_checkpoints) == math.ceil(
                max_duration_batches / huggingface_save_interval_batches)

            # Load the last huggingface checkpoint
            loaded_model = transformers.AutoModelForCausalLM.from_pretrained(
                os.path.join(tmp_path, 'checkpoints', 'huggingface',
                             f'ba{max_duration_batches}'),
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
                             f'ba{max_duration_batches}'),
                trust_remote_code=True,
            )

            check_hf_model_equivalence(trainer.state.model.model.to(precision),
                                       loaded_model)
            check_hf_tokenizer_equivalence(tokenizer, loaded_tokenizer)

    delete_transformers_cache()


@pytest.mark.parametrize('model', ['mpt', 'neo', 'llama2'])
def test_convert_and_generate(model: str, tmp_path: pathlib.Path):
    delete_transformers_cache()

    om_cfg = None
    if model == 'mpt':
        om_cfg = get_config(
            conf_path='scripts/train/yamls/pretrain/testing.yaml')
    elif model == 'neo':
        om_cfg = get_config(
            conf_path='scripts/train/yamls/pretrain/gpt-neo-125m.yaml')
        om_cfg['model']['config_overrides']['hidden_size'] = 36
    elif model == 'llama2':
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
def test_convert_and_generate_triton(tmp_path: pathlib.Path):
    delete_transformers_cache()

    cfg = get_config()
    cfg['model']['init_device'] = 'cpu'
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


def test_convert_and_generate_meta(tmp_path: pathlib.Path):
    delete_transformers_cache()

    from composer.utils import dist
    gathered_paths = dist.all_gather_object(tmp_path)
    tmp_path_gathered = gathered_paths[0]

    om_cfg = get_config(conf_path='scripts/train/yamls/pretrain/testing.yaml')

    om_cfg['model']['init_device'] = 'cpu'
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
