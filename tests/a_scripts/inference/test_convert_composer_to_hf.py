# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import glob
import json
import math
import os
import pathlib
import shutil
from argparse import Namespace
from typing import Any, Callable, Optional, Union, cast
from unittest import mock
from unittest.mock import ANY, MagicMock, patch

import catalogue
import numpy as np
import pytest
import torch
import torch.nn as nn
import transformers
from composer import ComposerModel, Trainer
from composer.core import State
from composer.loggers import Logger, MLFlowLogger
from composer.utils import dist, get_device
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from torch.distributed._tensor.api import DTensor
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    GenerationConfig,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from llmfoundry.callbacks import HuggingFaceCheckpointer
from llmfoundry.callbacks.hf_checkpointer import _maybe_get_license_filename
from llmfoundry.data.finetuning import build_finetuning_dataloader
from llmfoundry.models.mpt import MPTConfig, MPTForCausalLM
from llmfoundry.utils import edit_files_for_hf_compatibility
from llmfoundry.utils.builders import (
    build_composer_model,
    build_optimizer,
)
from llmfoundry.utils.config_utils import process_init_device, to_dict_container
from scripts.inference.convert_composer_to_hf import convert_composer_to_hf
from tests.data_utils import make_tiny_ft_dataset
from tests.fixtures.models import get_tokenizer_fixture_by_name

_OPTIMIZER_CFG = lambda: {
    'name': 'decoupled_adamw',
    'lr': 6e-4,
    'betas': [0.9, 0.95],
    'eps': 1e-8,
    'weight_decay': 0.0,
}


def _save_model_mock(*args: Any, path: str, **kwargs: Any):
    os.makedirs(path, exist_ok=True)


def check_hf_tokenizer_equivalence(
    tokenizer1: PreTrainedTokenizerBase,
    tokenizer2: PreTrainedTokenizerBase,
):
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
        type(tokenizer2),
    ).split('.')[-2:]

    expected_tokenizer_output = tokenizer2(
        'This is some text that should get tokenizer !? @ totallyarealtoken',
    )
    actual_tokenizer_output = tokenizer1(
        'This is some text that should get tokenizer !? @ totallyarealtoken',
    )
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
    if hasattr(
        tokenizer1,
        'deprecation_warnings',
    ) or hasattr(tokenizer2, 'deprecation_warnings'):
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

    # tokenizer.init_kwargs['merges_file'] is set when loading with AutoTokenizer.from_pretrained, but is set to
    # None when you save and reload the tokenizer only.
    # Otherwise, merges_file will be the path that the tokenizer was loaded from, which will just be a temporary directory for
    # the reloaded tokenizer, so we remove it and don't compare it between the two tokenizers.
    tokenizer1.__dict__['init_kwargs'].pop('merges_file', None)
    tokenizer2.__dict__['init_kwargs'].pop('merges_file', None)

    # vocab_file will be the path that the tokenizer was loaded from, which will just be a temporary directory for
    # the reloaded tokenizer, so we remove it and don't compare it between the two tokenizers
    tokenizer1.__dict__.pop('vocab_file', None)
    tokenizer2.__dict__.pop('vocab_file', None)
    tokenizer1.__dict__.pop('special_tokens_map_file', None)
    tokenizer2.__dict__.pop('special_tokens_map_file', None)

    # The tokenizer name is changed in transformers 4.31 when changing the tokenizer mapping, so we remove it and compare
    # if necessary. Checks whether the names are subsets of each other.
    tokenizer1_name = tokenizer1.__dict__['init_kwargs'].get(
        'auto_map',
        {},
    ).get('AutoTokenizer', [None])[0]
    tokenizer2_name = tokenizer2.__dict__['init_kwargs'].get(
        'auto_map',
        {},
    ).get('AutoTokenizer', [None])[0]
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
        additional_special_tokens_2,
    ) or additional_special_tokens_2.issubset(additional_special_tokens_1)

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

        # Handle the case when the attribute is an AddedToken object
        attr_value1 = attr1 if isinstance(
            attr1,
            str,
        ) else attr1.content if hasattr(  # type: ignore
            attr1,
            'content',
        ) else str(attr1)
        attr_value2 = attr2 if isinstance(
            attr2,
            str,
        ) else attr2.content if hasattr(  # type: ignore
            attr2,
            'content',
        ) else str(attr2)
        assert attr_value1 == attr_value2

    # Ignore 'extra_special_tokens' as it was added by the transformers library during save/load
    if 'extra_special_tokens' in tokenizer2.init_kwargs and 'extra_special_tokens' not in tokenizer1.init_kwargs:
        tokenizer2.init_kwargs.pop('extra_special_tokens')

    # Process special tokens map and added tokens decoder
    for dict_map_key in ['_special_tokens_map', '_added_tokens_decoder']:
        if dict_map_key in tokenizer1.__dict__ and dict_map_key in tokenizer2.__dict__:
            # Get the nested dictionaries
            token_map1 = tokenizer1.__dict__[dict_map_key]
            token_map2 = tokenizer2.__dict__[dict_map_key]

            # Process values in the first tokenizer's map
            for key in list(token_map1.keys()):
                if hasattr(token_map1[key], 'content'):
                    token_map1[key] = token_map1[key].content

            # Process values in the second tokenizer's map
            for key in list(token_map2.keys()):
                if hasattr(token_map2[key], 'content'):
                    token_map2[key] = token_map2[key].content

    # Final comparison of dictionaries
    t1_dict = tokenizer1.__dict__
    t2_dict = tokenizer2.__dict__

    assert t1_dict == t2_dict, 'Tokenizer dictionaries are not equal'


def remove_moe_world_size(config: MPTConfig):
    if hasattr(config, 'ffn_config'):
        if 'moe_world_size' in config.ffn_config:
            config.ffn_config.pop('moe_world_size')


def check_hf_model_equivalence(
    model1: PreTrainedModel,
    model2: PreTrainedModel,
    just_lora: bool = False,
):
    remove_moe_world_size(model1.config)  # type: ignore
    remove_moe_world_size(model2.config)  # type: ignore

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
            'AutoModelForCausalLM': 'modeling_mpt.MPTForCausalLM',
        }

    assert expected_model_config_dict == new_model_config_dict
    for (n1, p1), (
        _,
        p2,
    ) in zip(model1.named_parameters(), model2.named_parameters()):
        if not just_lora or 'lora' in n1:
            assert torch.equal(p1.cpu(), p2.cpu())


def check_safetensors_precision(
    model_path: str,
    model: torch.nn.Module,
    expected_precision: torch.dtype,
    tolerance: float = 0.2,
):
    """Verify that the safetensors files in model_path have the expected size.

    Args:
        model_path: Path to the directory containing the safetensors files
        model: The original model to count parameters from
        expected_precision: The expected precision (torch.float32, torch.bfloat16, etc.)
        tolerance: Allowed deviation from expected file size (as a ratio)

    Returns:
        bool: True if the safetensors files have the expected size, False otherwise
    """
    total_params = sum(p.numel() for p in model.parameters())
    # Determine expected bytes per parameter based on precision
    bytes_per_param = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int8: 1,
    }.get(expected_precision)
    assert bytes_per_param

    expected_size = total_params * bytes_per_param

    safetensors_files = glob.glob(os.path.join(model_path, '*.safetensors'))
    if not safetensors_files:
        # If no safetensors files found, check pytorch_model.bin
        safetensors_files = glob.glob(
            os.path.join(model_path, 'pytorch_model*.bin'),
        )

    if not safetensors_files:
        return False

    total_size = sum(os.path.getsize(f) for f in safetensors_files)
    size_ratio = total_size / expected_size

    is_correct_size = (1.0 - tolerance) <= size_ratio <= (1.0 + tolerance)
    return is_correct_size


# TODO(GRT-2435): Change to fixture
def delete_transformers_cache():
    # Only delete the files on local rank 0, otherwise race conditions are created
    if not dist.get_local_rank() == 0:
        return

    hf_cache_home = os.path.expanduser(
        os.getenv(
            'HF_HOME',
            os.path.join(
                os.getenv('XDG_CACHE_HOME', '~/.cache'),
                'huggingface',
            ),
        ),
    )
    HF_MODULES_CACHE = os.getenv(
        'HF_MODULES_CACHE',
        os.path.join(hf_cache_home, 'modules'),
    )
    if os.path.exists(HF_MODULES_CACHE) and os.path.isdir(HF_MODULES_CACHE):
        shutil.rmtree(HF_MODULES_CACHE)


def get_config(
    conf_path: str = 'scripts/train/yamls/pretrain/testing.yaml',
) -> DictConfig:
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    with open(conf_path) as f:
        test_cfg = om.load(f)

    return cast(DictConfig, test_cfg)


def test_callback_inits():
    # test with defaults
    _ = HuggingFaceCheckpointer(save_folder='test', save_interval='1ba')

    # test default metadata when mlflow registered name is given
    hf_checkpointer = HuggingFaceCheckpointer(
        save_folder='test',
        save_interval='1ba',
        mlflow_registered_model_name='test_model_name',
    )

    assert hf_checkpointer.mlflow_logging_config['task'] == 'llm/v1/completions'


class MockSpawnProcess:
    """Class for mocking `multiprocessing.context.SpawnProcess`.

    Runs `target(**kwargs)` on the main process.

    Mock classes are not picklable and therefore cannot be used with
    multiprocessing, so we need to patch SpawnProcess for tests.
    """

    def __init__(
        self,
        target: Callable,
        kwargs: dict[str, Any],
        exitcode: int = 0,
    ):
        self.target = target
        self.kwargs = kwargs
        self.exitcode = exitcode

    def start(self):
        self.target(**self.kwargs)

    def is_alive(self) -> bool:
        return False


def _create_mlflow_logger_mock() -> MagicMock:
    mlflow_logger_mock = MagicMock(spec=MLFlowLogger)
    mlflow_logger_mock._mlflow_client = MagicMock()
    mlflow_logger_mock._experiment_id = 'mlflow-experiment-id'
    mlflow_logger_mock._run_id = 'mlflow-run-id'
    mlflow_logger_mock._enabled = True
    mlflow_logger_mock.log_model = MagicMock()
    mlflow_logger_mock.model_registry_prefix = ''
    mlflow_logger_mock.model_registry_uri = None
    mlflow_logger_mock.state_dict = lambda *args, **kwargs: {}
    mlflow_logger_mock.save_model = MagicMock(wraps=_save_model_mock)
    mlflow_logger_mock.run_url = 'fake-url'
    mlflow_logger_mock.tracking_uri = None
    return mlflow_logger_mock


def _create_optimizer(original_model: torch.nn.Module) -> torch.optim.Optimizer:
    optimizer_config = _OPTIMIZER_CFG()
    optimizer_name = optimizer_config.pop('name')
    return build_optimizer(
        original_model,
        optimizer_name,
        optimizer_config,
    )


@pytest.mark.gpu
@pytest.mark.parametrize('mlflow_registry_error', [True, False])
@pytest.mark.parametrize(
    'mlflow_registered_model_name',
    [None, 'dummy-registered-name'],
)
@patch('os.cpu_count', MagicMock(return_value=1))
@patch(
    'llmfoundry.callbacks.hf_checkpointer.SpawnProcess',
    new=MockSpawnProcess,
)
def test_final_register_only(
    mlflow_registry_error: bool,
    mlflow_registered_model_name: Optional[str],
    tiny_ft_dataloader: DataLoader,
    tmp_path: pathlib.Path,
    build_tiny_mpt: Callable,
):
    if mlflow_registry_error and mlflow_registered_model_name is None:
        pytest.skip(
            'Cannot test mlflow_registry_error without mlflow_registered_model_name',
        )

    delete_transformers_cache()

    dist.initialize_dist(get_device('gpu'))

    precision_str = 'bfloat16'

    checkpointer_callback = HuggingFaceCheckpointer(
        save_folder=os.path.join(tmp_path, 'checkpoints'),
        save_interval='1dur',
        precision=precision_str,
        mlflow_registered_model_name=mlflow_registered_model_name,
        final_register_only=True,
    )

    original_model = build_tiny_mpt()

    optimizer = _create_optimizer(original_model)

    mlflow_logger_mock = _create_mlflow_logger_mock()

    checkpointer_callback._save_checkpoint = MagicMock(
        wraps=checkpointer_callback._save_checkpoint,
    )
    trainer = Trainer(
        model=original_model,
        device='gpu',
        train_dataloader=tiny_ft_dataloader,
        max_duration='1ba',
        callbacks=[checkpointer_callback],
        loggers=[mlflow_logger_mock],
        optimizers=optimizer,
        save_latest_filename=None,
    )

    with mock.patch(
        'llmfoundry.callbacks.hf_checkpointer.SpawnProcess',
        new=lambda target,
        kwargs: MockSpawnProcess(
            target,
            kwargs,
            exitcode=1 if mlflow_registry_error else 0,
        ),
    ):
        trainer.fit()

    if mlflow_registered_model_name is not None:
        assert mlflow_logger_mock.log_model.call_count == 1
        assert checkpointer_callback._save_checkpoint.call_count == 1
        assert checkpointer_callback._save_checkpoint.call_args_list[
            0].kwargs == {
                'register_to_mlflow': True,
                'upload_to_save_folder': False,
            }
    else:
        # No mlflow_registered_model_name, so we should only save the checkpoint
        assert mlflow_logger_mock.log_model.call_count == 0
        assert checkpointer_callback._save_checkpoint.call_count == 1
        assert checkpointer_callback._save_checkpoint.call_args_list[
            0].kwargs == {
                'register_to_mlflow': False,
                'upload_to_save_folder': True,
            }


@pytest.mark.gpu
@pytest.mark.parametrize('log_to_mlflow', [True, False])
@pytest.mark.parametrize(
    'hf_save_interval,save_interval,max_duration,expected_hf_checkpoints,expected_normal_checkpoints',
    [('3ba', '2ba', '4ba', 2, 2), ('1dur', '2ba', '1ep', 1, 2)],
)
@patch('os.cpu_count', MagicMock(return_value=1))
@patch(
    'llmfoundry.callbacks.hf_checkpointer.SpawnProcess',
    new=MockSpawnProcess,
)
def test_huggingface_conversion_callback_interval(
    tmp_path: pathlib.Path,
    log_to_mlflow: bool,
    hf_save_interval: str,
    save_interval: str,
    max_duration: str,
    expected_hf_checkpoints: int,
    expected_normal_checkpoints: int,
    tiny_ft_dataloader: DataLoader,
    mpt_tokenizer: PreTrainedTokenizerBase,
    build_tiny_mpt: Callable,
):
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

    optimizer = _create_optimizer(original_model)

    mlflow_logger_mock = _create_mlflow_logger_mock()

    checkpointer_callback.transform_model_pre_registration = MagicMock(
        wraps=checkpointer_callback.transform_model_pre_registration,
    )
    checkpointer_callback.pre_register_edit = MagicMock(
        wraps=checkpointer_callback.pre_register_edit,
    )
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
        assert mlflow_logger_mock.log_model.call_count == 1
        mlflow_logger_mock.log_model.assert_called_with(
            transformers_model=ANY,
            flavor='transformers',
            artifact_path='final_model_checkpoint',
            registered_model_name='dummy-registered-name',
            run_id='mlflow-run-id',
            await_registration_for=3600,
            metadata=ANY,
            task=ANY,
            input_example={
                'prompt': np.array(['What is Machine Learning?']),
            },
        )
        assert checkpointer_callback.transform_model_pre_registration.call_count == 1
        assert checkpointer_callback.pre_register_edit.call_count == 1
        assert mlflow_logger_mock.log_model.call_count == 1
    else:
        assert checkpointer_callback.transform_model_pre_registration.call_count == 0
        assert checkpointer_callback.pre_register_edit.call_count == 0
        assert mlflow_logger_mock.log_model.call_count == 0

    normal_checkpoints = [
        name for name in os.listdir(os.path.join(tmp_path, 'checkpoints'))
        if name != 'huggingface'
    ]

    huggingface_checkpoints = list(
        os.listdir(os.path.join(tmp_path, 'checkpoints', 'huggingface')),
    )
    assert len(normal_checkpoints) == expected_normal_checkpoints
    assert len(huggingface_checkpoints) == expected_hf_checkpoints

    # Get path to the last checkpoint
    checkpoint_path = os.path.join(
        tmp_path,
        'checkpoints',
        'huggingface',
        f'ba{batches_per_epoch}',
    )

    # Verify the safetensors file size matches the expected precision
    assert isinstance(trainer.state.model.model, torch.nn.Module)
    is_size_correct = check_safetensors_precision(
        model_path=checkpoint_path,
        model=trainer.state.model.model,
        expected_precision=precision,
    )
    assert is_size_correct, f"Safetensors file size doesn't match expected precision {precision_str}"

    # Load the last huggingface checkpoint
    loaded_model = transformers.AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
        torch_dtype=precision,
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
        checkpoint_path,
        trust_remote_code=True,
    )

    # Also check that at least one parameter has the expected precision
    for param_name, param in loaded_model.named_parameters():
        assert param.dtype == precision, \
            f'Parameter {param_name} has dtype {param.dtype}, expected {precision}'
        break

    check_hf_model_equivalence(
        trainer.state.model.model.to(precision),  # type: ignore
        loaded_model,
    )
    check_hf_tokenizer_equivalence(mpt_tokenizer, loaded_tokenizer)

    delete_transformers_cache()


def _get_model_and_tokenizer(
    model: str,
    max_seq_len: int,
    tie_word_embeddings: Optional[bool],
    precision: str,
):
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
            'fc_type': 'te' if precision == 'amp_fp8' else 'torch',
            'loss_fn': 'torch_crossentropy',
            'tie_word_embeddings': tie_word_embeddings,
        }
        tokenizer_name = 'EleutherAI/gpt-neox-20b'
    elif model == 'mptmoe':
        # Test export on moe_world_size 1
        model_cfg = {
            'name': 'mpt_causal_lm',
            'init_device': 'cpu',
            'd_model': 128,
            'n_heads': 2,
            'n_layers': 2,
            'expansion_ratio': 1,
            'ffn_config': {
                'ffn_type': 'mb_dmoe',
                'memory_optimized_mlp': True,
                'moe_lbl_in_fp32': False,
                'moe_loss_weight': 0.01,
                'moe_num_experts': 4,
                'moe_top_k': 2,
                'moe_world_size': 1,
                'uniform_expert_assignment': False,
                'mlp_impl': 'grouped',
            },
            'max_seq_len': max_seq_len,
            'vocab_size': 50368,
            'attn_config': {
                'attn_impl': 'torch',
            },
            'loss_fn': 'torch_crossentropy',
            'no_bias': True,
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
                'vocab_size': 50368,
            },
            'pretrained': False,
            'init_device': 'cpu',
        }
        tokenizer_name = 'EleutherAI/gpt-neox-20b'
    elif model == 'llama2':
        assert tie_word_embeddings is None
        if 'HF_TOKEN' not in os.environ:
            pytest.skip(
                'The CI cluster does not have access to the Llama models, so skip this test.',
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
    return model_cfg, tokenizer_name


def _assert_mlflow_logger_calls(
    mlflow_logger_mock: MagicMock,
    peft_config: Optional[dict] = None,
):
    if dist.get_global_rank() == 0:
        if peft_config is not None:
            expectation = {
                'flavor': 'peft',
                'path': ANY,
                'save_pretrained_dir': ANY,
                'metadata': {},
            }
            assert mlflow_logger_mock.save_model.call_count == 1
        else:
            default_input_example = {
                'prompt': np.array(['What is Machine Learning?']),
            }
            expectation = {
                'transformers_model': ANY,
                'flavor': 'transformers',
                'artifact_path': 'final_model_checkpoint',
                'registered_model_name': 'dummy-registered-name',
                'run_id': 'mlflow-run-id',
                'await_registration_for': 3600,
                'metadata': ANY,
                'task': ANY,
                'input_example': default_input_example,
            }
            assert mlflow_logger_mock.log_model.call_count == 1
        mlflow_logger_mock.log_model.assert_called_with(**expectation)
    else:
        assert mlflow_logger_mock.log_model.call_count == 0


def _get_fsdp_config(fsdp_state_dict_type: Optional[str]):
    fsdp_config = {
        'sharding_strategy': 'FULL_SHARD',
        'mixed_precision': 'PURE',
        'activation_checkpointing': False,
        'activation_checkpointing_reentrant': False,
        'activation_cpu_offload': False,
        'limit_all_gathers': True,
        'state_dict_type': fsdp_state_dict_type,
    }
    return fsdp_config


def _get_dataloader_cfg(tiny_dataset_folder_path: str, max_seq_len: int):
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
        'timeout': 0,
    }
    return dataloader_cfg


def _assert_checkpoint_equivalence(
    tmp_path: pathlib.Path,
    expected_normal_checkpoints: int,
    expected_hf_checkpoints: int,
    trainer: Trainer,
    batches_per_epoch: int,
    precision: torch.dtype,
    model: str,
    tokenizer: PreTrainedTokenizerBase,
    original_model: ComposerModel,
    fsdp_state_dict_type: Optional[str] = None,
    peft_config: Optional[dict] = None,
):
    """Asserts the equivalence of checkpoints.

    Asserts equivalence of checkpoints between the original mpt model and the converted hf model.

    Args:
        tmp_path (str): The path to the temporary directory where the checkpoints are saved.
        expected_normal_checkpoints (int): The expected number of normal checkpoints.
        expected_hf_checkpoints (int): The expected number of HuggingFace checkpoints.
        trainer (Trainer): The trainer object used for training the model.
        batches_per_epoch (int): The number of batches per epoch.
        precision (torch.dtype): The precision of the model.
        model (str): The type of model ('mpt', 'neo', or 'llama2').
        tokenizer (PreTrainedTokenizerBase): The model tokenizer.
        original_model (ComposerModel): The original model object.
        fsdp_state_dict_type (Optional[str], optional): The type of FSDP state dict. Defaults to None.
        peft_config (Optional[dict], optional): The PEFT configuration. Defaults to None.
    """
    loaded_model = None
    loaded_tokenizer = None
    # Only rank zero is saving the huggingface checkpoints, so only check
    # for equivalence on rank zero
    if dist.get_global_rank() == 0:
        normal_checkpoints = [
            name for name in os.listdir(os.path.join(tmp_path, 'checkpoints'))
            if name != 'huggingface'
        ]
        huggingface_checkpoints = list(
            os.listdir(os.path.join(tmp_path, 'checkpoints', 'huggingface')),
        )

        checkpoint_files = os.listdir(
            os.path.join(
                tmp_path,
                'checkpoints',
                'huggingface',
                huggingface_checkpoints[-1],
            ),
        )
        if peft_config is not None:
            assert 'adapter_config.json' in checkpoint_files
            assert 'adapter_model.safetensors' in checkpoint_files

        assert len(normal_checkpoints) == expected_normal_checkpoints
        assert len(huggingface_checkpoints) == expected_hf_checkpoints

        # Patch flash_attn package to be empty to simulate loading the model in
        # an environment without flash attention installed
        with patch.dict('sys.modules', {'flash_attn': None}):
            if peft_config is not None:
                composer_model = trainer.state.model.module if trainer.state.is_model_ddp else trainer.state.model
                composer_model.model.base_model.save_pretrained( # type: ignore
                    tmp_path / 'base-model',
                )

            checkpoint_path = os.path.join(
                tmp_path,
                'checkpoints',
                'huggingface',
                f'ba{batches_per_epoch}',
            )

            if peft_config is not None:
                with open(
                    os.path.join(checkpoint_path, 'adapter_config.json'),
                ) as _f:
                    adapter_config = json.load(_f)

                adapter_config['base_model_name_or_path'] = str(
                    tmp_path / 'base-model',
                )

                with open(
                    os.path.join(checkpoint_path, 'adapter_config.json'),
                    'w',
                ) as _f:
                    json.dump(adapter_config, _f)

            # Load the last huggingface checkpoint
            loaded_model = transformers.AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                trust_remote_code=True,
                torch_dtype=precision,
            )

        # Check that the loaded model has the correct precision, and then set it back
        # to the original for the equivalence check
        if peft_config is None:
            assert loaded_model.config.torch_dtype == precision
            loaded_model.config.torch_dtype = original_model.model.config.torch_dtype  # type: ignore

        if model == 'mpt':
            # Check that we have correctly set these attributes, and then set them back
            # to the original for the equivalence check
            assert loaded_model.config.attn_config['attn_impl'] == 'torch'
            assert loaded_model.config.init_device == 'cpu'
            loaded_model.config.attn_config[
                'attn_impl'
            ] = original_model.model.config.attn_config[  # type: ignore
                'attn_impl']
            loaded_model.config.init_device = original_model.model.config.init_device  # type: ignore

        loaded_tokenizer = transformers.AutoTokenizer.from_pretrained(
            os.path.join(
                tmp_path,
                'checkpoints',
                'huggingface',
                f'ba{batches_per_epoch}',
            ),
            trust_remote_code=True,
        )

        check_hf_model_equivalence(
            trainer.state.model.model.to(precision) if fsdp_state_dict_type  # type: ignore
            is not None else trainer.state.model.module.model.to(precision), # type: ignore
            loaded_model,
            just_lora=peft_config is not None,
        )
        check_hf_tokenizer_equivalence(tokenizer, loaded_tokenizer)


@pytest.mark.world_size(2)
@pytest.mark.gpu
@pytest.mark.parametrize(
    'model,tie_word_embeddings,peft_config',
    [
        ('mpt', True, None),
        ('mpt', False, None),
        ('mptmoe', None, None),
        ('neo', None, None),
        ('llama2', None, None),
        (
            'llama2',
            None,
            {
                'peft_type': 'LORA',
                'task_type': 'CAUSAL_LM',
                'lora_alpha': 32,
                'lora_dropout': 0.05,
                'r': 16,
                'target_modules': [
                    'q_proj',
                    'k_proj',
                    'v_proj',
                ],
            },
        ),
    ],
)
@pytest.mark.parametrize('fsdp_state_dict_type', ['full', 'sharded', None])
@pytest.mark.parametrize(
    'hf_save_interval,save_interval,max_duration,expected_hf_checkpoints,expected_normal_checkpoints,trainer_precision',
    [('1ba', '1ba', '1ba', 1, 1, 'amp_bf16'),
     ('1ba', '1ba', '1ba', 1, 1, 'amp_fp8')],
)
@patch('os.cpu_count', MagicMock(return_value=1))
@patch(
    'llmfoundry.callbacks.hf_checkpointer.SpawnProcess',
    new=MockSpawnProcess,
)
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
    trainer_precision: str,
    peft_config: Optional[dict],
    request: pytest.FixtureRequest,
):
    if model == 'mptmoe' and fsdp_state_dict_type is None:
        pytest.skip('mptmoe requires FSDP')
    if trainer_precision == 'amp_fp8':
        # Check if transformer-engine is installed for FP8.
        try:
            import transformer_engine.pytorch as te
        except ImportError:
            pytest.skip(
                'Precision amp_fp8 requires transformer-engine to be installed',
            )

        # Check we are using mpt models only for FP8.
        if (model == 'neo' or model == 'llama2'):
            pytest.skip(
                'Precision amp_fp8 works only for mpt models, not hf models',
            )

        # Check that we are using H100 or later for FP8.
        if not (torch.cuda.get_device_capability() >= (8, 9)):
            pytest.skip('Amp FP8 requires a GPU with compute capability >= 8.9')

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
        mlflow_registered_model_name='dummy-registered-name',
    )

    # Get small version of each model
    model_cfg, tokenizer_name = _get_model_and_tokenizer(
        model=model,
        max_seq_len=max_seq_len,
        tie_word_embeddings=tie_word_embeddings,
        precision=trainer_precision,
    )
    assert model_cfg is not None
    assert tokenizer_name is not None
    if peft_config is not None:
        model_cfg['peft_config'] = peft_config

    fsdp_config = _get_fsdp_config(fsdp_state_dict_type)
    optimizer_config = _OPTIMIZER_CFG()

    tiny_dataset_folder_path = os.path.join(os.getcwd(), 'test-ift-data-small')
    tiny_dataset_path = os.path.join(tiny_dataset_folder_path, 'train.jsonl')
    if dist.get_global_rank() == 0:
        make_tiny_ft_dataset(path=tiny_dataset_path, size=dataset_size)

    dataloader_cfg = _get_dataloader_cfg(tiny_dataset_folder_path, max_seq_len)

    dataloader_cfg = om.create(dataloader_cfg)

    tokenizer = get_tokenizer_fixture_by_name(request, tokenizer_name)

    dataloader_cfg.pop('name')
    train_dataloader = build_finetuning_dataloader(
        tokenizer=tokenizer,
        device_batch_size=device_batch_size,
        **dataloader_cfg,
    )

    name = model_cfg.pop('name')
    original_model = build_composer_model(
        name,
        tokenizer=tokenizer,
        cfg=model_cfg,
    )
    optimizer_name = optimizer_config.pop('name')
    optimizer = build_optimizer(
        original_model,
        optimizer_name,
        optimizer_config,
    )

    mlflow_logger_mock = MagicMock(spec=MLFlowLogger)
    mlflow_logger_mock.state_dict = lambda *args, **kwargs: {}
    mlflow_logger_mock.save_model = MagicMock(wraps=_save_model_mock)
    mlflow_logger_mock.log_model = MagicMock()
    mlflow_logger_mock.model_registry_prefix = ''
    mlflow_logger_mock._experiment_id = 'mlflow-experiment-id'
    mlflow_logger_mock._run_id = 'mlflow-run-id'
    mlflow_logger_mock._enabled = True
    mlflow_logger_mock.run_url = 'fake-url'
    mlflow_logger_mock.tracking_uri = None
    mlflow_logger_mock.model_registry_uri = None
    trainer = Trainer(
        model=original_model,
        device='gpu',
        precision=trainer_precision,
        parallelism_config={'fsdp': fsdp_config}
        if fsdp_state_dict_type is not None else None,
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

    _assert_mlflow_logger_calls(mlflow_logger_mock, peft_config)

    # summon full params to check equivalence
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    context_manager = te.onnx_export(  # type: ignore
        True,
    ) if trainer_precision == 'amp_fp8' else contextlib.nullcontext()
    with context_manager:
        with FSDP.summon_full_params(
            trainer.state.model,
            writeback=False,
            recurse=True,
        ):
            _assert_checkpoint_equivalence(
                tmp_path=tmp_path,
                expected_normal_checkpoints=expected_normal_checkpoints,
                expected_hf_checkpoints=expected_hf_checkpoints,
                trainer=trainer,
                batches_per_epoch=batches_per_epoch,
                original_model=original_model,
                precision=precision,
                model=model,
                tokenizer=tokenizer,
                fsdp_state_dict_type=fsdp_state_dict_type,
                peft_config=peft_config,
            )

    dist.barrier()
    delete_transformers_cache()


@patch('os.cpu_count', MagicMock(return_value=1))
@patch(
    'llmfoundry.callbacks.hf_checkpointer.SpawnProcess',
    new=MockSpawnProcess,
)
def test_transform_model_pre_registration(request: pytest.FixtureRequest):
    """Test `transform_model_pre_registration` method is called."""

    class ExtendedHuggingFaceCheckpointer(HuggingFaceCheckpointer):
        """Set PEFT to false before registering for testing."""

        def transform_model_pre_registration(self, model: PreTrainedModel):
            self.using_peft = False
            return super().transform_model_pre_registration(model)

    model_cfg, tokenizer_name = _get_model_and_tokenizer(
        model='neo',
        max_seq_len=10,
        tie_word_embeddings=None,
        precision='bfloat16',
    )
    model_cfg['peft_config'] = {
        'peft_type': 'LORA',
        'task_type': 'CAUSAL_LM',
        'lora_alpha': 32,
        'lora_dropout': 0.05,
        'r': 16,
        'target_modules': 'all-linear',
    }
    tokenizer = get_tokenizer_fixture_by_name(request, tokenizer_name)

    original_model = build_composer_model(
        model_cfg.pop('name'),
        tokenizer=tokenizer,
        cfg=model_cfg,
    )

    logger = MagicMock()
    state = MagicMock()
    state.timestamp.batch = 1
    state.is_model_ddp = False
    state.model = original_model
    state.model.tokenizer = tokenizer

    checkpointer = ExtendedHuggingFaceCheckpointer(
        save_folder='test',
        save_interval='1ba',
    )
    mlflow_logger_mock = _create_mlflow_logger_mock()
    checkpointer.mlflow_loggers = [mlflow_logger_mock]  # type: ignore

    assert model_cfg is not None
    assert tokenizer_name is not None

    checkpointer.using_peft = True
    checkpointer._save_checkpoint(
        state=state,
        logger=logger,
        upload_to_save_folder=True,
        register_to_mlflow=True,
    )

    assert mlflow_logger_mock.log_model.call_count == 1


# TODO(GRT-2431): Refactor as enums
@pytest.mark.parametrize(
    'model,tie_word_embeddings',
    [('mpt', True), ('mpt', False),
     pytest.param('mptmoe', None, marks=pytest.mark.gpu), ('neo', None),
     ('llama2', None)],
)
def test_convert_and_generate(
    model: str,
    tie_word_embeddings: bool,
    tmp_path: pathlib.Path,
):
    delete_transformers_cache()

    om_cfg = None
    if model == 'mpt':
        om_cfg = get_config(
            conf_path='scripts/train/yamls/pretrain/testing.yaml',
        )
        om_cfg['tie_word_embeddings'] = tie_word_embeddings
    elif model == 'mptmoe':
        om_cfg = get_config(
            conf_path='scripts/train/yamls/pretrain/testing-moe.yaml',
        )
    elif model == 'neo':
        assert tie_word_embeddings is None
        om_cfg = get_config(
            conf_path='scripts/train/yamls/pretrain/gpt-neo-125m.yaml',
        )
        om_cfg['model']['config_overrides']['hidden_size'] = 36
    elif model == 'llama2':
        assert tie_word_embeddings is None
        if 'HF_TOKEN' not in os.environ:
            pytest.skip(
                'The CI cluster does not have access to the Llama models, so skip this test.',
            )
        om_cfg = get_config(
            conf_path='scripts/train/yamls/pretrain/gpt-neo-125m.yaml',
        )
        om_cfg['model']['pretrained_model_name_or_path'
                       ] = 'meta-llama/Llama-2-7b-hf'
        om_cfg['model']['config_overrides']['num_hidden_layers'] = 2
        om_cfg['model']['use_auth_token'] = True
        om_cfg['tokenizer']['name'] = 'meta-llama/Llama-2-7b-hf'
    else:
        raise ValueError(f'Unknown model {model}')
    assert om_cfg is not None

    om_cfg['model']['init_device'] = 'cpu'
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        om_cfg.tokenizer.name,
        use_auth_token=model == 'llama2',
    )
    name = om_cfg.model.pop('name')
    original_model = build_composer_model(
        name=name,
        tokenizer=tokenizer,
        cfg=to_dict_container(om_cfg['model']),
    )
    trainer = Trainer(
        model=original_model,
        device='cpu' if not model == 'mptmoe' else 'gpu',
    )
    trainer.save_checkpoint(os.path.join(tmp_path, 'checkpoint.pt'))

    args = Namespace(
        composer_path=os.path.join(tmp_path, 'checkpoint.pt'),
        hf_output_path=os.path.join(tmp_path, 'hf-output-folder'),
        output_precision='fp32',
        local_checkpoint_save_location=None,
        hf_repo_for_upload=None,
        trust_remote_code=False,
        test_uploaded_model=False,
    )
    convert_composer_to_hf(args)

    loaded_config = transformers.AutoConfig.from_pretrained(
        os.path.join(tmp_path, 'hf-output-folder'),
        trust_remote_code=True,
    )
    loaded_model = transformers.AutoModelForCausalLM.from_pretrained(
        os.path.join(tmp_path, 'hf-output-folder'),
        config=loaded_config,
        trust_remote_code=True,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        os.path.join(tmp_path, 'hf-output-folder'),
        trust_remote_code=True,
    )

    device = 'cuda' if model == 'mptmoe' else 'cpu'
    precision = torch.bfloat16 if model == 'mptmoe' else torch.float32
    original_model.to(device)
    original_model.to(precision)
    loaded_model.to(device)
    loaded_model.to(precision)

    output = loaded_model.generate(
        tokenizer('hello', return_tensors='pt')['input_ids'].to(device),
        max_new_tokens=1,
    )
    assert output.shape == (1, 2 + (1 if model == 'llama2' else 0))

    assert isinstance(original_model.model, torch.nn.Module)
    assert sum(p.numel() for p in original_model.model.parameters()
              ) == sum(p.numel() for p in loaded_model.parameters())
    assert all(
        str(type(module1)).split('.')[-1] == str(type(module2)).split('.')[-1]
        for module1, module2 in
        zip(original_model.model.modules(), loaded_model.modules())
    )
    for p1, p2 in zip(
        original_model.model.parameters(),
        loaded_model.parameters(),
    ):
        assert torch.allclose(p1, p2)

    delete_transformers_cache()


@pytest.mark.parametrize(
    'conf_path',
    [
        'scripts/train/yamls/pretrain/testing.yaml',
        pytest.param(
            'scripts/train/yamls/pretrain/testing-moe.yaml',
            marks=pytest.mark.gpu,
        ),
    ],
)
@pytest.mark.parametrize('tie_word_embeddings', [True, False])
def test_convert_and_generate_meta(
    tie_word_embeddings: str,
    tmp_path: pathlib.Path,
    conf_path: str,
):
    delete_transformers_cache()

    from composer.utils import dist
    gathered_paths = dist.all_gather_object(tmp_path)
    tmp_path_gathered = gathered_paths[0]

    om_cfg = get_config(conf_path=conf_path)

    om_cfg['model']['init_device'] = 'cpu'
    om_cfg['tie_word_embeddings'] = tie_word_embeddings
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        om_cfg.tokenizer.name,
    )
    name = om_cfg.model.pop('name')
    original_model = build_composer_model(
        name=name,
        tokenizer=tokenizer,
        cfg=to_dict_container(om_cfg['model']),
    )
    trainer = Trainer(
        model=original_model,
        device='cpu' if not 'moe' in conf_path else 'gpu',
    )
    trainer.save_checkpoint(os.path.join(tmp_path_gathered, 'checkpoint.pt'))

    # patch in the meta device for testing
    sd = torch.load(
        os.path.join(tmp_path_gathered, 'checkpoint.pt'),
        map_location='cpu',
        weights_only=False,
    )
    sd['state']['integrations']['huggingface']['model']['config']['content'][
        'init_device'] = 'meta'
    torch.save(sd, os.path.join(tmp_path_gathered, 'checkpoint.pt'))

    args = Namespace(
        composer_path=os.path.join(tmp_path_gathered, 'checkpoint.pt'),
        hf_output_path=os.path.join(tmp_path_gathered, 'hf-output-folder'),
        output_precision='fp32',
        local_checkpoint_save_location=None,
        hf_repo_for_upload=None,
        trust_remote_code=False,
        test_uploaded_model=False,
    )
    convert_composer_to_hf(args)

    loaded_config = transformers.AutoConfig.from_pretrained(
        os.path.join(tmp_path_gathered, 'hf-output-folder'),
        trust_remote_code=True,
    )
    loaded_model = transformers.AutoModelForCausalLM.from_pretrained(
        os.path.join(tmp_path_gathered, 'hf-output-folder'),
        config=loaded_config,
        trust_remote_code=True,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        os.path.join(tmp_path_gathered, 'hf-output-folder'),
        trust_remote_code=True,
    )

    device = 'cuda' if 'moe' in conf_path else 'cpu'
    precision = torch.bfloat16 if 'moe' in conf_path else torch.float32
    original_model.to(device)
    original_model.to(precision)
    loaded_model.to(device)
    loaded_model.to(precision)

    output = loaded_model.generate(
        tokenizer('hello', return_tensors='pt')['input_ids'].to(device),
        max_new_tokens=1,
    )
    assert output.shape == (1, 2)

    assert isinstance(original_model.model, torch.nn.Module)
    assert sum(p.numel() for p in original_model.model.parameters()
              ) == sum(p.numel() for p in loaded_model.parameters())
    assert all(
        str(type(module1)).split('.')[-1] == str(type(module2)).split('.')[-1]
        for module1, module2 in
        zip(original_model.model.modules(), loaded_model.modules())
    )
    for p1, p2 in zip(
        original_model.model.parameters(),
        loaded_model.parameters(),
    ):
        assert torch.allclose(p1, p2)

    delete_transformers_cache()


@pytest.mark.world_size(4)
@pytest.mark.gpu
@pytest.mark.parametrize('num_experts', [2, 4, 8])
@pytest.mark.parametrize('sharding_strategy', ['FULL_SHARD'])
def test_mptmoe_huggingface_conversion_callback(
    tmp_path: pathlib.Path,
    num_experts: int,
    sharding_strategy: str,
    tiny_neox_tokenizer: PreTrainedTokenizerBase,
    hf_save_interval: str = '1ba',
    save_interval: str = '1ba',
    max_duration: str = '1ba',
    expected_hf_checkpoints: int = 1,
    expected_normal_checkpoints: int = 1,
):

    delete_transformers_cache()

    dist.initialize_dist(get_device('gpu'))
    if dist.get_world_size() != 4:
        pytest.skip('This test requires 4 GPUs')

    max_seq_len = 16
    device_batch_size = 1
    dataset_size = 2
    precision_torch = torch.float32
    precision_str = 'float32'
    batches_per_epoch = math.ceil(dataset_size / (device_batch_size * 2))

    checkpointer_callback = HuggingFaceCheckpointer(
        save_folder=os.path.join(tmp_path, 'checkpoints'),
        save_interval=hf_save_interval,
        precision=precision_str,
    )

    # get small version of each model
    model_cfg = None
    tokenizer_name = None

    # Test export on moe_world_size 1
    model_cfg = {
        'name': 'mpt_causal_lm',
        'init_device': 'cpu',
        'd_model': 128,
        'n_heads': 2,
        'n_layers': 2,
        'expansion_ratio': 1,
        'ffn_config': {
            'ffn_type':
                'mb_dmoe',
            'memory_optimized_mlp':
                True,
            'moe_lbl_in_fp32':
                False,
            'moe_loss_weight':
                0.01,
            'moe_num_experts':
                num_experts,
            'moe_top_k':
                2,
            'moe_world_size':
                2,
            'uniform_expert_assignment':
                True,
            'mlp_impl':
                'grouped',
            'mlp_type':
                'glu',
            'device_mesh': [1, 2] if sharding_strategy == 'HYBRID_SHARD' else [
                2,
            ],
        },
        'precision': 'amp_bf16',
        'max_seq_len': max_seq_len,
        'vocab_size': 50368,
        'attn_config': {
            'attn_impl': 'torch',
        },
        'loss_fn': 'torch_crossentropy',
        'no_bias': True,
    }
    tokenizer_name = 'EleutherAI/gpt-neox-20b'
    assert model_cfg is not None
    assert tokenizer_name is not None

    fsdp_config = {
        'sharding_strategy': sharding_strategy,
        'mixed_precision': 'PURE',
        'activation_checkpointing': False,
        'activation_checkpointing_reentrant': False,
        'activation_cpu_offload': False,
        'limit_all_gathers': True,
        'use_orig_params': True,
        'data_parallel_shard_degree': 4,
    }

    if sharding_strategy == 'HYBRID_SHARD':
        fsdp_config['data_parallel_shard_degree'] = 1

    tiny_dataset_folder_path = os.path.join(os.getcwd(), 'test-ift-data-small')
    tiny_dataset_path = os.path.join(tiny_dataset_folder_path, 'train.jsonl')
    if dist.get_global_rank() == 0:
        make_tiny_ft_dataset(path=tiny_dataset_path, size=dataset_size)

    dataloader_cfg = {
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
        'timeout': 0,
    }

    dataloader_cfg = om.create(dataloader_cfg)

    tokenizer = tiny_neox_tokenizer

    train_dataloader = build_finetuning_dataloader(
        **dataloader_cfg,
        tokenizer=tokenizer,
        device_batch_size=device_batch_size,
    )

    optimizer_config = {
        'name': 'decoupled_adamw',
        'lr': 6e-4,
        'betas': [0.9, 0.95],
        'eps': 1e-8,
        'weight_decay': 0.0,
    }
    optimizer_name = optimizer_config.pop('name')

    init_context = process_init_device(model_cfg, fsdp_config)
    name = model_cfg.pop('name')
    original_model = build_composer_model(
        name=name,
        tokenizer=tokenizer,
        init_context=init_context,
        cfg=model_cfg,
    )

    optimizer = build_optimizer(
        original_model,
        optimizer_name,
        optimizer_config,
    )
    trainer = Trainer(
        model=original_model,
        device='gpu',
        parallelism_config={'fsdp': fsdp_config},
        train_dataloader=train_dataloader,
        save_folder=os.path.join(tmp_path, 'checkpoints'),
        save_interval=save_interval,
        max_duration=max_duration,
        callbacks=[checkpointer_callback],
        optimizers=optimizer,
        save_latest_filename=None,
        precision=model_cfg.pop('precision', None),
        save_weights_only=True,
    )
    trainer.fit()
    batch = trainer.state.batch
    model_output_logits = trainer.state.model(batch).logits

    # summon full params to check equivalence
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    with FSDP.summon_full_params(
        trainer.state.model,
        writeback=False,
        recurse=True,
    ):
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
            huggingface_checkpoints = list(
                os.listdir(
                    os.path.join(tmp_path, 'checkpoints', 'huggingface'),
                ),
            )
            assert len(normal_checkpoints) == expected_normal_checkpoints
            assert len(huggingface_checkpoints) == expected_hf_checkpoints

            # Patch flash_attn package to be empty to simulate loading the model in
            # an environment without flash atttention installed
            with patch.dict('sys.modules', {'flash_attn': None}):
                # Load the last huggingface checkpoint
                loaded_model = transformers.AutoModelForCausalLM.from_pretrained(
                    os.path.join(
                        tmp_path,
                        'checkpoints',
                        'huggingface',
                        f'ba1',
                    ),
                    trust_remote_code=True,
                )

            # Check that the loaded model has the correct precision, and then set it back
            # to the original for the equivalence check
            assert loaded_model.config.torch_dtype == precision_torch
            loaded_model.config.torch_dtype = original_model.model.config.torch_dtype  # type: ignore

            loaded_tokenizer = transformers.AutoTokenizer.from_pretrained(
                os.path.join(
                    tmp_path,
                    'checkpoints',
                    'huggingface',
                    f'ba{batches_per_epoch}',
                ),
                trust_remote_code=True,
            )
        assert isinstance(trainer.state.model.model, torch.nn.Module)
        for n, p in trainer.state.model.model.named_parameters():
            if isinstance(p, DTensor):
                submodule_name, param_name = '.'.join(
                    n.split('.')[:-1],
                ), n.split('.')[-1]
                submodule = trainer.state.model.model.get_submodule(
                    submodule_name,
                )
                param_tensor = p.full_tensor()
                param = torch.nn.Parameter(param_tensor)
                submodule.register_parameter(param_name, param)

        if dist.get_global_rank() == 0:
            assert loaded_tokenizer is not None
            assert loaded_model is not None
            assert isinstance(
                trainer.state.model.model.module,
                transformers.PreTrainedModel,
            )
            check_hf_model_equivalence(
                trainer.state.model.model.module,
                loaded_model,
            )
            check_hf_tokenizer_equivalence(tokenizer, loaded_tokenizer)

            # Check output equivalence
            loaded_model = loaded_model.cuda().bfloat16()  # type: ignore
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.cuda()
            loaded_model_logits = loaded_model(
                input_ids=batch.get('input_ids', None),
                attention_mask=batch.get('attention_mask', None),
                sequence_id=batch.get('sequence_id', None),
                inputs_embeds=batch.get('inputs_embeds', None),
            ).logits
            assert torch.equal(loaded_model_logits, model_output_logits)

    dist.barrier()

    delete_transformers_cache()


def test_mpt_convert_simple(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
):
    delete_transformers_cache()

    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    original_config_auto_class = MPTConfig._auto_class
    original_model_auto_class = MPTForCausalLM._auto_class
    CONFIG_MAPPING._extra_content['mpt'] = MPTConfig
    MPTConfig.register_for_auto_class()
    MPTForCausalLM.register_for_auto_class('AutoModelForCausalLM')

    model_cfg = {
        'name': 'mpt_causal_lm',
        'init_device': 'cpu',
        'd_model': 64,
        'n_heads': 2,
        'n_layers': 2,
        'expansion_ratio': 4,
        'max_seq_len': 256,
        'vocab_size': 50368,
        'attn_config': {
            'attn_impl': 'torch',
        },
        'loss_fn': 'torch_crossentropy',
        'tie_word_embeddings': False,
    }

    original_model = build_composer_model(
        name='mpt_causal_lm',
        tokenizer=None,
        cfg=model_cfg,
    )

    original_model.model.save_pretrained(str(tmp_path))  # type: ignore

    edit_files_for_hf_compatibility(str(tmp_path))

    monkeypatch.setattr(catalogue, 'REGISTRY', {})

    _ = transformers.AutoModelForCausalLM.from_pretrained(
        tmp_path,
        trust_remote_code=True,
    )

    delete_transformers_cache()

    del CONFIG_MAPPING._extra_content['mpt']
    MPTConfig._auto_class = original_config_auto_class
    MPTForCausalLM._auto_class = original_model_auto_class


@pytest.mark.parametrize(
    'license_file_name',
    ['LICENSE', 'LICENSE.txt', 'license', 'license.md', None],
)
def test_license_file_finder(
    tmp_path: pathlib.Path,
    license_file_name: Optional[str],
):
    if license_file_name is not None:
        with open(os.path.join(tmp_path, license_file_name), 'w') as f:
            f.write('test')

    found_path = _maybe_get_license_filename(str(tmp_path))
    assert (found_path == license_file_name
           ) if license_file_name is not None else (found_path is None)


@pytest.mark.parametrize('generation_config', [None, {}, {'max_length': 200}])
def test_generation_config_variants(
    generation_config: Optional[Union[dict[str, Any], GenerationConfig]],
):

    class MockModel(PreTrainedModel, nn.Module):

        def __init__(self, config: PretrainedConfig):
            nn.Module.__init__(self)
            self.config = config
            # Ensure generation_config is always a GenerationConfig object
            if isinstance(config.generation_config, dict):
                self.generation_config = GenerationConfig(
                    **config.generation_config,
                )
            else:
                self.generation_config = config.generation_config

    config = AutoConfig.from_pretrained('gpt2')
    # Convert dict to GenerationConfig if needed
    if isinstance(generation_config, dict):
        generation_config = GenerationConfig(**generation_config)
    config.generation_config = generation_config

    mock_model = MockModel(config)
    logger = MagicMock()
    state = MagicMock()
    state.timestamp.batch = 1
    state.is_model_ddp = False
    state.model.model = mock_model
    state.model.tokenizer = None

    checkpointer = HuggingFaceCheckpointer(
        save_folder='test',
        save_interval='1ba',
    )

    checkpointer._save_checkpoint(
        state=state,
        logger=logger,
        upload_to_save_folder=False,
        register_to_mlflow=False,
    )


@patch(
    'llmfoundry.callbacks.hf_checkpointer.SpawnProcess',
    new=MockSpawnProcess,
)
def test_save_additional_contents(request: pytest.FixtureRequest):
    """Tests the call of save additional contents."""

    class ExtendedHuggingFaceCheckpointer(HuggingFaceCheckpointer):

        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, **kwargs)
            self._calls = 0
            self._save_dir_calls: list[str] = []

        @property
        def calls(self) -> int:
            return self._calls

        @property
        def save_dir_calls(self) -> list[str]:
            return self._save_dir_calls

        def save_additional_contents(
            self,
            state: State,
            logger: Logger,
            save_dir: str,
        ):
            self._calls += 1
            self._save_dir_calls.append(save_dir)

    model_cfg, tokenizer_name = _get_model_and_tokenizer(
        model='neo',
        max_seq_len=10,
        tie_word_embeddings=None,
        precision='bfloat16',
    )
    tokenizer = get_tokenizer_fixture_by_name(request, tokenizer_name)

    original_model = build_composer_model(
        model_cfg.pop('name'),
        tokenizer=tokenizer,
        cfg=model_cfg,
    )

    logger = MagicMock()
    state = MagicMock()
    state.timestamp.batch = 1
    state.is_model_ddp = False
    state.model = original_model
    state.model.tokenizer = tokenizer

    checkpointer = ExtendedHuggingFaceCheckpointer(
        save_folder='test',
        save_interval='1ba',
    )
    mlflow_logger_mock = _create_mlflow_logger_mock()
    checkpointer.mlflow_loggers = [mlflow_logger_mock]  # type: ignore
    checkpointer._save_checkpoint(
        state=state,
        logger=logger,
        upload_to_save_folder=True,
        register_to_mlflow=True,
    )

    assert checkpointer.calls == 2
    assert checkpointer.save_dir_calls[0] != checkpointer.save_dir_calls[1]
