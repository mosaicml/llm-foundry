# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import re
import unittest.mock as mock
from copy import deepcopy
from typing import Any, Dict, Union

import pytest
import torch
import torch.nn as nn
from composer.callbacks import Generate
from omegaconf import OmegaConf as om
from transformers import PreTrainedTokenizerBase

from llmfoundry.callbacks import HuggingFaceCheckpointer
from llmfoundry.tokenizers.tiktoken import TiktokenTokenizerWrapper
from llmfoundry.utils.builders import (build_callback, build_optimizer,
                                       build_tokenizer)


@pytest.mark.parametrize('tokenizer_name,tokenizer_kwargs', [
    ('tiktoken', {
        'model_name': 'gpt-4'
    }),
    ('EleutherAI/gpt-neo-125M', {
        'model_max_length': 10
    }),
    ('mosaicml/mpt-7b', {
        'model_max_length': 20
    }),
])
def test_tokenizer_builder(tokenizer_name: str, tokenizer_kwargs: dict):
    tokenizer = build_tokenizer(tokenizer_name, tokenizer_kwargs)

    if tokenizer_name == 'tiktoken':
        assert isinstance(tokenizer, TiktokenTokenizerWrapper)
        assert tokenizer.model_name == tokenizer_kwargs['model_name']
    else:
        assert tokenizer.model_max_length == tokenizer_kwargs[
            'model_max_length']
        assert isinstance(tokenizer, PreTrainedTokenizerBase)


def test_build_callback_fails():
    with pytest.raises(ValueError):
        build_callback('nonexistent_callback', {})


@pytest.mark.parametrize(
    'interval_key,interval_value',
    [('interval', '10ba'), ('batch_log_interval', 10)],
)
def test_build_generate_callback(
    interval_key: str,
    interval_value: Union[str, int],
):

    with mock.patch.object(Generate, '__init__',
                           autospec=True) as mock_generate:
        mock_generate.return_value = None
        build_callback(
            'generate_callback', {
                'prompts': ['hello'],
                interval_key: interval_value,
                'foo': 'bar',
                'something': 'else',
            })

        assert mock_generate.call_count == 1
        _, _, kwargs = mock_generate.mock_calls[0]
        assert kwargs['prompts'] == ['hello']
        assert kwargs['interval'] == '10ba'
        assert kwargs['something'] == 'else'
        assert kwargs['foo'] == 'bar'


def test_build_generate_callback_unspecified_interval():
    with pytest.raises(KeyError):
        with mock.patch.object(Generate, '__init__',
                               autospec=True) as mock_generate:
            mock_generate.return_value = None
            build_callback('generate_callback', {
                'prompts': ['hello'],
                'foo': 'bar',
                'something': 'else',
            })


def test_build_hf_checkpointer_callback():
    with mock.patch.object(HuggingFaceCheckpointer,
                           '__init__') as mock_hf_checkpointer:
        mock_hf_checkpointer.return_value = None
        save_folder = 'path_to_save_folder'
        save_interval = 1
        mlflow_logging_config_dict = {
            'metadata': {
                'databricks_model_family': 'MptForCausalLM',
                'databricks_model_size_parameters': '7b',
                'databricks_model_source': 'mosaic-fine-tuning',
                'task': 'llm/v1/completions'
            }
        }
        build_callback(name='hf_checkpointer',
                       kwargs=om.create({
                           'save_folder': save_folder,
                           'save_interval': save_interval,
                           'mlflow_logging_config': mlflow_logging_config_dict
                       }))

        assert mock_hf_checkpointer.call_count == 1
        _, _, kwargs = mock_hf_checkpointer.mock_calls[0]
        assert kwargs['save_folder'] == save_folder
        assert kwargs['save_interval'] == save_interval
        assert isinstance(kwargs['mlflow_logging_config'], dict)
        assert isinstance(kwargs['mlflow_logging_config']['metadata'], dict)
        assert kwargs['mlflow_logging_config'] == mlflow_logging_config_dict


class _DummyModule(nn.Module):

    def __init__(self, device: str = 'cpu', dtype: torch.dtype = torch.float32):
        super().__init__()
        self.linear0 = nn.Linear(4, 3, device=device, dtype=dtype)
        self.norm0 = nn.LayerNorm(3, device=device, dtype=dtype)
        self.linear1 = nn.Linear(3, 5, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type:ignore
        return self.linear1(self.norm0(self.linear0(x)))


@pytest.mark.parametrize('name, optimizer_config', [
    ('decoupled_adamw', {}),
    ('decoupled_lionw', {}),
    ('clip_lion', {}),
    ('adalr_lion', {}),
    pytest.param('decoupled_lionw_8b', {}, marks=pytest.mark.gpu),
])
@pytest.mark.parametrize('opt_additional_config', [
    {
        'disable_grad': 'norm'
    },
    {
        'disable_grad': ['norm', 'bias']
    },
    {
        'param_groups': [{
            'param_str_match': 'norm',
            'lr': 1e-9,
            'weight_decay': 0.0,
        },]
    },
    {
        'param_groups': [{
            'param_str_match': 'norm',
            'lr': 1e-4,
            'weight_decay': 0.0,
        },],
        'disable_grad': ['bias'],
    },
])
def test_build_optimizer(name: str, optimizer_config: Dict[str, Any],
                         opt_additional_config: Dict[str, Any]):
    model = _DummyModule()
    optimizer_config.update(deepcopy(opt_additional_config))
    optimizer = build_optimizer(model, name, optimizer_config)

    if 'disable_grad' in opt_additional_config.keys():
        disable_grad = opt_additional_config['disable_grad']
        if isinstance(disable_grad, str):
            disable_grad = [disable_grad]
        for n, p in model.named_parameters():
            for k in disable_grad:
                if re.search(k, n):
                    assert not p.requires_grad

    if 'param_groups' in opt_additional_config.keys():
        for param_group_config, param_group in zip(
                opt_additional_config['param_groups'],
                optimizer.param_groups[1:]):
            param_group_config = deepcopy(param_group_config)
            param_str_match = param_group_config.pop('param_str_match')

            for k, v in param_group_config.items():
                assert param_group[k] == v

            param_ids = [id(p) for p in param_group['params']]
            for n, p in model.named_parameters():
                if re.search(param_str_match, n):
                    assert id(p) in param_ids
