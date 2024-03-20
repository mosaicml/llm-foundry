# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import re
import unittest.mock as mock
from copy import deepcopy
from typing import Any, Dict, Union
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from composer.callbacks import Generate
from composer.core import Evaluator
from composer.loggers import WandBLogger
from omegaconf import DictConfig, ListConfig
from transformers import PreTrainedTokenizerBase

from llmfoundry.callbacks import HuggingFaceCheckpointer
from llmfoundry.tokenizers.tiktoken import TiktokenTokenizerWrapper
from llmfoundry.utils.builders import (add_metrics_to_eval_loaders,
                                       build_callback, build_eval_loaders,
                                       build_evaluators, build_logger,
                                       build_optimizer, build_tokenizer)


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


def test_tokenizer_no_EOS():
    with pytest.raises(
            ValueError,
            match='The tokenizer bert-base-uncased must have an eos_token.'):
        build_tokenizer('bert-base-uncased', {})


def test_build_callback_fails():
    with pytest.raises(ValueError):
        build_callback('nonexistent_callback', {}, {})


@pytest.mark.parametrize(
    'interval_key,interval_value',
    [('interval', '10ba')],
)
def test_build_generate_callback(
    interval_key: str,
    interval_value: Union[str, int],
):

    with mock.patch.object(Generate, '__init__',
                           autospec=True) as mock_generate:
        mock_generate.return_value = None
        build_callback(
            'generate_callback',
            {
                'prompts': ['hello'],
                interval_key: interval_value,
                'foo': 'bar',
                'something': 'else',
            },
            {},
        )

        assert mock_generate.call_count == 1
        _, _, kwargs = mock_generate.mock_calls[0]
        assert kwargs['prompts'] == ['hello']
        assert kwargs['interval'] == '10ba'
        assert kwargs['something'] == 'else'
        assert kwargs['foo'] == 'bar'


def test_build_generate_callback_unspecified_interval():
    with pytest.raises(TypeError):
        with mock.patch.object(Generate, '__init__',
                               autospec=True) as mock_generate:
            mock_generate.return_value = None
            build_callback(
                'generate_callback',
                {
                    'prompts': ['hello'],
                    'foo': 'bar',
                    'something': 'else',
                },
                {},
            )


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
                       kwargs={
                           'save_folder': save_folder,
                           'save_interval': save_interval,
                           'mlflow_logging_config': mlflow_logging_config_dict
                       },
                       config={})

        assert mock_hf_checkpointer.call_count == 1
        _, _, kwargs = mock_hf_checkpointer.mock_calls[0]
        assert kwargs['save_folder'] == save_folder
        assert kwargs['save_interval'] == save_interval
        assert isinstance(kwargs['mlflow_logging_config'], dict)
        assert isinstance(kwargs['mlflow_logging_config']['metadata'], dict)
        assert kwargs['mlflow_logging_config'] == mlflow_logging_config_dict


def test_build_logger():
    with pytest.raises(ValueError):
        _ = build_logger('unknown', {})

    logger_cfg = {
        'project': 'foobar',
        'init_kwargs': {
            'config': {
                'foo': 'bar',
            }
        }
    }
    wandb_logger = build_logger('wandb', logger_cfg)  # type: ignore
    assert isinstance(wandb_logger, WandBLogger)
    assert wandb_logger.project == 'foobar'

    # confirm the typing conversion from DictConfig to dict,
    # wandb.init() will fail if config is not explicitly
    # dict type
    ik = wandb_logger._init_kwargs
    assert ik == {'config': {'foo': 'bar'}, 'project': 'foobar'}
    assert isinstance(ik, dict)
    assert isinstance(ik['config'], dict)


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
            'param_str_match': 'no.*.bias',
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
    optimizer_config = deepcopy(optimizer_config)
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


def test_build_evaluators_empty():
    evaluators, logger_keys, eval_gauntlet_callback = build_evaluators(
        None,
        None,
        None,
        tokenizer=None,  # type: ignore
        device_eval_batch_size=1,
        icl_seq_len=2,
        icl_subset_num_batches=3)
    assert evaluators == []
    assert logger_keys == []
    assert eval_gauntlet_callback is None


def test_build_eval_loaders(monkeypatch: pytest.MonkeyPatch):
    tokenizer = TiktokenTokenizerWrapper(model_name='gpt-4')

    eval_loader_cfg = DictConfig({
        'name': 'text',
        'dataset': {
            # mocked, not needed
        },
        'drop_last': False,
        'num_workers': 8,
    })
    monkeypatch.setattr('llmfoundry.data.text_data.StreamingTextDataset',
                        lambda *args, **kwargs: MagicMock())
    eval_loaders = build_eval_loaders(eval_loader_cfg, tokenizer, 2)

    assert len(eval_loaders) == 1

    assert eval_loaders[0].label == 'eval'
    assert eval_loaders[0].dataloader is not None
    assert eval_loaders[0].metric_names == []

    multi_eval_loader_cfg = ListConfig([
        {
            'name': 'text',
            'label': 'test1',
            'dataset': {
                # mocked, not needed
            },
            'drop_last': False,
            'num_workers': 8,
        },
        {
            'name': 'text',
            'label': 'test2',
            'dataset': {
                # mocked, not needed
            },
            'drop_last': False,
            'num_workers': 8,
        }
    ])
    monkeypatch.setattr('llmfoundry.data.text_data.StreamingTextDataset',
                        lambda *args, **kwargs: MagicMock())
    eval_loaders2 = build_eval_loaders(multi_eval_loader_cfg, tokenizer, 2)

    assert len(eval_loaders2) == 2

    assert eval_loaders2[0].label == 'eval/test1'
    assert eval_loaders2[0].dataloader is not None
    assert eval_loaders2[0].metric_names == []

    assert eval_loaders2[1].label == 'eval/test2'
    assert eval_loaders2[1].dataloader is not None
    assert eval_loaders2[1].metric_names == []


def test_add_metrics_to_eval_loaders():
    evaluators = [
        Evaluator(
            label='first',
            metric_names=['a', 'b'],
            dataloader=None,  # type: ignore
            device_eval_microbatch_size=1,
        ),
        Evaluator(
            label='second',
            metric_names=[],
            dataloader=None,  # type: ignore
            device_eval_microbatch_size=1,
        ),
        Evaluator(
            label='third',
            metric_names=['c'],
            dataloader=None,  # type: ignore
            device_eval_microbatch_size=1,
        )
    ]

    new_evaluators = add_metrics_to_eval_loaders(evaluators, ['new1', 'new2'])
    assert len(new_evaluators) == 3

    assert new_evaluators[0].label == 'second'
    assert new_evaluators[0].metric_names == ['new1', 'new2']

    assert new_evaluators[1].label == 'first'
    assert new_evaluators[1].metric_names == ['a', 'b']

    assert new_evaluators[2].label == 'third'
    assert new_evaluators[2].metric_names == ['c']
