# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import unittest.mock as mock
from typing import Union

import pytest
from composer.callbacks import Generate
from omegaconf import OmegaConf as om
from transformers import PreTrainedTokenizerBase

from llmfoundry.callbacks import HuggingFaceCheckpointer
from llmfoundry.tokenizers.tiktoken import TiktokenTokenizerWrapper
from llmfoundry.utils.builders import build_callback, build_tokenizer


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
