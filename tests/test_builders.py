# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import unittest.mock as mock
from typing import Union

import pytest
from composer.callbacks import Generate
from transformers import PreTrainedTokenizerBase

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
