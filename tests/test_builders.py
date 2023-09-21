# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from transformers import PreTrainedTokenizerBase

from llmfoundry.tokenizers.tiktoken import TiktokenTokenizerWrapper
from llmfoundry.utils.builders import build_tokenizer


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
