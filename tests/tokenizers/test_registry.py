# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional

import pytest
from transformers import BatchEncoding, PreTrainedTokenizer

from llmfoundry.registry import tokenizers
from llmfoundry.utils import build_tokenizer


class DummyTokenizer(PreTrainedTokenizer):
    """A dummy tokenizer that inherits from ``PreTrainedTokenizer``."""

    def __init__(
        self,
        model_name: Optional[str] = 'dummy',
        **kwargs: Optional[Dict[str, Any]],
    ):
        """Dummy constructor that has no real purpose."""
        super().__init__(
            model_name=model_name,
            eos_token='0',
            pad_token='1',
            **kwargs,
        )

    def get_vocab(self) -> Dict[str, int]:
        return {}

    def tokenize(self, text: str) -> List[str]:
        return [text]

    def encode(self, text: str) -> List[int]:
        return [ord(character) for character in text]

    def decode(self, token_ids: List[int]) -> List[str]:
        return ''.join([chr(token) for token in token_ids])

    def __call__(self, text: str) -> BatchEncoding:
        raise NotImplementedError()


def test_tokenizer_registry():
    tokenizers.register('dummy', func=DummyTokenizer)
    tokenizer = build_tokenizer(tokenizer_name='dummy', tokenizer_kwargs={})
    assert type(tokenizer) == DummyTokenizer

    text = 'Hello World'
    tokenized = tokenizer.tokenize(text)
    assert tokenized == ['Hello World']

    encoded = tokenizer.encode(text)
    assert encoded == [72, 101, 108, 108, 111, 32, 87, 111, 114, 108, 100]

    decoded = tokenizer.decode(encoded)
    assert text == decoded

    with pytest.raises(
        NotImplementedError,
    ):  # Test that ``__call__`` method throws NotImplementedError
        tokenizer(text=text)
