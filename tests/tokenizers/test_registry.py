# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional

from transformers import PreTrainedTokenizer

from llmfoundry.registry import tokenizers
from llmfoundry.utils import build_tokenizer


class DummyTokenizer(PreTrainedTokenizer):
    """A dummy tokenizer that inherits from ``PreTrainedTokenizer``."""

    def __init__(
        self,
        model_name: Optional[str] = 'dummy',
        **kwargs: Optional[dict[str, Any]],
    ):
        """Dummy constructor that has no real purpose."""
        super().__init__(
            model_name=model_name,
            eos_token='0',
            pad_token='1',
            **kwargs,
        )

    def get_vocab(self) -> dict[str, int]:
        return {}


def test_tokenizer_registry():
    tokenizers.register('dummy', func=DummyTokenizer)
    tokenizer = build_tokenizer(tokenizer_name='dummy', tokenizer_kwargs={})
    assert type(tokenizer) == DummyTokenizer
