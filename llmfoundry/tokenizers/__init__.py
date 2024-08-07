# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from llmfoundry.registry import tokenizers
from llmfoundry.tokenizers.tiktoken import TiktokenTokenizerWrapper

tokenizers.register('tiktoken', func=TiktokenTokenizerWrapper)

__all__ = [
    'TiktokenTokenizerWrapper',
]
