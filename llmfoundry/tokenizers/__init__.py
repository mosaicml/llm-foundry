# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from llmfoundry.registry import tokenizers
from llmfoundry.tokenizers.tiktoken import TiktokenTokenizerWrapper
from llmfoundry.tokenizers.utils import get_date_string

tokenizers.register('tiktoken', func=TiktokenTokenizerWrapper)

__all__ = [
    'TiktokenTokenizerWrapper',
    'get_date_string',
]
