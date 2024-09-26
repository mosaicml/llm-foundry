# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from llmfoundry.registry import tp_strategies
from llmfoundry.tp.ffn_tp_strategy import ffn

tp_strategies.register('ffn', func=ffn)

__all__ = [
    'ffn',
]
