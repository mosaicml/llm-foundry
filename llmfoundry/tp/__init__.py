# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from llmfoundry.registry import tp_strategies
from llmfoundry.tp.tp_strategies import ffn_tp_strategies

tp_strategies.register('ffn', func=ffn_tp_strategies)

__all__ = [
    'ffn_tp_strategies',
]
