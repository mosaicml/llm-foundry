# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Type

import torch

from llmfoundry.utils.registry_utils import create_registry

_norm_description = (
    'The norms registry is used to register classes that implement normalization layers.'
)
norms = create_registry('llmfoundry',
                        'norms',
                        generic_type=Type[torch.nn.Module],
                        entry_points=True,
                        description=_norm_description)

_attention_class_description = (
    'The attention_class registry is used to register classes that implement attention layers. See '
    + 'attention.py for expected constructor signature.')
attention_class = create_registry('llmfoundry',
                                  'attention_class',
                                  generic_type=Type[torch.nn.Module],
                                  entry_points=True,
                                  description=_attention_class_description)

_attention_implementation_description = (
    'The attention_implementation registry is used to register functions that implement the attention operation.'
    + 'See attention.py for expected function signature.')
attention_implementation = create_registry(
    'llmfoundry',
    'attention_implementation',
    generic_type=Callable,
    entry_points=True,
    description=_attention_implementation_description)

__all__ = [
    'norms',
    'attention_class',
    'attention_implementation',
]
