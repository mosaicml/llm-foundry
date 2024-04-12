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
_fc_description = (
    'The fully connected layers registry is used to register classes that implement fully connected layers (i.e. torch.nn.Linear).'
    +
    'These classes should take in_features and out_features in as args, at a minimum.'
)
fcs = create_registry('llmfoundry',
                      'fcs',
                      generic_type=Type[torch.nn.Module],
                      entry_points=True,
                      description=_fc_description)

_ffns_description = (
    'The ffns registry is used to register functions that build ffn layers.' +
    'See ffn.py for examples.')
ffns = create_registry('llmfoundry',
                       'ffns',
                       generic_type=Callable,
                       entry_points=True,
                       description=_ffns_description)

_ffns_with_norm_description = (
    'The ffns_with_norm registry is used to register functions that build ffn layers that apply a normalization layer.'
    + 'See ffn.py for examples.')
ffns_with_norm = create_registry('llmfoundry',
                                 'ffns_with_norm',
                                 generic_type=Callable,
                                 entry_points=True,
                                 description=_ffns_with_norm_description)

_ffns_with_megablocks_description = (
    'The ffns_with_megablocks registry is used to register functions that build ffn layers using MegaBlocks.'
    + 'See ffn.py for examples.')
ffns_with_megablocks = create_registry(
    'llmfoundry',
    'ffns_with_megablocks',
    generic_type=Callable,
    entry_points=True,
    description=_ffns_with_megablocks_description)

_attention_classes_description = (
    'The attention_classes registry is used to register classes that implement attention layers. See '
    + 'attention.py for expected constructor signature.')
attention_classes = create_registry('llmfoundry',
                                    'attention_classes',
                                    generic_type=Type[torch.nn.Module],
                                    entry_points=True,
                                    description=_attention_classes_description)

_attention_implementations_description = (
    'The attention_implementations registry is used to register functions that implement the attention operation.'
    + 'See attention.py for expected function signature.')
attention_implementations = create_registry(
    'llmfoundry',
    'attention_implementations',
    generic_type=Callable,
    entry_points=True,
    description=_attention_implementations_description)

__all__ = [
    'norms',
    'ffns',
    'ffns_with_norm',
    'ffns_with_megablocks',
    'attention_classes',
    'attention_implementations',
    'fcs',
]
