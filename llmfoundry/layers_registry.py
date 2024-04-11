# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Type

import torch

from llmfoundry.utils.registry_utils import create_registry

# Layers
_norm_description = """The norms registry is used to register classes that implement normalization layers."""
norms = create_registry('llmfoundry',
                        'norms',
                        generic_type=Type[torch.nn.Module],
                        entry_points=True,
                        description=_norm_description)

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

__all__ = [
    'norms',
    'ffns',
    'ffns_with_norm',
    'ffns_with_megablocks',
]
