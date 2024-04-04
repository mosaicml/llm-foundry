# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from typing import Type

import torch

from llmfoundry.utils.registry_utils import create_registry

# Layers
_norm_description = """The norms registry is used to register classes that implement normalization layers."""
norms = create_registry('llmfoundry',
                        'norms',
                        generic_type=Type[torch.nn.Module],
                        entry_points=True,
                        description=_norm_description)

_fc_description = """The fully connected layers registry is used to register classes that implement fully connected layers."""
fcs = create_registry('llmfoundry',
                      'fcs',
                      generic_type=Type[torch.nn.Module],
                      entry_points=True,
                      description=_fc_description)

__all__ = [
    'norms',
    'fcs',
]
