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

_param_init_fns_description = (
    'The param_init_fns registry is used to register functions that initialize parameters.'
    +
    'These will be called on a module to initialize its parameters. See param_init_fns.py for examples.'
)
param_init_fns = create_registry('llmfoundry',
                                 'param_init_fns',
                                 generic_type=Callable[..., None],
                                 entry_points=True,
                                 description=_param_init_fns_description)

_module_init_fns_description = """The module_init_fns registry is used to register functions that initialize specific modules.
These functions should return True if they initialize the module, and False otherwise. This allows them to be called without knowing their contents.
They should take in the module, init_div_is_residual, and div_is_residual arguments."""
module_init_fns = create_registry('llmfoundry',
                                  'module_init_fns',
                                  generic_type=Callable[..., bool],
                                  entry_points=True,
                                  description=_module_init_fns_description)

__all__ = [
    'norms',
    'param_init_fns',
    'module_init_fns',
    'attention_classes',
    'attention_implementations',
    'fcs',
]
