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

_param_init_fns_description = """The param_init_fns registry is used to register functions that initialize parameters."""
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
]
