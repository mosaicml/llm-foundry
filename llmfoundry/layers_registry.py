import torch

from typing import Type

from llmfoundry.utils.registry_utils import create_registry

# Layers
_norm_description = """The norms registry is used to register classes that implement normalization layers."""
norms = create_registry('llmfoundry',
                        'norms',
                        generic_type=Type[torch.nn.Module],
                        entry_points=True,
                        description=_norm_description)

__all__ = [
    'norms',
]