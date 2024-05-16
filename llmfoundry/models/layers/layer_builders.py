# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional, Union

import torch

from llmfoundry.layers_registry import (
    attention_classes,
    fcs,
    ffns,
    ffns_with_megablocks,
    ffns_with_norm,
    norms,
)
from llmfoundry.utils.registry_utils import construct_from_registry

__all__ = [
    'build_attention_layer',
    'build_ffn',
    'build_fc',
    'build_norm',
]


def build_norm(
    name: str,
    normalized_shape: Union[int, List[int], torch.Size],
    device: Optional[str] = None,
):
    kwargs = {
        'normalized_shape': normalized_shape,
        'device': device,
    }

    return construct_from_registry(
        name=name,
        registry=norms,
        pre_validation_function=torch.nn.Module,
        kwargs=kwargs,
    )


def build_ffn(
    name: str,
    d_model: int,
    expansion_ratio: float,
    device: Optional[str],
    bias: bool,
    ffn_kwargs: Dict[str, Any],
):

    registry_to_use = ffns
    if name in ffns_with_norm:
        registry_to_use = ffns_with_norm

    if name in ffns_with_megablocks:
        registry_to_use = ffns_with_megablocks

    kwargs = {
        'd_model': d_model,
        'expansion_ratio': expansion_ratio,
        'device': device,
        'bias': bias,
        **{k: v for k, v in ffn_kwargs.items() if k != 'ffn_type'},
    }

    def _validation_function(maybe_module: Any):
        if not isinstance(maybe_module, torch.nn.Module):
            raise ValueError(f'Function {name} must return a torch.nn.Module.')

    result = construct_from_registry(
        name=name,
        registry=registry_to_use,
        post_validation_function=_validation_function,
        partial_function=False,
        kwargs=kwargs,
    )

    if name in ffns_with_norm:
        result._has_norm = True

    if name in ffns_with_megablocks:
        result._uses_megablocks = True

    return result


def build_attention_layer(
    name: str,
    attn_kwargs: Dict[str, Any],
):
    return construct_from_registry(
        name=name,
        registry=attention_classes,
        pre_validation_function=torch.nn.Module,
        kwargs=attn_kwargs,
    )


def build_fc(
    name: str,
    in_features: int,
    out_features: int,
    fc_kwargs: Dict[str, Any],
):
    kwargs = {
        'in_features': in_features,
        'out_features': out_features,
        **{k: v for k, v in fc_kwargs.items() if k != 'name'},
    }

    return construct_from_registry(
        name=name,
        registry=fcs,
        pre_validation_function=torch.nn.Module,
        kwargs=kwargs,
    )
