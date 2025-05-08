# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional, Union

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
    normalized_shape: Union[int, list[int], torch.Size],
    eps: Optional[float] = 1e-5,
    device: Optional[str] = None,
):
    kwargs = {
        'normalized_shape': normalized_shape,
        'eps': eps,
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
    ffn_kwargs: dict[str, Any],
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
    attn_kwargs: dict[str, Any],
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
    fc_kwargs: dict[str, Any],
):
    kwargs = {
        'in_features': in_features,
        'out_features': out_features,
        **{k: v for k, v in fc_kwargs.items() if k != 'name'},
    }

    module = construct_from_registry(
        name=name,
        registry=fcs,
        pre_validation_function=torch.nn.Module,
        kwargs=kwargs,
    )
    # from torch._C._distributed_c10d import ReduceOp
    # if isinstance(module, torch.nn.Linear):
    #     def post_hook_fn(module, grad_input, grad_output):
    #         print('grad_input: ', grad_input[0].norm().item(), 'grad_output: ', grad_output[0].norm().item())
    #     def forward_hook_fn(module, input, output):
    #         print('module: ', module, 'input: ', input[0].norm().item(), 'output: ', output.norm().item())
    #     def weight_hook(grad):
    #         grad = grad.clone().detach().reshape(-1)
    #         # print('weight_grad: ', grad.norm().item())
    #         # grad_sc_output = grad.new_empty(grad.shape[0] // 2)
    #         # print('reduce scatter dtype: ', grad_sc_output.dtype, grad.dtype)
    #         # torch.distributed.reduce_scatter_tensor(grad_sc_output, grad, op=ReduceOp.AVG)
    #         # print('local sc weight grad: ', grad_sc_output.float().norm().item())
    #         torch.distributed.all_reduce(grad, op=ReduceOp.AVG)
    #         print('weight_grad: ', grad.float().norm().item())
    #     def bias_hook(grad):
    #         grad = grad.clone().detach().reshape(-1)    
    #         # print('bias_grad: ', grad.norm().item())
    #         # grad_sc_output = grad.new_empty(grad.shape[0] // 2)
    #         # torch.distributed.reduce_scatter_tensor(grad_sc_output, grad, op=ReduceOp.AVG)
    #         # print('local sc bias grad: ', grad_sc_output.float().norm().item())
    #         torch.distributed.all_reduce(grad, op=ReduceOp.AVG)
    #         print('bias_grad: ', grad.float().norm().item())
    #     def pre_backward_hook_fn(module: torch.nn.Linear, grad_output: torch.Tensor):
    #         print('module: ', module)
    #         if not module.weight._backward_hooks:
    #             module.weight.register_hook(weight_hook)
    #         if not module.bias._backward_hooks:
    #             module.bias.register_hook(bias_hook)

    #     module.register_full_backward_hook(post_hook_fn)
    #     module.register_forward_hook(forward_hook_fn)
    #     module.register_full_backward_pre_hook(pre_backward_hook_fn)

    return module
