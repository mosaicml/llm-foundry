# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Union

import torch

from llmfoundry.layers_registry import norms

__all__ = [
    'LPLayerNorm',
    'LPRMSNorm',
    'RMSNorm',
    'TritonRMSNorm',
    'rms_norm',
]

norms.register(name='layernorm', func=torch.nn.LayerNorm)


def _cast_if_autocast_enabled(tensor: torch.Tensor) -> torch.Tensor:
    if torch.is_autocast_enabled():
        if tensor.device.type == 'cuda':
            dtype = torch.get_autocast_gpu_dtype()
        elif tensor.device.type == 'cpu':
            dtype = torch.get_autocast_cpu_dtype()
        else:
            raise NotImplementedError()
        return tensor.to(dtype=dtype)
    return tensor


@norms.register_class('low_precision_layernorm')
class LPLayerNorm(torch.nn.LayerNorm):

    def __init__(
        self,
        normalized_shape: Union[int, List[int], torch.Size],
        eps: float = 1e-05,
        elementwise_affine: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        module_device = x.device
        downcast_x = _cast_if_autocast_enabled(x)
        downcast_weight = _cast_if_autocast_enabled(
            self.weight) if self.weight is not None else self.weight
        downcast_bias = _cast_if_autocast_enabled(
            self.bias) if self.bias is not None else self.bias
        with torch.autocast(enabled=False, device_type=module_device.type):
            return torch.nn.functional.layer_norm(
                downcast_x,
                self.normalized_shape,
                downcast_weight,
                downcast_bias,
                self.eps,
            )


def rms_norm(x: torch.Tensor,
             weight: Optional[torch.Tensor] = None,
             eps: float = 1e-5) -> torch.Tensor:
    output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        return output * weight
    return output


@norms.register_class('rmsnorm')
class RMSNorm(torch.nn.Module):

    def __init__(
        self,
        normalized_shape: Union[int, List[int], torch.Size],
        eps: float = 1e-5,
        weight: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.eps = eps
        if weight:
            self.weight = torch.nn.Parameter(
                torch.ones(normalized_shape, dtype=dtype, device=device))
        else:
            self.register_parameter('weight', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x.float(), self.weight, self.eps).to(dtype=x.dtype)


@norms.register_class('low_precision_rmsnorm')
class LPRMSNorm(RMSNorm):

    def __init__(
        self,
        normalized_shape: Union[int, List[int], torch.Size],
        eps: float = 1e-5,
        weight: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(
            normalized_shape=normalized_shape,
            eps=eps,
            weight=weight,
            dtype=dtype,
            device=device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        downcast_x = _cast_if_autocast_enabled(x)
        downcast_weight = _cast_if_autocast_enabled(
            self.weight) if self.weight is not None else self.weight
        with torch.autocast(enabled=False, device_type=x.device.type):
            return rms_norm(downcast_x, downcast_weight,
                            self.eps).to(dtype=x.dtype)


@norms.register_class('triton_rmsnorm')
class TritonRMSNorm(torch.nn.Module):

    def __init__(
        self,
        normalized_shape: Union[int, List[int], torch.Size],
        eps: float = 1e-5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.eps = eps

        try:
            from flash_attn.ops.triton.layer_norm import rms_norm_fn
        except ImportError:
            raise ImportError(
                'triton_rms_norm requires Flash Attention to be installed. ' +
                'Please pip install flash-attn.')

        if not isinstance(normalized_shape, int):
            raise ValueError('TritonRMSNorm only supports 1D tensors')

        self.rms_norm_fn = rms_norm_fn

        self.weight = torch.nn.Parameter(
            torch.ones(normalized_shape, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor):
        # Flash Attention expect a flat tensor
        return self.rms_norm_fn(
            x,
            self.weight,
            None,  # no bias
            residual=None,
            eps=self.eps,
            dropout_p=0.0,  # no dropout by default
            prenorm=False,
            residual_in_fp32=False,
        )
