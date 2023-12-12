# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""MPT Blocks used for the MPT Model."""

import logging
from typing import Any, Optional, Union

import torch
import torch.nn as nn

from llmfoundry.models.layers.fc import FC_CLASS_REGISTRY

try:
    import transformer_engine.pytorch as te
except:
    te = None

log = logging.getLogger(__name__)


def _resolve_ffn_hidden_and_exp_ratio(
    d_model: int,
    expansion_ratio: Union[int, float],
    ffn_hidden_size: Optional[int] = None,
) -> tuple[Union[int, float], int]:
    if ffn_hidden_size is not None:
        log.info(
            f'`expansion_ratio` (={expansion_ratio}) ignored when `ffn_hidden_size` (={ffn_hidden_size}) is specified.'
        )
    else:
        ffn_hidden_size = int(d_model * expansion_ratio)
        if ffn_hidden_size != d_model * expansion_ratio:
            raise ValueError(
                f'`d_model * expansion_ratio` ({ffn_hidden_size}) must be an integer.'
            )
    return expansion_ratio, ffn_hidden_size


class MPTMLP(nn.Module):

    def __init__(
        self,
        d_model: int,
        expansion_ratio: Union[int, float],
        fc_type: str = 'torch',
        ffn_hidden_size: Optional[int] = None,
        device: Optional[str] = None,
        bias: bool = True,
    ):
        super().__init__()
        expansion_ratio, ffn_hidden_size = _resolve_ffn_hidden_and_exp_ratio(
            d_model, expansion_ratio, ffn_hidden_size)
        self.fc_kwargs: dict[str, Any] = {
            'bias': bias,
        }
        if fc_type != 'te':
            self.fc_kwargs['device'] = device

        self.up_proj = FC_CLASS_REGISTRY[fc_type](
            d_model,
            ffn_hidden_size,
            **self.fc_kwargs,
        )
        self.act = nn.GELU(approximate='none')
        self.down_proj = FC_CLASS_REGISTRY[fc_type](
            ffn_hidden_size,
            d_model,
            **self.fc_kwargs,
        )
        self.down_proj._is_residual = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.up_proj(x)))


class MPTGeGLU(MPTMLP):

    def __init__(
        self,
        d_model: int,
        expansion_ratio: Union[int, float],
        fc_type: str = 'torch',
        ffn_hidden_size: Optional[int] = None,
        device: Optional[str] = None,
        bias: bool = True,
    ):
        super().__init__(
            d_model=d_model,
            expansion_ratio=expansion_ratio,
            fc_type=fc_type,
            ffn_hidden_size=ffn_hidden_size,
            device=device,
            bias=bias,
        )
        self.gate = FC_CLASS_REGISTRY[fc_type](
            d_model,
            self.up_proj.out_features,
            **self.fc_kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.up_proj(x)) * self.gate(x))


FFN_CLASS_REGISTRY = {
    'mptmlp': MPTMLP,
    'mptgeglu': MPTGeGLU,
}

if te is not None:
    te.LayerNormMLP._has_norm = True
    FFN_CLASS_REGISTRY['te_ln_mlp'] = te.LayerNormMLP


def build_ffn(
    d_model: int,
    expansion_ratio: Union[int, float],
    fc_type: str = 'torch',
    ffn_hidden_size: Optional[int] = None,
    device: Optional[str] = None,
    bias: bool = True,
    **kwargs: Any,
) -> nn.Module:
    ffn_type = kwargs.pop('ffn_type')
    if ffn_type in ['mptmlp', 'mptgeglu']:
        if len(kwargs) > 0:
            raise ValueError(
                f'MPTMLP (or MPTGeGLU) got an unexpected keyword argument: {kwargs}'
            )
        return FFN_CLASS_REGISTRY[ffn_type](
            d_model=d_model,
            expansion_ratio=expansion_ratio,
            fc_type=fc_type,
            ffn_hidden_size=ffn_hidden_size,
            device=device,
            bias=bias,
        )
    elif ffn_type == 'te_ln_mlp':
        assert te is not None
        _, ffn_hidden_size = _resolve_ffn_hidden_and_exp_ratio(
            d_model, expansion_ratio, ffn_hidden_size)
        return te.LayerNormMLP(
            hidden_size=d_model,
            ffn_hidden_size=ffn_hidden_size,
            bias=bias,
            **kwargs,
        )

    raise ValueError(f'{ffn_type=} not recognized.')
