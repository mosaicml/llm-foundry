# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""GPT Blocks used for the GPT Model."""

from typing import Optional

import torch
import torch.nn as nn

from llmfoundry.models.layers.attention import ATTN_CLASS_REGISTRY
from llmfoundry.models.layers.fc import FC_CLASS_REGISTRY
from llmfoundry.models.layers.norm import NORM_CLASS_REGISTRY

try:
    import transformer_engine.pytorch as te
except:
    te = None


class MPTMLP(nn.Module):

    def __init__(
        self,
        d_model: int,
        expansion_ratio: int,
        fc_type: str = 'torch',
        device: Optional[str] = None,
    ):
        super().__init__()
        fc_kwargs = {}
        if fc_type != 'te':
            fc_kwargs['device'] = device
        self.up_proj = FC_CLASS_REGISTRY[fc_type](
            d_model,
            expansion_ratio * d_model,
            **fc_kwargs,
        )
        self.act = nn.GELU(approximate='none')
        self.down_proj = FC_CLASS_REGISTRY[fc_type](
            expansion_ratio * d_model,
            d_model,
            **fc_kwargs,
        )
        self.down_proj._is_residual = True  # type: ignore

    def forward(self, x):
        return self.down_proj(self.act(self.up_proj(x)))


FFN_CLASS_REGISTRY = {
    'mptmlp': MPTMLP,
}

if te is not None:
    te.LayerNormMLP._has_norm = True
    FFN_CLASS_REGISTRY['te_ln_mlp'] = te.LayerNormMLP


def build_ffn(
    d_model: int,
    expansion_ratio: int,
    fc_type: str = 'torch',
    device: Optional[str] = None,
    **kwargs,
):
    ffn_type = kwargs.pop('ffn_type')
    if ffn_type == 'mptmlp':
        if kwargs is not None and len(kwargs) > 0:
            raise ValueError(
                f'MPTMLP got an unexpected keyword argument: {kwargs}')
        return MPTMLP(
            d_model=d_model,
            expansion_ratio=expansion_ratio,
            fc_type=fc_type,
            device=device,
        )
    elif ffn_type == 'te_ln_mlp':
        return te.LayerNormMLP(
            hidden_size=d_model,
            ffn_hidden_size=d_model * expansion_ratio,
            **kwargs,
        )

    raise ValueError(f'{ffn_type=} not recognized.')
