# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""GPT Blocks used for the GPT Model."""

from typing import Optional

import torch
import torch.nn as nn
from composer.algorithms.low_precision_layernorm.low_precision_layernorm import \
    LPLayerNorm
from omegaconf import DictConfig

from examples.llm.src.models.layers.attention import MultiheadAttention


class GPTMLP(nn.Module):

    def __init__(self, cfg: DictConfig, device: Optional[str] = None):
        super().__init__()
        self.mlp_up = nn.Linear(cfg.d_model,
                                cfg.mlp_ratio * cfg.d_model,
                                device=device)
        self.mlp_act = nn.GELU(approximate='none')
        self.mlp_down = nn.Linear(cfg.mlp_ratio * cfg.d_model,
                                  cfg.d_model,
                                  device=device)
        self.mlp_down._is_residual = True  # type: ignore

    def forward(self, x):
        return self.mlp_down(self.mlp_act(self.mlp_up(x)))


class GPTBlock(nn.Module):

    def __init__(self, cfg: DictConfig, device: Optional[str] = None):
        super().__init__()
        layernorm_class = LPLayerNorm if cfg.get('low_precision_layernorm',
                                                 False) else nn.LayerNorm

        self.ln_1 = layernorm_class(cfg.d_model, device=device)
        self.attn = MultiheadAttention(cfg, device)
        self.ln_2 = layernorm_class(cfg.d_model, device=device)
        self.mlp = GPTMLP(cfg, device=device)
        self.resid_attn_dropout = nn.Dropout(cfg.resid_pdrop)
        self.resid_mlp_dropout = nn.Dropout(cfg.resid_pdrop)

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.ByteTensor] = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        a = self.ln_1(x)
        b, _ = self.attn(a,
                         attn_bias=attn_bias,
                         key_padding_mask=key_padding_mask,
                         is_causal=is_causal)
        x = x + self.resid_attn_dropout(b)
        m = self.ln_2(x)
        n = self.mlp(m)
        x = x + self.resid_mlp_dropout(n)
        return x
