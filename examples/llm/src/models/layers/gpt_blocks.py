# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""GPT Blocks used for the GPT Model."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from composer.algorithms.low_precision_layernorm.low_precision_layernorm import \
    LPLayerNorm

from examples.llm.src.models.layers.attention import MultiheadAttention


class GPTMLP(nn.Module):

    def __init__(self,
                 d_model: int,
                 mlp_ratio: int,
                 device: Optional[str] = None):
        super().__init__()
        self.mlp_up = nn.Linear(d_model, mlp_ratio * d_model, device=device)
        self.mlp_act = nn.GELU(approximate='none')
        self.mlp_down = nn.Linear(mlp_ratio * d_model, d_model, device=device)
        self.mlp_down._is_residual = True  # type: ignore

    def forward(self, x):
        return self.mlp_down(self.mlp_act(self.mlp_up(x)))


class GPTBlock(nn.Module):

    def __init__(self,
                 attn_impl: str,
                 d_model: int,
                 n_heads: int,
                 mlp_ratio: int,
                 attn_clip_qkv: Optional[float] = None,
                 attn_qk_ln: bool = False,
                 softmax_scale: Optional[float] = None,
                 attn_pdrop: float = 0.0,
                 alibi: bool = False,
                 resid_pdrop: float = 0.0,
                 low_precision_layernorm: bool = False,
                 device: Optional[str] = None,
                 **kwargs):
        del kwargs  # unused, just to capture any extra args from the config
        super().__init__()

        layernorm_class = LPLayerNorm if low_precision_layernorm else nn.LayerNorm

        self.ln_1 = layernorm_class(d_model, device=device)
        self.attn = MultiheadAttention(
            attn_impl=attn_impl,
            attn_clip_qkv=attn_clip_qkv,
            attn_qk_ln=attn_qk_ln,
            softmax_scale=softmax_scale,
            attn_pdrop=attn_pdrop,
            d_model=d_model,
            n_heads=n_heads,
            device=device,
        )
        self.ln_2 = layernorm_class(d_model, device=device)
        self.mlp = GPTMLP(
            d_model=d_model,
            mlp_ratio=mlp_ratio,
            device=device,
        )
        self.resid_attn_dropout = nn.Dropout(resid_pdrop)
        self.resid_mlp_dropout = nn.Dropout(resid_pdrop)

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attn_bias: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.ByteTensor] = None,
        is_causal: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        a = self.ln_1(x)
        b, _, past_key_value = self.attn(a,
                                         past_key_value=past_key_value,
                                         attn_bias=attn_bias,
                                         attention_mask=attention_mask,
                                         is_causal=is_causal)
        x = x + self.resid_attn_dropout(b)
        m = self.ln_2(x)
        n = self.mlp(m)
        x = x + self.resid_mlp_dropout(n)
        return x, past_key_value
