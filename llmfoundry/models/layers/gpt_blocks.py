# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""GPT Blocks used for the GPT Model."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from llmfoundry.models.layers.attention import ATTN_CLASS_REGISTRY
from llmfoundry.models.layers.norm import NORM_CLASS_REGISTRY


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

    def __init__(
            self,
            d_model: int,
            mlp_ratio: int,
            attn_config: Dict = {
                'attn_type': 'multihead_attention',
                'n_heads': 16,
                'attn_pdrop': 0.0,
                'attn_impl': 'triton',
                'qk_ln': False,
                'clip_qkv': None,
                'softmax_scale': None,
                'prefix_lm': False,
                'attn_uses_sequence_id': False,
                'alibi': False,
                'alibi_bias_max': 8,
            },
            resid_pdrop: float = 0.0,
            norm_type: str = 'low_precision_layernorm',
            device: Optional[str] = None,
            **kwargs):
        del kwargs  # unused, just to capture any extra args from the config
        super().__init__()

        norm_class = NORM_CLASS_REGISTRY[norm_type.lower()]
        attn_class = ATTN_CLASS_REGISTRY[attn_config['attn_type']]

        self.norm_1 = norm_class(d_model, device=device)
        self.attn = attn_class(
            attn_impl=attn_config['attn_impl'],
            clip_qkv=attn_config['clip_qkv'],
            qk_ln=attn_config['qk_ln'],
            softmax_scale=attn_config['softmax_scale'],
            attn_pdrop=attn_config['attn_pdrop'],
            d_model=d_model,
            n_heads=attn_config['n_heads'],
            device=device,
        )
        self.norm_2 = norm_class(d_model, device=device)
        self.ffn = GPTMLP(
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
        a = self.norm_1(x)
        b, _, past_key_value = self.attn(a,
                                         past_key_value=past_key_value,
                                         attn_bias=attn_bias,
                                         attention_mask=attention_mask,
                                         is_causal=is_causal)
        x = x + self.resid_attn_dropout(b)
        m = self.norm_2(x)
        n = self.ffn(m)
        x = x + self.resid_mlp_dropout(n)
        return x, past_key_value
