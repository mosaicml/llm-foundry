# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""GPT Blocks used for the GPT Model."""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from llmfoundry.models.layers.attention import ATTN_CLASS_REGISTRY
from llmfoundry.models.layers.ffn import FFN_CLASS_REGISTRY, build_ffn
from llmfoundry.models.layers.norm import NORM_CLASS_REGISTRY


class MPTBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        expansion_ratio: int,
        attn_config: Optional[Dict] = None,
        ffn_config: Optional[Dict] = None,
        resid_pdrop: float = 0.0,
        norm_type: str = 'low_precision_layernorm',
        fc_type: str = 'torch',
        device: Optional[str] = None,
        **kwargs: Any,
    ):
        if attn_config is None:
            attn_config = {
                'attn_type': 'multihead_attention',
                'attn_pdrop': 0.0,
                'attn_impl': 'triton',
                'qk_ln': False,
                'clip_qkv': None,
                'softmax_scale': None,
                'prefix_lm': False,
                'attn_uses_sequence_id': False,
                'alibi': False,
                'alibi_bias_max': 8,
            }

        if ffn_config is None:
            ffn_config = {
                'ffn_type': 'mptmlp',
            }

        del kwargs  # unused, just to capture any extra args from the config
        super().__init__()

        norm_class = NORM_CLASS_REGISTRY[norm_type.lower()]
        assert isinstance(attn_config['attn_type'], str)
        attn_class = ATTN_CLASS_REGISTRY[attn_config['attn_type']]

        # necessary to avoid passing extraneous args into attn_class while allowing the use of **kwargs
        args_to_exclude_in_attn_class = {
            'attn_type', 'prefix_lm', 'alibi', 'attn_uses_sequence_id',
            'alibi_bias_max'
        }
        attn_config_subset_for_attn_class = {
            k: v
            for k, v in attn_config.items()
            if k not in args_to_exclude_in_attn_class
        }

        self.norm_1 = norm_class(d_model, device=device)
        self.attn = attn_class(d_model=d_model,
                               n_heads=n_heads,
                               fc_type=fc_type,
                               device=device,
                               **attn_config_subset_for_attn_class)
        self.norm_2 = None
        if not getattr(FFN_CLASS_REGISTRY[ffn_config['ffn_type']], '_has_norm',
                       False):
            self.norm_2 = norm_class(d_model, device=device)
        self.ffn = build_ffn(
            d_model=d_model,
            expansion_ratio=expansion_ratio,
            device=device,
            **ffn_config,
        )
        self.resid_attn_dropout = nn.Dropout(resid_pdrop)
        self.resid_ffn_dropout = nn.Dropout(resid_pdrop)

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attn_bias: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.ByteTensor] = None,
        is_causal: bool = True,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[
            torch.Tensor, torch.Tensor]]]:
        a = self.norm_1(x)
        b, attn_weights, past_key_value = self.attn(
            a,
            past_key_value=past_key_value,
            attn_bias=attn_bias,
            attention_mask=attention_mask,
            is_causal=is_causal,
            needs_weights=output_attentions,
        )
        x = x + self.resid_attn_dropout(b)
        m = x
        if self.norm_2 is not None:
            m = self.norm_2(x)
        n = self.ffn(m)
        x = x + self.resid_ffn_dropout(n)
        return x, attn_weights, past_key_value
