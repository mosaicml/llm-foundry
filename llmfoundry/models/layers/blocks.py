# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""GPT Blocks used for the GPT Model."""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from llmfoundry.layers_registry import ffns_with_norm
from llmfoundry.models.layers.layer_builders import (build_attention_layer,
                                                     build_ffn, build_norm)

try:
    from flash_attn.bert_padding import unpad_input, pad_input  # type: ignore # yapf: disable # isort: skip
except:
    unpad_input, pad_input = None, None

attn_config_defaults: Dict = {
    'attn_type': 'multihead_attention',
    'attn_pdrop': 0.0,
    'attn_impl': 'flash',
    'qk_ln': False,
    'qk_gn': False,
    'clip_qkv': None,
    'softmax_scale': None,
    'attn_uses_sequence_id': False,
    'sliding_window_size': -1,
    'alibi': False,
    'alibi_bias_max': 8,
    'rope': False,
    'rope_theta': 10000,
    'rope_impl': 'dail',
    'rope_dail_config': {
        'type': 'original',
        'pos_idx_in_fp32': True,
        'xpos_scale_base': 512,
    },
    'rope_hf_config': {
        'type': 'no_scaling',
        'factor': 1.0,
    },
}


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
        no_bias: bool = False,
        use_pad_tok_in_ffn: bool = True,
        **kwargs: Any,
    ):
        if attn_config is None:
            attn_config = attn_config_defaults

        if ffn_config is None:
            ffn_config = {
                'ffn_type': 'mptmlp',
            }
        self.fuse_norm_attn_norm = kwargs.get('fuse_norm_attn_norm', False)

        del kwargs  # unused, just to capture any extra args from the config
        super().__init__()

        ffn_type = ffn_config['ffn_type']

        self.ffn = build_ffn(
            name=ffn_type,
            d_model=d_model,
            expansion_ratio=expansion_ratio,
            device=device,
            bias=not no_bias,
            ffn_kwargs=ffn_config,
        )

        if self.fuse_norm_attn_norm:
            self.norm_attn_norm = FusedNormAttentionNorm(
                d_model=d_model,
                n_heads=n_heads,
                attn_config=attn_config,
                ffn_type=ffn_type,
                fc_type=fc_type,
                resid_pdrop=resid_pdrop,
                norm_type=norm_type,
                device=device,
                no_bias=no_bias,
            )
        else:
            assert isinstance(attn_config['attn_type'], str)
            # Necessary to avoid passing extraneous args into attn_class while allowing the use of **kwargs
            args_to_exclude_in_attn_class = {
                'attn_type', 'alibi', 'attn_uses_sequence_id', 'alibi_bias_max',
                'rope', 'rope_theta', 'rope_impl', 'rope_dail_config',
                'rope_hf_config'
            }
            attn_config_subset_for_attn_class = {
                k: v
                for k, v in attn_config.items()
                if k not in args_to_exclude_in_attn_class
            }

            self.norm_1 = build_norm(
                name=norm_type.lower(),
                normalized_shape=d_model,
                device=device,
            )
            self.attn = build_attention_layer(
                name=attn_config['attn_type'],
                attn_kwargs={
                    'd_model': d_model,
                    'n_heads': n_heads,
                    'fc_type': fc_type,
                    'device': device,
                    'bias': not no_bias,
                    **attn_config_subset_for_attn_class
                },
            )
            self.norm_2 = None
            if not getattr(self.ffn, '_has_norm', False):
                self.norm_2 = build_norm(
                    name=norm_type.lower(),
                    normalized_shape=d_model,
                    device=device,
                )

        self.resid_attn_dropout = nn.Dropout(resid_pdrop)
        self.resid_ffn_dropout = nn.Dropout(resid_pdrop)
        self.use_pad_tok_in_ffn = use_pad_tok_in_ffn

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attn_bias: Optional[torch.Tensor] = None,
        rotary_emb_w_meta_info: Optional[Dict] = None,
        attention_mask: Optional[torch.ByteTensor] = None,
        is_causal: bool = True,
        output_attentions: bool = False,
        alibi_slopes: Optional[torch.Tensor] = None,
        flash_attn_padding_info: Optional[dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[
            torch.Tensor, torch.Tensor]]]:
        if self.fuse_norm_attn_norm:
            x, m, attn_weights, past_key_value = self.norm_attn_norm(
                x,
                past_key_value=past_key_value,
                attn_bias=attn_bias,
                rotary_emb_w_meta_info=rotary_emb_w_meta_info,
                attention_mask=attention_mask,
                is_causal=is_causal,
                output_attentions=output_attentions,
                alibi_slopes=alibi_slopes,
                flash_attn_padding_info=flash_attn_padding_info,
            )
        else:
            a = self.norm_1(x)
            b, attn_weights, past_key_value = self.attn(
                a,
                past_key_value=past_key_value,
                attn_bias=attn_bias,
                rotary_emb_w_meta_info=rotary_emb_w_meta_info,
                attention_mask=attention_mask,
                is_causal=is_causal,
                needs_weights=output_attentions,
                alibi_slopes=alibi_slopes,
                flash_attn_padding_info=flash_attn_padding_info,
            )
            x = x + self.resid_attn_dropout(b)
            m = x
            if self.norm_2 is not None:
                m = self.norm_2(x)

        batch_size, seq_len = m.size()[:2]
        indices = None
        if not self.use_pad_tok_in_ffn:
            assert unpad_input is not None
            m, indices, _, _ = unpad_input(m, attention_mask)
        n = self.ffn(m)
        if not self.use_pad_tok_in_ffn:
            assert pad_input is not None
            n = pad_input(n, indices, batch_size, seq_len)
        x = x + self.resid_ffn_dropout(n)
        return x, attn_weights, past_key_value


class FusedNormAttentionNorm(nn.Module):

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        attn_config: Optional[Dict] = None,
        ffn_type: str = 'mptmlp',
        fc_type: str = 'torch',
        resid_pdrop: float = 0.0,
        norm_type: str = 'low_precision_layernorm',
        device: Optional[str] = None,
        no_bias: bool = False,
        **kwargs: Any,
    ):
        super().__init__()
        assert attn_config is not None
        assert isinstance(attn_config['attn_type'], str)

        # necessary to avoid passing extraneous args into attn_class while allowing the use of **kwargs
        args_to_exclude_in_attn_class = {
            'attn_type', 'alibi', 'attn_uses_sequence_id', 'alibi_bias_max',
            'rope', 'rope_theta', 'rope_impl', 'rope_dail_config',
            'rope_hf_config'
        }
        attn_config_subset_for_attn_class = {
            k: v
            for k, v in attn_config.items()
            if k not in args_to_exclude_in_attn_class
        }
        self.norm_1 = build_norm(
            name=norm_type.lower(),
            normalized_shape=d_model,
            device=device,
        )
        self.attn = build_attention_layer(
            name=attn_config['attn_type'],
            attn_kwargs={
                'd_model': d_model,
                'n_heads': n_heads,
                'fc_type': fc_type,
                'device': device,
                'bias': not no_bias,
                **attn_config_subset_for_attn_class
            },
        )

        self.norm_2 = None
        if not ffn_type in ffns_with_norm:
            self.norm_2 = build_norm(
                name=norm_type.lower(),
                normalized_shape=d_model,
                device=device,
            )
        self.resid_attn_dropout = nn.Dropout(resid_pdrop)

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attn_bias: Optional[torch.Tensor] = None,
        rotary_emb_w_meta_info: Optional[Dict] = None,
        attention_mask: Optional[torch.ByteTensor] = None,
        is_causal: bool = True,
        output_attentions: bool = False,
        alibi_slopes: Optional[torch.Tensor] = None,
        flash_attn_padding_info: Optional[dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        a = self.norm_1(x)
        b, attn_weights, past_key_value = self.attn(
            a,
            past_key_value=past_key_value,
            attn_bias=attn_bias,
            rotary_emb_w_meta_info=rotary_emb_w_meta_info,
            attention_mask=attention_mask,
            is_causal=is_causal,
            needs_weights=output_attentions,
            alibi_slopes=alibi_slopes,
            flash_attn_padding_info=flash_attn_padding_info,
        )
        x = x + self.resid_attn_dropout(b)
        m = x
        if self.norm_2 is not None:
            m = self.norm_2(x)

        return x, m, attn_weights, past_key_value
