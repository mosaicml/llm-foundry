# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""GPT Blocks used for the GPT Model."""

import copy
from typing import Any, Dict, Optional, Set, Tuple

import torch
import torch.nn as nn

from llmfoundry.layers_registry import ffns_with_norm
from llmfoundry.models.layers.layer_builders import (
    build_attention_layer,
    build_ffn,
    build_norm,
)
from llmfoundry.models.utils.config_defaults import (
    attn_config_defaults,
    fc_type_defaults,
)

try:
    from flash_attn.bert_padding import unpad_input, pad_input  # type: ignore # yapf: disable # isort: skip
except:
    unpad_input, pad_input = None, None

__all__ = [
    'MPTBlock',
    'FusedNormAttentionNorm',
]


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
        fc_type: Optional[dict[str, Any]] = None,
        device: Optional[str] = None,
        no_bias: bool = False,
        use_pad_tok_in_ffn: bool = True,
        **kwargs: Any,
    ):
        if attn_config is None:
            attn_config = attn_config_defaults

        if ffn_config is None:
            self.ffn_config: dict[str, Any] = {
                'ffn_type': 'mptmlp',
            }
        else:
            self.ffn_config = ffn_config

        if fc_type is None:
            fc_type = copy.deepcopy(fc_type_defaults)
        fc_type['bias'] = not no_bias
        fc_type['device'] = device

        self.ffn_config['fc_type'] = fc_type

        self.fuse_norm_attn_norm = kwargs.get('fuse_norm_attn_norm', False)

        del kwargs  # unused, just to capture any extra args from the config
        super().__init__()

        ffn_type = self.ffn_config['ffn_type']
        ffn_has_norm = ffn_type in ffns_with_norm

        if self.fuse_norm_attn_norm:
            self.norm_attn_norm = FusedNormAttentionNorm(
                d_model=d_model,
                n_heads=n_heads,
                args_to_exclude_in_attn_class=self.
                args_to_exclude_in_attn_class,
                attn_config=attn_config,
                ffn_has_norm=ffn_has_norm,
                fc_type=fc_type,
                resid_pdrop=resid_pdrop,
                norm_type=norm_type,
                device=device,
                no_bias=no_bias,
            )
        else:
            assert isinstance(attn_config['attn_type'], str)
            # Necessary to avoid passing extraneous args into attn_class while allowing the use of **kwargs
            attn_config_subset_for_attn_class = {
                k: v
                for k, v in attn_config.items()
                if k not in self.args_to_exclude_in_attn_class
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
                    **attn_config_subset_for_attn_class,
                },
            )
            self.norm_2 = None
            if not ffn_has_norm:
                self.norm_2 = build_norm(
                    name=norm_type.lower(),
                    normalized_shape=d_model,
                    device=device,
                )

        self.ffn = build_ffn(
            name=ffn_type,
            d_model=d_model,
            expansion_ratio=expansion_ratio,
            device=device,
            bias=not no_bias,
            ffn_kwargs=self.ffn_config,
        )

        self.resid_attn_dropout = nn.Dropout(resid_pdrop)
        self.resid_ffn_dropout = nn.Dropout(resid_pdrop)
        self.use_pad_tok_in_ffn = use_pad_tok_in_ffn

    @property
    def args_to_exclude_in_attn_class(self):
        return {
            'attn_type',
            'alibi',
            'attn_uses_sequence_id',
            'alibi_bias_max',
            'rope',
            'rope_theta',
            'rope_impl',
            'rope_dail_config',
            'rope_hf_config',
        }

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

        n = self.apply_ffn(attention_mask, m)
        x = x + self.resid_ffn_dropout(n)
        return x, attn_weights, past_key_value

    def apply_ffn(
        self,
        attention_mask: Optional[torch.ByteTensor],
        m: torch.Tensor,
    ) -> torch.Tensor:
        """Apply feed forward layers to the input.

        Args:
            attention_mask (Optional[torch.ByteTensor]): The attention mask.
            m (torch.Tensor): The input.

        Returns:
            n (torch.Tensor): The output.
        """
        batch_size, seq_len = m.size()[:2]
        indices = None
        if not self.use_pad_tok_in_ffn and attention_mask is not None:
            assert unpad_input is not None
            attention_mask = self.slice_attention_mask(attention_mask, seq_len)
            m, indices, _, _ = unpad_input(m, attention_mask)
        n = self.ffn(m)
        if not self.use_pad_tok_in_ffn and attention_mask is not None:
            assert pad_input is not None
            n = pad_input(n, indices, batch_size, seq_len)
        return n

    def slice_attention_mask(
        self,
        attention_mask: torch.ByteTensor,
        seq_len: int,
    ) -> torch.ByteTensor:
        """Slice attention mask to the correct size.

        Can be overridden by subclasses to apply different slicing logic.

        Args:
            attention_mask (torch.ByteTensor): The attention mask.
            seq_len (int): The sequence length.

        Returns:
            torch.ByteTensor: The sliced attention mask.
        """
        return attention_mask


class FusedNormAttentionNorm(nn.Module):

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        args_to_exclude_in_attn_class: Set[str],
        attn_config: Optional[Dict] = None,
        ffn_has_norm: bool = False,
        fc_type: Optional[dict[str, Any]] = None,
        resid_pdrop: float = 0.0,
        norm_type: str = 'low_precision_layernorm',
        device: Optional[str] = None,
        no_bias: bool = False,
        **kwargs: Any,
    ):
        super().__init__()
        assert attn_config is not None
        assert isinstance(attn_config['attn_type'], str)

        # Usually, fc_type dict should be passed in through MPTBlock's __init__ function.
        if fc_type is None:
            fc_type = copy.deepcopy(fc_type_defaults)
            fc_type['bias'] = not no_bias
            fc_type['device'] = device

        # Necessary to avoid passing extraneous args into attn_class while allowing the use of **kwargs
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
                **attn_config_subset_for_attn_class,
            },
        )

        self.norm_2 = None
        if not ffn_has_norm:
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
