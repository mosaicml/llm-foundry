# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""Attention layers for the GPT models."""

import warnings
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import DictConfig


class TorchCausalAttention(nn.Module):

    def __init__(self, cfg: DictConfig, device: Optional[str] = None):
        super().__init__()
        self.mhsa = nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=cfg.n_heads,
            dropout=cfg.attn_pdrop,
            bias=True,
            batch_first=True,
            device=device,
        )
        self.mhsa.out_proj._is_residual = True  # type: ignore

        warnings.warn(
            DeprecationWarning(
                'Using `attn_impl: torch` is deprecated; recommened using `attn_impl: flash`.'
            ))

    def forward(self, x, key_padding_mask, attn_mask=None):
        return self.mhsa(x,
                         x,
                         x,
                         attn_mask=attn_mask,
                         key_padding_mask=~key_padding_mask,
                         need_weights=True)

    @staticmethod
    def mask_shape(n_heads, seq_len, alibi):
        if alibi:
            return (n_heads, seq_len, seq_len)
        return (seq_len, seq_len)

    @staticmethod
    def attn_mask_(attn_mask, n_heads, seq_len, alibi=False, alibi_bias_max=8):
        # in-place fill causal attn mask
        #
        # Two important disclaimers
        # 1. Torch uses additive attention. If your attn_mask/key_padding mask is a float tensor, it will add the floats
        #   directly to your attention matrix. If they are boolean masks, True will be converted to -inf before adding the
        #   mask to your attentions. See https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html#torch.nn.MultiheadAttention.forward
        #   Basically True/-inf indicates tokens we do not want to attend to.
        #
        # 2. This is is the exact opposite behavior of Huggingface's tokenizers, which use the convention that True denotes tokens
        #   we do want to attend to. See https://huggingface.co/docs/transformers/glossary#attention-mask
        attn_mask.fill_(float('-inf'))
        attn_mask.triu_(diagonal=1)

        if alibi:
            device, dtype = attn_mask.device, attn_mask.dtype
            a_bias = alibi_bias(n_heads,
                                seq_len,
                                full=True,
                                alibi_bias_max=alibi_bias_max,
                                device=device,
                                dtype=dtype)
            attn_mask.add_(a_bias.squeeze())

        return attn_mask


class FlashCausalAttention(nn.Module):

    def __init__(self, cfg: DictConfig, device: Optional[str] = None):
        super().__init__()
        try:
            from flash_attn.flash_attention import (  # type: ignore
                FlashAttention, FlashMHA)
        except ImportError as e:
            raise e

        self.attn_qk_ln = cfg.get('attn_qk_ln')
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads

        if self.attn_qk_ln:
            self.W_qkv = nn.Linear(self.d_model,
                                   3 * self.d_model,
                                   bias=True,
                                   device=device)
            self.causal_attn = FlashAttention(attention_dropout=cfg.attn_pdrop,
                                              device=device)
            self.out_proj = nn.Linear(self.d_model,
                                      self.d_model,
                                      bias=True,
                                      device=device)

            self.out_proj._is_residual = True  # type: ignore

            self.q_ln = nn.LayerNorm(self.d_model, device=device)
            self.k_ln = nn.LayerNorm(self.d_model, device=device)
        else:
            self.mhsa = FlashMHA(
                embed_dim=cfg.d_model,
                num_heads=cfg.n_heads,
                attention_dropout=cfg.attn_pdrop,
                bias=True,
                batch_first=True,
                causal=True,
                device=device,
            )
            self.mhsa.out_proj._is_residual = True

    def forward(self, x, key_padding_mask, attn_mask=None):
        assert attn_mask is None
        if self.attn_qk_ln:
            qkv = self.W_qkv(x)

            # Applying layernorm to qk
            dtype = qkv.dtype
            q, k, v = qkv.split(self.d_model, dim=-1)
            q = self.q_ln(q).to(dtype)
            k = self.k_ln(k).to(dtype)
            qkv = torch.cat([q, k, v], dim=-1)

            # attention
            qkv = rearrange(qkv,
                            'b s (three h d) -> b s three h d',
                            three=3,
                            h=self.n_heads)

            context, attn_weights = self.causal_attn(
                qkv,
                key_padding_mask=key_padding_mask,
                causal=True,
                need_weights=False)
            return self.out_proj(rearrange(
                context, 'b s h d -> b s (h d)')), attn_weights

        else:
            return self.mhsa(x,
                             key_padding_mask=key_padding_mask,
                             need_weights=False)

    @staticmethod
    def mask_shape(*args, **kwargs):
        return None

    @staticmethod
    def attn_mask_(*args, **kwargs):
        return None


class TritonFlashCausalAttention(nn.Module):
    """Multi-headed self attention using triton FlashAttn kernel.

    This also includes bias for Alibi integration.
    """

    def __init__(self, cfg: DictConfig, device: Optional[str] = None):
        super().__init__()
        try:
            from examples.llm.src.layers.flash_attention import \
                FlashMHA  # type: ignore
        except ImportError as e:
            raise e

        assert cfg.attn_pdrop == 0, 'triton kernel does not support attn_dropout'

        self.mhsa = FlashMHA(
            embed_dim=cfg.d_model,
            num_heads=cfg.n_heads,
            bias=True,
            batch_first=True,
            causal=True,
            device=device,
        )
        self.mhsa.out_proj._is_residual = True  # type: ignore

        warnings.warn(
            'While `attn_impl: triton` can be faster than `attn_impl: flash` '
            'it uses more memory. When training larger models this can trigger '
            'alloc retries which hurts performance. If encountered, we recommend '
            'using `attn_impl: flash`.')

    def forward(self, x, key_padding_mask=None, attn_mask=None):
        assert key_padding_mask is None
        return self.mhsa(x,
                         key_padding_mask=None,
                         attn_mask=attn_mask,
                         need_weights=False)

    @staticmethod
    def mask_shape(n_heads, seq_len, alibi):
        return (1, n_heads, 1, seq_len) if alibi else None

    @staticmethod
    def attn_mask_(attn_mask, n_heads, seq_len, alibi=False, alibi_bias_max=8):
        if attn_mask is not None:
            # in-place fill causal attn mask
            attn_mask.zero_()

            if alibi:
                device, dtype = attn_mask.device, attn_mask.dtype
                attn_mask.add_(
                    alibi_bias(n_heads,
                               seq_len,
                               full=False,
                               alibi_bias_max=alibi_bias_max,
                               device=device,
                               dtype=dtype))

        return attn_mask


def alibi_bias(n_heads,
               seq_len,
               full=False,
               alibi_bias_max=8,
               device=None,
               dtype=None):
    alibi_bias = torch.arange(1 - seq_len, 1, dtype=dtype,
                              device=device).view(1, 1, 1, seq_len)
    if full:
        # generate 1 x Heads x SeqLen x SeqLen alibi bias mask
        # otherwise the mask is 1 x Heads x 1 x SeqLen (which is braodcasted up to the approproate size)
        alibi_bias = alibi_bias - torch.arange(
            1 - seq_len, 1, dtype=dtype, device=device).view(1, 1, seq_len, 1)
        alibi_bias.abs_().mul_(
            -1
        )  # since we're using causal flag, this isn't really needed, but why not include it

    m = torch.arange(1, n_heads + 1, dtype=dtype, device=device)
    m.mul_(alibi_bias_max / n_heads)
    alibi_bias = alibi_bias * (1. / (2**m.view(1, n_heads, 1, 1)))
    return alibi_bias
