# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""Triton implementation of Flash Attention Layer.

Modified from:
https://raw.githubusercontent.com/HazyResearch/flash-attention/main/flash_attn/flash_attention.py
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange  # type: ignore (reportMissingImports)
from torch import Tensor


class FlashAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.

    Args:
        num_heads: number of attention heads
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
    """

    def __init__(self, num_heads, softmax_scale=None, device=None, dtype=None):
        # fail fast if triton is not available
        try:
            from flash_attn import flash_attn_triton  # type: ignore

            del flash_attn_triton
        except ImportError:
            raise ImportError(
                'examples was installed without flash attention + triton support. Please make sure you are in an environment with CUDA available and pip install .[llm]'
            )

        super().__init__()
        self.num_heads = num_heads
        self.softmax_scale = softmax_scale

    def forward(
        self,
        qkv,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""Multiheaded softmax attention.

        Arguments:
            qkv: The tensor containing the query, key, and value. (B, S, 3, H, D)
            key_padding_mask: not implemented for triton kernel.
            attn_mask: If specified, a 4D mask of floats which will be added to the attention weight. Must braodcast to (B, H, S, S).
            is_causal: If specified, applies a causal mask as attention mask. Default: ``False``.
        """
        from flash_attn import flash_attn_triton  # type: ignore

        assert qkv.dtype in [torch.float16, torch.bfloat16]
        assert qkv.is_cuda

        if key_padding_mask is not None and key_padding_mask.bool().logical_not(
        ).any():
            raise NotImplementedError(
                f'assumes key_padding_mask is taken care of by attn_mask')
        qkv = rearrange(qkv, 'b s (t h d) -> b s t h d', t=3, h=self.num_heads)

        attn_output = flash_attn_triton.flash_attn_qkvpacked_func(
            qkv, attn_mask, is_causal, self.softmax_scale)
        output = rearrange(attn_output, 'b s h d -> b s (h d)')
        return output, None


class FlashMHA(nn.Module):

    def __init__(self,
                 embed_dim,
                 num_heads,
                 bias=True,
                 batch_first=True,
                 causal=False,
                 softmax_scale=None,
                 device=None,
                 dtype=None,
                 **kwargs) -> None:
        assert batch_first
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal

        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, 'self.kdim must be divisible by num_heads'
        self.head_dim = self.embed_dim // num_heads
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, 'Only support head_dim <= 128 and divisible by 8'

        self.Wqkv = nn.Linear(embed_dim,
                              3 * embed_dim,
                              bias=bias,
                              **factory_kwargs)
        self.inner_attn = FlashAttention(num_heads=num_heads,
                                         softmax_scale=softmax_scale,
                                         **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim,
                                  embed_dim,
                                  bias=bias,
                                  **factory_kwargs)

    def forward(self,
                x,
                key_padding_mask: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None,
                need_weights: bool = False):
        r"""Multiheaded softmax attention.

        Args:
            x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
            key_padding_mask: not implemented for triton kernel.
            attn_mask: If specified, a 4D mask of floats which will be added to the attention weight. Must braodcast to (B, H, S, S).
            need_weights: not implemented for triton kernel.
        """
        if need_weights:
            raise NotImplementedError(f'Not implemented for triton kernel.')

        qkv = self.Wqkv(x)
        context, attn_weights = self.inner_attn(
            qkv,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            is_causal=self.causal)
        return self.out_proj(context), attn_weights
