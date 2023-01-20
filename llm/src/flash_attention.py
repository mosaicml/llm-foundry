# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""Triton implementation of Flash Attention Layer.

Modified from:
https://raw.githubusercontent.com/HazyResearch/flash-attention/main/flash_attn/flash_attention.py
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

try:
    from src.flash_attn_triton import flash_attn_qkvpacked_func
except ImportError as e:
    raise e


class FlashAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.

    Args:
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
    """

    def __init__(self, num_heads, softmax_scale=None, device=None, dtype=None):
        super().__init__()
        self.num_heads = num_heads
        self.softmax_scale = softmax_scale

    def forward(
            self,
            qkv,
            key_padding_mask: Optional[Tensor] = None,
            attn_mask: Optional[Tensor] = None,
            is_causal: bool = False,
            need_weights: bool = False,
            average_attn_weights: bool = True
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""Multiheaded softmax attention.

        Arguments:
            qkv: The tensor containing the query, key, and value. (B, S, 3, H, D) if key_padding_mask is None
                if unpadded: (nnz, 3, h, d)
            key_padding_mask: If specified, a mask of shape :math:`(N, S)` indicating which elements within ``key``
                to ignore for the purpose of attention (i.e. treat as "padding"). For unbatched `query`, shape should be :math:`(S)`.
                Binary and byte masks are supported.
                For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
                the purpose of attention. For a float mask, it will be directly added to the corresponding ``key`` value.
            attn_mask: If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape
                :math:`(L, S)` or :math:`(N\cdot\text{num\_heads}, L, S)`, where :math:`N` is the batch size,
                :math:`L` is the target sequence length, and :math:`S` is the source sequence length. A 2D mask will be
                broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch.
                Binary, byte, and float masks are supported. For a binary mask, a ``True`` value indicates that the
                corresponding position is not allowed to attend. For a byte mask, a non-zero value indicates that the
                corresponding position is not allowed to attend. For a float mask, the mask values will be added to
                the attention weight.
            is_causal: If specified, applies a causal mask as attention mask. Mutually exclusive with providing attn_mask.
                Default: ``False``.
            need_weights: If specified, returns ``attn_output_weights`` in addition to ``attn_outputs``.
                Default: ``True``.
            average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across
                heads. Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an
                effect when ``need_weights=True``. Default: ``True`` (i.e. average weights across heads)
        """
        assert not need_weights and not average_attn_weights
        assert qkv.dtype in [torch.float16, torch.bfloat16]
        assert qkv.is_cuda

        if key_padding_mask is not None and key_padding_mask.bool().logical_not(
        ).any():
            raise NotImplementedError(
                f'assumes key_padding_mask is taken care of by attn_mask')
        qkv = rearrange(qkv, 'b s (t h d) -> b s t h d', t=3, h=self.num_heads)

        attn_output = flash_attn_qkvpacked_func(qkv, attn_mask, is_causal,
                                                self.softmax_scale)
        output = rearrange(attn_output, 'b s h d -> b s (h d)')
        return output, None


class FlashMHA(nn.Module):

    def __init__(self,
                 embed_dim,
                 num_heads,
                 bias=True,
                 batch_first=True,
                 causal=False,
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
                                         softmax_scale=None,
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
            key_padding_mask: bool tensor of shape (batch, seqlen)
            attn_mask: If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape
                :math:`(L, S)` or :math:`(N\cdot\text{num\_heads}, L, S)`, where :math:`N` is the batch size,
                :math:`L` is the target sequence length, and :math:`S` is the source sequence length. A 2D mask will be
                broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch.
                Binary, byte, and float masks are supported. For a binary mask, a ``True`` value indicates that the
                corresponding position is not allowed to attend. For a byte mask, a non-zero value indicates that the
                corresponding position is not allowed to attend. For a float mask, the mask values will be added to
                the attention weight.
            need_weights: If specified, returns ``attn_output_weights`` in addition to ``attn_outputs``.
                Default: ``True``.
        """
        qkv = self.Wqkv(x)
        context, attn_weights = self.inner_attn(
            qkv,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            is_causal=self.causal,
            need_weights=need_weights,
            average_attn_weights=False)
        return self.out_proj(context), attn_weights
