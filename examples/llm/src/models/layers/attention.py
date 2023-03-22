# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""Attention layers."""

import math
import warnings
from typing import Optional

import torch
import torch.nn as nn
from composer.algorithms.low_precision_layernorm.low_precision_layernorm import \
    LPLayerNorm
from einops import rearrange
from torch import nn


def _reset_is_causal(num_query_tokens: int, num_key_tokens: int,
                     original_is_causal: bool):
    if original_is_causal and num_query_tokens != num_key_tokens:
        if num_query_tokens != 1:
            raise NotImplementedError(
                'MosaicGPT does not support query and key with different number of tokens, unless number of query tokens is 1.'
            )
        else:
            return False
    return original_is_causal


def scaled_multihead_dot_product_attention(
    query,
    key,
    value,
    n_heads,
    softmax_scale=None,
    attn_bias=None,
    query_padding_mask=None,
    key_padding_mask=None,
    is_causal=False,
    dropout_p=0.0,
    training=False,
    needs_weights=False,
):
    if query_padding_mask is not None:
        query = query.masked_fill(~query_padding_mask.unsqueeze(-1), 0)
    if key_padding_mask is not None:
        key = key.masked_fill(~key_padding_mask.unsqueeze(-1), 0)
        value = value.masked_fill(~key_padding_mask.unsqueeze(-1), 0)

    q = rearrange(query, 'b s (h d) -> b h s d', h=n_heads)
    k = rearrange(key, 'b s (h d) -> b h d s', h=n_heads)  # includes key.t()
    v = rearrange(value, 'b s (h d) -> b h s d', h=n_heads)

    b, _, s_q, d = q.shape
    s_k = k.size(-1)

    if softmax_scale is None:
        softmax_scale = 1 / math.sqrt(d)

    attn_weight = q.matmul(k) * softmax_scale

    if attn_bias is not None:
        if (attn_bias.size(-1) != 1 and
                attn_bias.size(-1) != s_k) or (attn_bias.size(-2) != 1 and
                                               attn_bias.size(-2) != s_q):
            raise RuntimeError(
                f'attn_bias (shape: {attn_bias.shape}) is expected to broadcast to shape: {attn_weight.shape}.'
            )
        attn_weight = attn_weight + attn_bias

    if query_padding_mask is not None:
        attn_weight = attn_weight.masked_fill(
            ~query_padding_mask.view(b, 1, s_q, 1), -float('inf'))
    if key_padding_mask is not None:
        attn_weight = attn_weight.masked_fill(
            ~key_padding_mask.view((b, 1, 1, s_k)), -float('inf'))

    if is_causal:
        s = max(s_q, s_k)
        causal_mask = attn_weight.new_ones(s, s, dtype=torch.bool)
        causal_mask = causal_mask.tril()
        causal_mask = causal_mask.logical_not()
        causal_mask = causal_mask[-s_q:, -s_k:]
        attn_weight = attn_weight.masked_fill(causal_mask.view(1, 1, s_q, s_k),
                                              -float('inf'))

    attn_weight = torch.softmax(attn_weight, dim=-1)

    if query_padding_mask is not None:
        attn_weight = attn_weight.masked_fill(
            ~query_padding_mask.view(b, 1, s_q, 1), 0)
    if key_padding_mask is not None:
        attn_weight = attn_weight.masked_fill(
            ~key_padding_mask.view(b, 1, 1, s_k), 0)

    if dropout_p:
        attn_weight = torch.nn.functional.dropout(attn_weight,
                                                  p=dropout_p,
                                                  training=training,
                                                  inplace=True)

    out = attn_weight.matmul(v)
    out = rearrange(out, 'b h s d -> b s (h d)')

    if needs_weights:
        return out, attn_weight
    return out, None


def check_valid_inputs(*tensors, valid_dtypes=[torch.float16, torch.bfloat16]):
    for tensor in tensors:
        if tensor.dtype not in valid_dtypes:
            raise TypeError(f'{tensor.dtype=} must be in {valid_dtypes=}.')
        if not tensor.is_cuda:
            raise TypeError(f'Inputs must be cuda tensors ({tensor.is_cuda=}).')


def flash_attn_fn(
    query,
    key,
    value,
    n_heads,
    softmax_scale=None,
    attn_bias=None,
    query_padding_mask=None,
    key_padding_mask=None,
    is_causal=False,
    dropout_p=0.0,
    training=False,
    needs_weights=False,
):
    try:
        from flash_attn import bert_padding  # type: ignore
        from flash_attn import flash_attn_interface  # type: ignore
    except ImportError as e:
        raise e

    check_valid_inputs(query, key, value)

    if attn_bias is not None:
        raise NotImplementedError(f'attn_bias not implemented for flash attn.')

    batch_size, seqlen = query.shape[:2]

    if query_padding_mask is None:
        query_padding_mask = torch.ones_like(query[:, :, 0], dtype=torch.bool)
    if key_padding_mask is None:
        key_padding_mask = torch.ones_like(key[:, :, 0], dtype=torch.bool)

    query_unpad, indices_q, cu_seqlens_q, max_seqlen_q = bert_padding.unpad_input(
        query, query_padding_mask)
    query_unpad = rearrange(query_unpad, 'nnz (h d) -> nnz h d', h=n_heads)

    key_unpad, _, cu_seqlens_k, max_seqlen_k = bert_padding.unpad_input(
        key, key_padding_mask)
    key_unpad = rearrange(key_unpad, 'nnz (h d) -> nnz h d', h=n_heads)

    value_unpad, _, _, _ = bert_padding.unpad_input(value, key_padding_mask)
    value_unpad = rearrange(value_unpad, 'nnz (h d) -> nnz h d', h=n_heads)

    dropout_p = dropout_p if training else 0.0

    reset_is_causal = _reset_is_causal(query.size(1), key.size(1), is_causal)

    output_unpad = flash_attn_interface.flash_attn_unpadded_func(
        query_unpad,
        key_unpad,
        value_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale=softmax_scale,
        causal=reset_is_causal,
        return_attn_probs=needs_weights)

    output = bert_padding.pad_input(
        rearrange(output_unpad, 'nnz h d -> nnz (h d)'), indices_q, batch_size,
        seqlen)
    return output, None


def triton_flash_attn_fn(
    query,
    key,
    value,
    n_heads,
    softmax_scale=None,
    attn_bias=None,
    query_padding_mask=None,
    key_padding_mask=None,
    is_causal=False,
    dropout_p=0.0,
    training=False,
    needs_weights=False,
):
    try:
        from flash_attn import flash_attn_triton  # type: ignore
    except ImportError as e:
        raise e

    check_valid_inputs(query, key, value)

    if dropout_p:
        raise NotImplementedError(
            f'Dropout not implemented for attn_impl: triton.')

    if needs_weights:
        raise NotImplementedError(
            f'attn_impl: triton cannot return attn weights.')

    if query_padding_mask is not None:
        query = query.masked_fill(~query_padding_mask.unsqueeze(-1), 0)
    if key_padding_mask is not None:
        key = key.masked_fill(~key_padding_mask.unsqueeze(-1), 0)
        value = value.masked_fill(~key_padding_mask.unsqueeze(-1), 0)

    if query_padding_mask is not None or key_padding_mask is not None:
        b_size, s_q, s_k = query.size(0), 1, 1
        if query_padding_mask is not None:
            s_q = query_padding_mask.size(1)
        if key_padding_mask is not None:
            s_k = key_padding_mask.size(1)

        if attn_bias is not None:
            attn_bias = attn_bias.expand(b_size, -1, -1, -1)
        else:
            attn_bias = query.new_zeros(b_size, 1, s_q, s_k)

        if query_padding_mask is not None:
            attn_bias = attn_bias.masked_fill(
                ~query_padding_mask.view((b_size, 1, s_q, 1)), -float('inf'))

        if key_padding_mask is not None:
            attn_bias = attn_bias.masked_fill(
                ~key_padding_mask.view((b_size, 1, 1, s_k)), -float('inf'))

    query = rearrange(query, 'b s (h d) -> b s h d', h=n_heads)
    key = rearrange(key, 'b s (h d) -> b s h d', h=n_heads)
    value = rearrange(value, 'b s (h d) -> b s h d', h=n_heads)

    reset_is_causal = _reset_is_causal(query.size(1), key.size(1), is_causal)
    attn_output = flash_attn_triton.flash_attn_func(query, key, value,
                                                    attn_bias, reset_is_causal,
                                                    softmax_scale)

    output = attn_output.view(*attn_output.shape[:2], -1)
    if query_padding_mask is not None:
        output = output.masked_fill(~query_padding_mask.unsqueeze(-1), 0)

    return output, None


class MultiheadAttention(nn.Module):
    """Multi-head self attention.

    Using torch or triton attention implemetation enables user to also use
    additive bias.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        attn_impl: str = 'triton',
        attn_clip_qkv: Optional[float] = None,
        attn_qk_ln: bool = False,
        softmax_scale: Optional[float] = None,
        attn_pdrop: float = 0.0,
        low_precision_layernorm: bool = False,
        device: Optional[str] = None,
    ):
        super().__init__()

        self.attn_impl = attn_impl
        self.clip_qkv = attn_clip_qkv
        self.attn_qk_ln = attn_qk_ln

        self.d_model = d_model
        self.n_heads = n_heads
        self.softmax_scale = softmax_scale
        if self.softmax_scale is None:
            self.softmax_scale = 1 / math.sqrt(self.d_model / self.n_heads)
        self.attn_dropout_p = attn_pdrop

        self.Wqkv = nn.Linear(self.d_model, 3 * self.d_model, device=device)
        # for param init fn; enables shape based init of fused layers
        fuse_splits = (d_model, 2 * d_model)
        self.Wqkv._fused = (0, fuse_splits)  # type: ignore

        if self.attn_qk_ln:
            layernorm_class = LPLayerNorm if low_precision_layernorm else nn.LayerNorm
            self.q_ln = layernorm_class(self.d_model, device=device)
            self.k_ln = layernorm_class(self.d_model, device=device)

        if self.attn_impl == 'flash':
            self.attn_fn = flash_attn_fn
        elif self.attn_impl == 'triton':
            self.attn_fn = triton_flash_attn_fn
            warnings.warn(
                'While `attn_impl: triton` can be faster than `attn_impl: flash` '
                'it uses more memory. When training larger models this can trigger '
                'alloc retries which hurts performance. If encountered, we recommend '
                'using `attn_impl: flash`.')
        elif self.attn_impl == 'torch':
            self.attn_fn = scaled_multihead_dot_product_attention
            warnings.warn(
                'Using `attn_impl: torch`; recommened using `attn_impl: flash`.'
            )
        else:
            raise ValueError(f'{attn_impl=} is an invalid setting.')

        self.out_proj = nn.Linear(self.d_model, self.d_model, device=device)
        self.out_proj._is_residual = True  # type: ignore

    def forward(self,
                x,
                past_key_value=None,
                attn_bias=None,
                key_padding_mask=None,
                is_causal=True,
                needs_weights=False):
        qkv = self.Wqkv(x)

        if self.clip_qkv:
            qkv.clamp_(min=-self.clip_qkv, max=self.clip_qkv)

        query, key, value = qkv.chunk(3, dim=2)

        query_padding_mask = None
        if key_padding_mask is not None:
            query_padding_mask = key_padding_mask[:, -query.size(1):]

        if self.attn_qk_ln:
            # Applying layernorm to qk
            dtype = query.dtype
            query = self.q_ln(query).to(dtype)
            key = self.k_ln(key).to(dtype)

        if past_key_value is not None:
            if len(past_key_value) != 0:
                key = torch.cat([past_key_value[0], key], dim=1)
                value = torch.cat([past_key_value[1], value], dim=1)

            past_key_value = (key, value)

        if attn_bias is not None:
            attn_bias = attn_bias[:, :, -query.size(1):, -key.size(1):]

        context, attn_weights = self.attn_fn(
            query,
            key,
            value,
            self.n_heads,
            softmax_scale=self.softmax_scale,
            attn_bias=attn_bias,
            query_padding_mask=query_padding_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            dropout_p=self.attn_dropout_p,
            training=self.training,
            needs_weights=needs_weights,
        )

        return self.out_proj(context), attn_weights, past_key_value


def attn_bias_shape(attn_impl, n_heads, seq_len, alibi, causal):
    if attn_impl == 'flash':
        return None
    elif attn_impl == 'triton':
        if alibi:
            if not causal:
                return (1, n_heads, seq_len, seq_len)
            return (1, n_heads, 1, seq_len)
        return None
    elif attn_impl == 'torch':
        if alibi:
            if not causal:
                return (1, n_heads, seq_len, seq_len)
            return (1, n_heads, 1, seq_len)
        return None
    else:
        raise ValueError(f'{attn_impl=} is an invalid setting.')


def attn_bias(attn_impl,
              attn_bias,
              n_heads,
              seq_len,
              causal=False,
              alibi=False,
              alibi_bias_max=8):
    if attn_impl == 'flash':
        return None
    elif attn_impl == 'triton':
        if alibi:
            # in place add alibi to attn bias
            device, dtype = attn_bias.device, attn_bias.dtype
            attn_bias = attn_bias.add(
                alibi_bias(n_heads,
                           seq_len,
                           full=not causal,
                           alibi_bias_max=alibi_bias_max,
                           device=device,
                           dtype=dtype))
        return attn_bias
    elif attn_impl == 'torch':
        if attn_bias is not None:
            if alibi:
                # in place add alibi to attn bias
                device, dtype = attn_bias.device, attn_bias.dtype
                attn_bias = attn_bias.add(
                    alibi_bias(n_heads,
                               seq_len,
                               full=not causal,
                               alibi_bias_max=alibi_bias_max,
                               device=device,
                               dtype=dtype))
            return attn_bias
    else:
        raise ValueError(f'{attn_impl=} is an invalid setting.')


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
        # otherwise the mask is 1 x Heads x 1 x SeqLen (which is broadcast to the appropriate size)
        alibi_bias = alibi_bias - torch.arange(
            1 - seq_len, 1, dtype=dtype, device=device).view(1, 1, seq_len, 1)
        alibi_bias = alibi_bias.abs().mul(-1)

    m = torch.arange(1, n_heads + 1, dtype=dtype, device=device)
    m = m.mul(alibi_bias_max / n_heads)
    alibi_bias = alibi_bias * (1. / (2**m.view(1, n_heads, 1, 1)))
    return alibi_bias
