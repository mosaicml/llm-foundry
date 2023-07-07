# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Attention layers."""

import math
import warnings
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
from packaging import version
from torch import nn

from llmfoundry.models.layers.norm import LPLayerNorm


def _reset_is_causal(num_query_tokens: int, num_key_tokens: int,
                     original_is_causal: bool):
    # disable causal when it is not needed
    # necessary for flash & triton for generation with kv_cache
    if original_is_causal and num_query_tokens != num_key_tokens:
        if num_query_tokens != 1:
            raise NotImplementedError(
                'MPT does not support query and key with different number of tokens, unless number of query tokens is 1.'
            )
        else:
            return False
    return original_is_causal


def scaled_multihead_dot_product_attention(
    query,
    key,
    value,
    n_heads,
    past_key_value=None,
    softmax_scale=None,
    attn_bias=None,
    key_padding_mask=None,
    is_causal=False,
    dropout_p=0.0,
    training=False,
    needs_weights=False,
    multiquery=False,
):
    q = rearrange(query, 'b s (h d) -> b h s d', h=n_heads)
    kv_n_heads = 1 if multiquery else n_heads
    k = rearrange(key, 'b s (h d) -> b h d s', h=kv_n_heads)
    v = rearrange(value, 'b s (h d) -> b h s d', h=kv_n_heads)

    if past_key_value is not None:
        # attn_impl: flash & triton use kernels which expect input shape [b, s, h, d_head].
        # kv_cache is therefore stored using that shape.
        # attn_impl: torch stores the kv_cache in the ordering which is most advantageous
        # for its attn computation ie
        # keys are stored as tensors with shape [b, h, d_head, s] and
        # values are stored as tensors with shape [b, h, s, d_head]
        if len(past_key_value) != 0:
            k = torch.cat([past_key_value[0], k], dim=3)
            v = torch.cat([past_key_value[1], v], dim=2)

        past_key_value = (k, v)

    b, _, s_q, d = q.shape
    s_k = k.size(-1)

    if softmax_scale is None:
        softmax_scale = 1 / math.sqrt(d)

    attn_weight = q.matmul(k) * softmax_scale

    if attn_bias is not None:
        # clamp to 0 necessary for torch 2.0 compile()
        _s_q = max(0, attn_bias.size(2) - s_q)
        _s_k = max(0, attn_bias.size(3) - s_k)
        attn_bias = attn_bias[:, :, _s_q:, _s_k:]

        if (attn_bias.size(-1) != 1 and
                attn_bias.size(-1) != s_k) or (attn_bias.size(-2) != 1 and
                                               attn_bias.size(-2) != s_q):
            raise RuntimeError(
                f'attn_bias (shape: {attn_bias.shape}) is expected to broadcast to shape: {attn_weight.shape}.'
            )
        attn_weight = attn_weight + attn_bias

    min_val = torch.finfo(q.dtype).min

    if key_padding_mask is not None:
        if attn_bias is not None:
            warnings.warn(
                'Propogating key_padding_mask to the attention module ' +\
                'and applying it within the attention module can cause ' +\
                'unneccessary computation/memory usage. Consider integrating ' +\
                'into attn_bias once and passing that to each attention ' +\
                'module instead.'
            )
        attn_weight = attn_weight.masked_fill(
            ~key_padding_mask.view((b, 1, 1, s_k)), min_val)

    if is_causal and (not q.size(2) == 1):
        s = max(s_q, s_k)
        causal_mask = attn_weight.new_ones(s, s, dtype=torch.float32)
        causal_mask = causal_mask.tril()
        causal_mask = causal_mask.to(torch.bool)
        causal_mask = ~causal_mask
        causal_mask = causal_mask[-s_q:, -s_k:]
        attn_weight = attn_weight.masked_fill(causal_mask.view(1, 1, s_q, s_k),
                                              min_val)

    attn_weight = torch.softmax(attn_weight, dim=-1)

    if dropout_p:
        attn_weight = torch.nn.functional.dropout(attn_weight,
                                                  p=dropout_p,
                                                  training=training,
                                                  inplace=True)

    out = attn_weight.to(v.dtype).matmul(v)
    out = rearrange(out, 'b h s d -> b s (h d)')

    if needs_weights:
        return out, attn_weight, past_key_value
    return out, None, past_key_value


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
    past_key_value=None,
    softmax_scale=None,
    attn_bias=None,
    key_padding_mask=None,
    is_causal=False,
    dropout_p=0.0,
    training=False,
    needs_weights=False,
    multiquery=False,
):
    try:
        from flash_attn import bert_padding, flash_attn_interface  # type: ignore # yapf: disable # isort: skip
    except:
        raise RuntimeError('Please install flash-attn==1.0.3.post0')

    check_valid_inputs(query, key, value)

    if past_key_value is not None:
        if len(past_key_value) != 0:
            key = torch.cat([past_key_value[0], key], dim=1)
            value = torch.cat([past_key_value[1], value], dim=1)

        past_key_value = (key, value)

    if attn_bias is not None:
        # clamp to 0 necessary for torch 2.0 compile()
        _s_q = max(0, attn_bias.size(2) - query.size(1))
        _s_k = max(0, attn_bias.size(3) - key.size(1))
        attn_bias = attn_bias[:, :, _s_q:, _s_k:]

    if attn_bias is not None:
        raise NotImplementedError(f'attn_bias not implemented for flash attn.')

    batch_size, seqlen = query.shape[:2]

    if key_padding_mask is None:
        key_padding_mask = torch.ones_like(key[:, :, 0], dtype=torch.bool)
    query_padding_mask = key_padding_mask[:, -query.size(1):]

    query_unpad, indices_q, cu_seqlens_q, max_seqlen_q = bert_padding.unpad_input(
        query, query_padding_mask)
    query_unpad = rearrange(query_unpad, 'nnz (h d) -> nnz h d', h=n_heads)

    key_unpad, _, cu_seqlens_k, max_seqlen_k = bert_padding.unpad_input(
        key, key_padding_mask)
    key_unpad = rearrange(key_unpad,
                          'nnz (h d) -> nnz h d',
                          h=1 if multiquery else n_heads)

    value_unpad, _, _, _ = bert_padding.unpad_input(value, key_padding_mask)
    value_unpad = rearrange(value_unpad,
                            'nnz (h d) -> nnz h d',
                            h=1 if multiquery else n_heads)

    if multiquery:
        # Expanding a tensor does not allocate new memory, but only creates a new
        # view on the existing tensor where a dimension of size one is expanded
        # to a larger size by setting the stride to 0.
        # - pytorch docs
        #
        # hopefully the kernels can utilize this and we're jot just wasting BW here
        key_unpad = key_unpad.expand(key_unpad.size(0), n_heads,
                                     key_unpad.size(-1))
        value_unpad = value_unpad.expand(value_unpad.size(0), n_heads,
                                         value_unpad.size(-1))

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
    return output, None, past_key_value


def triton_flash_attn_fn(
    query,
    key,
    value,
    n_heads,
    past_key_value=None,
    softmax_scale=None,
    attn_bias=None,
    key_padding_mask=None,
    is_causal=False,
    dropout_p=0.0,
    training=False,
    needs_weights=False,
    multiquery=False,
):
    try:
        from llmfoundry.models.layers.flash_attn_triton import flash_attn_func
    except:
        _installed = False
        if version.parse(torch.__version__) < version.parse('2.0.0'):
            _installed = True
            # if torch1.13.1 revert to using triton flash attn from HazyResearch
            # with flash-attn==1.0.3.post0 and triton==2.0.0.dev20221202
            try:
                from flash_attn.flash_attn_triton import flash_attn_func
            except:
                _installed = False
        if not _installed:
            # installing triton-pre-mlir works for both torch1.13.1 and torch2.0+
            # default recommendation is to install this variant
            raise RuntimeError(
                'Requirements for `attn_impl: triton` not installed. Either (1) have a CUDA-compatible GPU '
                'and `pip install .[gpu]` if installing from llm-foundry source or '
                '`pip install triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir#subdirectory=python` '
                'if installing from pypi, or (2) use torch attn model.attn_config.attn_impl=torch (torch attn_impl will be slow). '
                'Note: (1) requires you have CMake and PyTorch already installed.'
            )

    check_valid_inputs(query, key, value)

    if past_key_value is not None:
        if len(past_key_value) != 0:
            key = torch.cat([past_key_value[0], key], dim=1)
            value = torch.cat([past_key_value[1], value], dim=1)

        past_key_value = (key, value)

    if attn_bias is not None:
        # clamp to 0 necessary for torch 2.0 compile()
        _s_q = max(0, attn_bias.size(2) - query.size(1))
        _s_k = max(0, attn_bias.size(3) - key.size(1))
        attn_bias = attn_bias[:, :, _s_q:, _s_k:]

    if dropout_p:
        raise NotImplementedError(
            f'Dropout not implemented for attn_impl: triton.')

    if needs_weights:
        raise NotImplementedError(
            f'attn_impl: triton cannot return attn weights.')

    if key_padding_mask is not None:
        warnings.warn(
            'Propagating key_padding_mask to the attention module ' +\
            'and applying it within the attention module can cause ' +\
            'unnecessary computation/memory usage. Consider integrating ' +\
            'into attn_bias once and passing that to each attention ' +\
            'module instead.'
        )
        b_size, s_k = key_padding_mask.shape[:2]

        if attn_bias is None:
            attn_bias = query.new_zeros(b_size, 1, 1, s_k)

        attn_bias = attn_bias.masked_fill(
            ~key_padding_mask.view((b_size, 1, 1, s_k)),
            torch.finfo(query.dtype).min)

    query = rearrange(query, 'b s (h d) -> b s h d', h=n_heads)
    key = rearrange(key, 'b s (h d) -> b s h d', h=1 if multiquery else n_heads)
    value = rearrange(value,
                      'b s (h d) -> b s h d',
                      h=1 if multiquery else n_heads)

    if multiquery:
        # necessary to repeat instead of expand tensor because
        # output contains NaN in edge cases such as with head dimension = 8
        key = key.repeat(1, 1, n_heads, 1)
        value = value.repeat(1, 1, n_heads, 1)

    reset_is_causal = _reset_is_causal(query.size(1), key.size(1), is_causal)
    attn_output = flash_attn_func(query, key, value, attn_bias, reset_is_causal,
                                  softmax_scale)

    output = attn_output.view(*attn_output.shape[:2], -1)

    return output, None, past_key_value


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
        clip_qkv: Optional[float] = None,
        qk_ln: bool = False,
        softmax_scale: Optional[float] = None,
        attn_pdrop: float = 0.0,
        low_precision_layernorm: bool = False,
        verbose: int = 0,
        device: Optional[str] = None,
    ):
        super().__init__()

        self.attn_impl = attn_impl
        self.clip_qkv = clip_qkv
        self.qk_ln = qk_ln

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

        if self.qk_ln:
            layernorm_class = LPLayerNorm if low_precision_layernorm else nn.LayerNorm
            self.q_ln = layernorm_class(self.d_model, device=device)
            self.k_ln = layernorm_class(self.d_model, device=device)

        if self.attn_impl == 'flash':
            self.attn_fn = flash_attn_fn
        elif self.attn_impl == 'triton':
            self.attn_fn = triton_flash_attn_fn
            if verbose:
                warnings.warn(
                    'While `attn_impl: triton` can be faster than `attn_impl: flash` ' +\
                    'it uses more memory. When training larger models this can trigger '  +\
                    'alloc retries which hurts performance. If encountered, we recommend ' +\
                    'using `attn_impl: flash` if your model does not use `alibi` or `prefix_lm`.'
                )
        elif self.attn_impl == 'torch':
            self.attn_fn = scaled_multihead_dot_product_attention
            if torch.cuda.is_available() and verbose:
                warnings.warn(
                    'Using `attn_impl: torch`. If your model does not use `alibi` or ' +\
                    '`prefix_lm` we recommend using `attn_impl: flash` otherwise ' +\
                    'we recommend using `attn_impl: triton`.'
                )
        else:
            raise ValueError(f'{attn_impl=} is an invalid setting.')

        self.out_proj = nn.Linear(self.d_model, self.d_model, device=device)
        self.out_proj._is_residual = True  # type: ignore

    def forward(
        self,
        x,
        past_key_value=None,
        attn_bias=None,
        attention_mask=None,
        is_causal=True,
        needs_weights=False,
    ):
        qkv = self.Wqkv(x)

        if self.clip_qkv:
            qkv.clamp_(min=-self.clip_qkv, max=self.clip_qkv)

        query, key, value = qkv.chunk(3, dim=2)

        key_padding_mask = attention_mask

        if self.qk_ln:
            # Applying layernorm to qk
            dtype = query.dtype
            query = self.q_ln(query).to(dtype)
            key = self.k_ln(key).to(dtype)

        context, attn_weights, past_key_value = self.attn_fn(
            query,
            key,
            value,
            self.n_heads,
            past_key_value=past_key_value,
            softmax_scale=self.softmax_scale,
            attn_bias=attn_bias,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            dropout_p=self.attn_dropout_p,
            training=self.training,
            needs_weights=needs_weights,
        )

        return self.out_proj(context), attn_weights, past_key_value


class MultiQueryAttention(nn.Module):
    """Multi-Query self attention.

    Using torch or triton attention implemetation enables user to also use
    additive bias.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        attn_impl: str = 'triton',
        clip_qkv: Optional[float] = None,
        qk_ln: bool = False,
        softmax_scale: Optional[float] = None,
        attn_pdrop: float = 0.0,
        low_precision_layernorm: bool = False,
        verbose: int = 0,
        device: Optional[str] = None,
    ):
        super().__init__()

        self.attn_impl = attn_impl
        self.clip_qkv = clip_qkv
        self.qk_ln = qk_ln

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.softmax_scale = softmax_scale
        if self.softmax_scale is None:
            self.softmax_scale = 1 / math.sqrt(self.head_dim)
        self.attn_dropout_p = attn_pdrop

        # NOTE: if we ever want to make attn TensorParallel, I'm pretty sure we'll
        # want to split Wqkv into Wq and Wkv where Wq can be TensorParallel but
        # Wkv shouldn't be TensorParallel
        # - vchiley
        self.Wqkv = nn.Linear(
            d_model,
            d_model + 2 * self.head_dim,
            device=device,
        )
        # for param init fn; enables shape based init of fused layers
        fuse_splits = (d_model, d_model + self.head_dim)
        self.Wqkv._fused = (0, fuse_splits)  # type: ignore

        if self.qk_ln:
            layernorm_class = LPLayerNorm if low_precision_layernorm else nn.LayerNorm
            self.q_ln = layernorm_class(d_model, device=device)
            self.k_ln = layernorm_class(self.head_dim, device=device)

        if self.attn_impl == 'flash':
            self.attn_fn = flash_attn_fn
        elif self.attn_impl == 'triton':
            self.attn_fn = triton_flash_attn_fn
            if verbose:
                warnings.warn(
                    'While `attn_impl: triton` can be faster than `attn_impl: flash` ' +\
                    'it uses more memory. When training larger models this can trigger '  +\
                    'alloc retries which hurts performance. If encountered, we recommend ' +\
                    'using `attn_impl: flash` if your model does not use `alibi` or `prefix_lm`.'
                )
        elif self.attn_impl == 'torch':
            self.attn_fn = scaled_multihead_dot_product_attention
            if torch.cuda.is_available() and verbose:
                warnings.warn(
                    'Using `attn_impl: torch`. If your model does not use `alibi` or ' +\
                    '`prefix_lm` we recommend using `attn_impl: flash` otherwise ' +\
                    'we recommend using `attn_impl: triton`.'
                )
        else:
            raise ValueError(f'{attn_impl=} is an invalid setting.')

        self.out_proj = nn.Linear(self.d_model, self.d_model, device=device)
        self.out_proj._is_residual = True  # type: ignore

    def forward(
        self,
        x,
        past_key_value=None,
        attn_bias=None,
        attention_mask=None,
        is_causal=True,
        needs_weights=False,
    ):
        qkv = self.Wqkv(x)

        if self.clip_qkv:
            qkv.clamp_(min=-self.clip_qkv, max=self.clip_qkv)

        query, key, value = qkv.split(
            [self.d_model, self.head_dim, self.head_dim], dim=2)

        key_padding_mask = attention_mask

        if self.qk_ln:
            # Applying layernorm to qk
            dtype = query.dtype
            query = self.q_ln(query).to(dtype)
            key = self.k_ln(key).to(dtype)

        context, attn_weights, past_key_value = self.attn_fn(
            query,
            key,
            value,
            self.n_heads,
            past_key_value=past_key_value,
            softmax_scale=self.softmax_scale,
            attn_bias=attn_bias,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            dropout_p=self.attn_dropout_p,
            training=self.training,
            needs_weights=needs_weights,
            multiquery=True,
        )

        return self.out_proj(context), attn_weights, past_key_value


def attn_bias_shape(attn_impl, n_heads, seq_len, alibi, prefix_lm, causal,
                    use_sequence_id):
    if attn_impl == 'flash':
        return None
    elif attn_impl in ['torch', 'triton']:
        if alibi:
            if (prefix_lm or not causal) or use_sequence_id:
                return (1, n_heads, seq_len, seq_len)
            return (1, n_heads, 1, seq_len)
        elif prefix_lm or use_sequence_id:
            return (1, 1, seq_len, seq_len)
        return None
    else:
        raise ValueError(f'{attn_impl=} is an invalid setting.')


def build_attn_bias(
    attn_impl,
    attn_bias,
    n_heads,
    seq_len,
    causal=False,
    alibi=False,
    alibi_bias_max=8,
):
    if attn_impl == 'flash':
        return None
    elif attn_impl in ['torch', 'triton']:
        if alibi:
            # in place add alibi to attn bias
            device, dtype = attn_bias.device, attn_bias.dtype
            attn_bias = attn_bias.add(
                build_alibi_bias(
                    n_heads,
                    seq_len,
                    full=not causal,
                    alibi_bias_max=alibi_bias_max,
                    device=device,
                    dtype=dtype,
                ))
        return attn_bias
    else:
        raise ValueError(f'{attn_impl=} is an invalid setting.')


def gen_slopes(n_heads, alibi_bias_max=8, device=None):
    _n_heads = 2**math.ceil(math.log2(n_heads))
    m = torch.arange(1, _n_heads + 1, dtype=torch.float32, device=device)
    m = m.mul(alibi_bias_max / _n_heads)
    slopes = (1. / torch.pow(2, m))

    if _n_heads != n_heads:
        # if n_heads is not a power of two,
        # Huggingface and FasterTransformer calculate slopes normally,
        # then return this strided concatenation of slopes
        slopes = torch.concat([slopes[1::2], slopes[::2]])[:n_heads]

    return slopes.view(1, n_heads, 1, 1)


def build_alibi_bias(
    n_heads,
    seq_len,
    full=False,
    alibi_bias_max=8,
    device=None,
    dtype=None,
):
    alibi_bias = torch.arange(1 - seq_len, 1, dtype=torch.int32,
                              device=device).view(1, 1, 1, seq_len)
    if full:
        # generate 1 x Heads x SeqLen x SeqLen alibi bias mask
        # otherwise the mask is 1 x Heads x 1 x SeqLen (which is broadcast to the appropriate size)
        alibi_bias = alibi_bias - torch.arange(
            1 - seq_len, 1, dtype=torch.int32, device=device).view(
                1, 1, seq_len, 1)
        alibi_bias = alibi_bias.abs().mul(-1)

    slopes = gen_slopes(n_heads, alibi_bias_max, device=device)
    alibi_bias = alibi_bias * slopes
    return alibi_bias.to(dtype=dtype)


ATTN_CLASS_REGISTRY = {
    'multihead_attention': MultiheadAttention,
    'multiquery_attention': MultiQueryAttention,
}
