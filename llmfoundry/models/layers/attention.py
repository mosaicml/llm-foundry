# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Attention layers."""

import math
import warnings
from typing import Any, Optional

import torch
import torch.nn as nn
import transformers
from einops import rearrange
from packaging import version
from torch import nn

from llmfoundry.models.layers.fc import FC_CLASS_REGISTRY
from llmfoundry.models.layers.norm import NORM_CLASS_REGISTRY


def is_flash_v2_installed(v2_version: str = '2.0.0'):
    assert version.parse(v2_version) >= version.parse('2.0.0')
    try:
        import flash_attn as flash_attn
    except:
        return False
    return version.parse(flash_attn.__version__) >= version.parse(v2_version)


def is_flash_v1_installed():
    try:
        import flash_attn as flash_attn
    except:
        return False
    return version.parse(flash_attn.__version__) < version.parse('2.0.0')


def is_transformers_version_gte(hf_version: str) -> bool:
    return version.parse(transformers.__version__) >= version.parse(hf_version)


def check_alibi_support(attention_impl: str) -> bool:
    return attention_impl != 'flash' or is_flash_v2_installed(
        v2_version='v2.4.2')


# Before importing any transformers models, we need to disable transformers flash attention if
# we are in an environment with flash attention version <2. Transformers hard errors on a not properly
# gated import otherwise.
if is_flash_v1_installed():
    import transformers
    transformers.utils.is_flash_attn_available = lambda: False

from transformers.models.llama.modeling_llama import apply_rotary_pos_emb


def _reset_is_causal(num_query_tokens: int, num_key_tokens: int,
                     original_is_causal: bool) -> bool:
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


def repeat_kv_for_gqa(hidden: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Perform repeat of kv heads along a particular dimension.

    hidden.shape expected to be: (batch size, seq len, kv_n_heads, head_dim)
    n_rep: amount of repetitions of kv_n_heads
    Unlike torch.repeat_interleave, this function avoids allocating new memory.
    """
    if n_rep == 1:
        return hidden

    b, s, kv_n_heads, d = hidden.shape

    hidden = hidden[:, :, :, None, :].expand(b, s, kv_n_heads, n_rep, d)

    return hidden.reshape(b, s, kv_n_heads * n_rep, d)


def scaled_multihead_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    n_heads: int,
    kv_n_heads: Optional[int] = None,
    past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    softmax_scale: Optional[float] = None,
    attn_bias: Optional[torch.Tensor] = None,
    key_padding_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    dropout_p: float = 0.0,
    training: bool = False,
    needs_weights: bool = False,
    multiquery: bool = False,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor,
                                                                torch.Tensor]]]:
    if multiquery:
        warnings.warn(
            DeprecationWarning(
                'The direct use of the multiquery arg is deprecated. Setting kv_n_heads=1 automatically. Please set kv_n_heads=1 explicitly to remove this warning.'
            ))
        kv_n_heads = 1
    elif kv_n_heads is None:
        warnings.warn(
            DeprecationWarning(
                'Not specifying a value for the kv_n_heads arg is deprecated. Setting kv_n_heads=n_heads automatically. Please set kv_n_heads=n_heads explicitly to remove this warning.'
            ))
        kv_n_heads = n_heads

    q = rearrange(query, 'b s (h d) -> b h s d', h=n_heads)
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

    # grouped query case
    if kv_n_heads > 1 and kv_n_heads < n_heads:
        # necessary to do a transpose to swap (b h s d) -> (b s h d) for repeat_kv_for_gqa function
        k = repeat_kv_for_gqa(k.transpose(1, 2),
                              n_heads // kv_n_heads).transpose(1, 2)
        v = repeat_kv_for_gqa(v.transpose(1, 2),
                              n_heads // kv_n_heads).transpose(1, 2)

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
                'Propagating key_padding_mask to the attention module ' +\
                'and applying it within the attention module can cause ' +\
                'unnecessary computation/memory usage. Consider integrating ' +\
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


def check_valid_inputs(*tensors: torch.Tensor,
                       valid_dtypes: Optional[list[torch.dtype]] = None):
    if valid_dtypes is None:
        valid_dtypes = [torch.float16, torch.bfloat16]
    for tensor in tensors:
        if tensor.dtype not in valid_dtypes:
            raise TypeError(f'{tensor.dtype=} must be in {valid_dtypes=}.')
        if not tensor.is_cuda:
            raise TypeError(f'Inputs must be cuda tensors ({tensor.is_cuda=}).')


def flash_attn_fn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    n_heads: int,
    kv_n_heads: Optional[int] = None,
    past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    softmax_scale: Optional[float] = None,
    attn_bias: Optional[torch.Tensor] = None,
    key_padding_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    dropout_p: float = 0.0,
    training: bool = False,
    needs_weights: bool = False,
    multiquery: bool = False,
    should_repeat_kv_for_gqa: Optional[bool] = True,
    sliding_window_size: int = -1,
    alibi_slopes: Optional[torch.Tensor] = None,
    flash_attn_padding_info: Optional[dict[str, torch.Tensor]] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor,
                                                                torch.Tensor]]]:
    if key_padding_mask is not None:
        raise ValueError('key_padding_mask should be None for flash attn.')
    del key_padding_mask
    if flash_attn_padding_info is None:
        raise ValueError('flash_attn_padding_info is required for flash attn.')
    try:
        from flash_attn import bert_padding, flash_attn_interface  # type: ignore # yapf: disable # isort: skip
    except:
        raise RuntimeError(
            'Please install flash-attn==1.0.9 or flash-attn==2.3.6')

    check_valid_inputs(query, key, value)

    if multiquery:
        warnings.warn(
            DeprecationWarning(
                'The direct use of the multiquery arg is deprecated. Setting kv_n_heads=1 automatically. Please set kv_n_heads=1 explicitly to remove this warning.'
            ))
        kv_n_heads = 1
    elif kv_n_heads is None:
        warnings.warn(
            DeprecationWarning(
                'Not specifying a value for the kv_n_heads arg is deprecated. Setting kv_n_heads=n_heads automatically. Please set kv_n_heads=n_heads explicitly to remove this warning.'
            ))
        kv_n_heads = n_heads

    if past_key_value is not None:
        if len(past_key_value) != 0:
            key = torch.cat([past_key_value[0], key], dim=1)
            value = torch.cat([past_key_value[1], value], dim=1)

        past_key_value = (key, value)

    if attn_bias is not None:
        raise NotImplementedError(f'attn_bias not implemented for flash attn.')

    batch_size, seqlen = query.shape[:2]

    indices_q = flash_attn_padding_info['indices_q']
    indices_k = flash_attn_padding_info['indices_k']
    indices_v = flash_attn_padding_info['indices_v']
    cu_seqlens_q = flash_attn_padding_info['cu_seqlens_q']
    cu_seqlens_k = flash_attn_padding_info['cu_seqlens_k']
    max_seqlen_q = flash_attn_padding_info['max_seqlen_q']
    max_seqlen_k = flash_attn_padding_info['max_seqlen_k']

    query_unpad = bert_padding.index_first_axis(
        rearrange(query, 'b s ... -> (b s) ...'), indices_q)
    query_unpad = rearrange(query_unpad, 'nnz (h d) -> nnz h d', h=n_heads)

    key_unpad = bert_padding.index_first_axis(
        rearrange(key, 'b s ... -> (b s) ...'), indices_k)
    key_unpad = rearrange(key_unpad, 'nnz (h d) -> nnz h d', h=kv_n_heads)

    value_unpad = bert_padding.index_first_axis(
        rearrange(value, 'b s ... -> (b s) ...'), indices_v)
    value_unpad = rearrange(value_unpad, 'nnz (h d) -> nnz h d', h=kv_n_heads)

    if (kv_n_heads < n_heads) and (not is_flash_v2_installed()) and (
            not should_repeat_kv_for_gqa):
        raise ValueError(
            'For Grouped Query Attention or Multi Query Attention, should_repeat_kv_for_gqa should be set to True if not using Flash Attention v2.'
        )

    if should_repeat_kv_for_gqa:
        # multi-query case
        if kv_n_heads == 1:
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
        # grouped query case
        elif kv_n_heads < n_heads:
            # Each query belong to a group of kv heads of group size n_heads // kv_n_heads
            # We repeat each kv head by the group size number to use the underlying MHA kernels

            # since repeat_kv_for_gqa expects input dims of (b, s, kv_n_heads, d)
            # we use .view to modify {key, value}_unpad appropriately

            key_unpad = repeat_kv_for_gqa(
                key_unpad.view(1, key_unpad.size(0), kv_n_heads, -1),
                n_heads // kv_n_heads).view(key_unpad.size(0), n_heads, -1)
            value_unpad = repeat_kv_for_gqa(
                value_unpad.view(1, value_unpad.size(0), kv_n_heads, -1),
                n_heads // kv_n_heads).view(value_unpad.size(0), n_heads, -1)

    dropout_p = dropout_p if training else 0.0

    reset_is_causal = _reset_is_causal(query.size(1), key.size(1), is_causal)

    if is_flash_v1_installed():
        output_unpad = flash_attn_interface.flash_attn_unpadded_func(
            q=query_unpad,
            k=key_unpad,
            v=value_unpad,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=reset_is_causal,
            return_attn_probs=needs_weights)
    elif is_flash_v2_installed():
        alibi_kwargs = {}
        if check_alibi_support('flash'):
            alibi_kwargs = {'alibi_slopes': alibi_slopes}
        elif alibi_slopes is not None:
            raise ValueError(
                'alibi_slopes is only supported for flash-attn>=2.4.2')
        output_unpad = flash_attn_interface.flash_attn_varlen_func(
            q=query_unpad,
            k=key_unpad,
            v=value_unpad,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=reset_is_causal,
            return_attn_probs=needs_weights,
            window_size=(sliding_window_size, sliding_window_size),
            **alibi_kwargs)
    else:
        raise RuntimeError(
            'flash-attn==1.0.9 or flash-attn==2.4.2 is required.')

    output = bert_padding.pad_input(
        rearrange(output_unpad, 'nnz h d -> nnz (h d)'), indices_q, batch_size,
        seqlen)
    return output, None, past_key_value


def triton_flash_attn_fn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    n_heads: int,
    kv_n_heads: Optional[int] = None,
    past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    softmax_scale: Optional[float] = None,
    attn_bias: Optional[torch.Tensor] = None,
    key_padding_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    dropout_p: float = 0.0,
    training: bool = False,
    needs_weights: bool = False,
    multiquery: bool = False,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor,
                                                                torch.Tensor]]]:
    try:
        from llmfoundry.models.layers.flash_attn_triton import flash_attn_func
    except:
        _installed = False
        if version.parse(torch.__version__) < version.parse('2.0.0'):
            _installed = True
            # if torch1.13.1 revert to using triton flash attn from HazyResearch
            # with flash-attn==1.0.9 and triton==2.0.0.dev20221202
            try:
                from flash_attn.flash_attn_triton import flash_attn_func
            except:
                _installed = False
        if not _installed:
            # installing triton-pre-mlir works for both torch1.13.1 and torch2.0+
            # default recommendation is to install this variant
            raise RuntimeError(
                'Requirements for `attn_impl: triton` not installed. Either (1) have a CUDA-compatible GPU '
                +
                'and `pip install .[gpu]` if installing from llm-foundry source or '
                +
                '`pip install triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir#subdirectory=python` '
                +
                'if installing from pypi, or (2) use torch attn model.attn_config.attn_impl=torch (torch attn_impl will be slow). '
                +
                'Note: (1) requires you have CMake and PyTorch already installed.'
            )

    check_valid_inputs(query, key, value)

    if multiquery:
        warnings.warn(
            DeprecationWarning(
                'The direct use of the multiquery arg is deprecated. Setting kv_n_heads=1 automatically. Please set kv_n_heads=1 explicitly to remove this warning.'
            ))
        kv_n_heads = 1
    elif kv_n_heads is None:
        warnings.warn(
            DeprecationWarning(
                'Not specifying a value for the kv_n_heads arg is deprecated. Setting kv_n_heads=n_heads automatically. Please set kv_n_heads=n_heads explicitly to remove this warning.'
            ))
        kv_n_heads = n_heads

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
    dropout_p = dropout_p if training else 0.0

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
    key = rearrange(key, 'b s (h d) -> b s h d', h=kv_n_heads)
    value = rearrange(value, 'b s (h d) -> b s h d', h=kv_n_heads)

    # multi-query case
    if kv_n_heads == 1:
        # necessary to repeat instead of expand tensor because
        # output contains NaN in edge cases such as with head dimension = 8
        key = key.repeat(1, 1, n_heads, 1)
        value = value.repeat(1, 1, n_heads, 1)
    # grouped query case
    elif kv_n_heads < n_heads:
        # Each query belong to a group of kv heads of group size n_heads // kv_n_heads
        # We repeat each kv head by the group size number to use the underlying MHA kernels
        key = repeat_kv_for_gqa(key, n_heads // kv_n_heads)
        value = repeat_kv_for_gqa(value, n_heads // kv_n_heads)

    reset_is_causal = _reset_is_causal(query.size(1), key.size(1), is_causal)
    attn_output = flash_attn_func(  # type: ignore
        query, key, value, attn_bias, reset_is_causal, softmax_scale)

    output = attn_output.view(*attn_output.shape[:2], -1)  # type: ignore

    return output, None, past_key_value


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention (GQA) is a generalization of Multi-head (MHA).

    and Multi-query attention (MQA).

    This allows the user to set a variable of number of kv_n_heads, rather than
    just n_heads or 1, as in MHA and MQA. Using torch or triton attention
    implementation enables user to also use additive bias.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        kv_n_heads: int,
        attn_impl: str = 'triton',
        clip_qkv: Optional[float] = None,
        qk_ln: bool = False,
        softmax_scale: Optional[float] = None,
        attn_pdrop: float = 0.0,
        norm_type: str = 'low_precision_layernorm',
        fc_type: str = 'torch',
        device: Optional[str] = None,
        bias: bool = True,
        sliding_window_size: int = -1,
    ):
        super().__init__()

        self.attn_impl = attn_impl
        self.clip_qkv = clip_qkv
        self.qk_ln = qk_ln

        self.d_model = d_model
        self.n_heads = n_heads
        self.kv_n_heads = kv_n_heads
        self.sliding_window_size = sliding_window_size

        self.head_dim = d_model // n_heads

        if self.kv_n_heads <= 0:
            raise ValueError('kv_n_heads should be greater than zero.')

        if self.kv_n_heads > self.n_heads:
            raise ValueError(
                'The number of KV heads should be less than or equal to Q heads.'
            )

        if self.n_heads % self.kv_n_heads != 0:
            raise ValueError(
                'Each Q head should get the same number of KV heads, so n_heads must be divisible by kv_n_heads.'
            )

        self.softmax_scale = softmax_scale
        if self.softmax_scale is None:
            self.softmax_scale = 1 / math.sqrt(self.d_model / self.n_heads)
        self.attn_dropout_p = attn_pdrop

        fc_kwargs: dict[str, Any] = {
            'bias': bias,
        }
        if fc_type != 'te':
            fc_kwargs['device'] = device
        self.Wqkv = FC_CLASS_REGISTRY[fc_type](
            self.d_model,
            self.d_model + 2 * self.kv_n_heads * self.head_dim,
            **fc_kwargs,
        )
        # for param init fn; enables shape based init of fused layers
        fuse_splits = [
            i * self.head_dim
            for i in range(1, self.n_heads + 2 * self.kv_n_heads)
        ]
        self.Wqkv._fused = (0, fuse_splits)

        if self.qk_ln:
            norm_class = NORM_CLASS_REGISTRY[norm_type.lower()]
            self.q_ln = norm_class(self.d_model, device=device)
            self.k_ln = norm_class(self.kv_n_heads * self.head_dim,
                                   device=device)

        if self.attn_impl == 'flash':
            self.attn_fn = flash_attn_fn
        elif self.attn_impl == 'triton':
            self.attn_fn = triton_flash_attn_fn
        elif self.attn_impl == 'torch':
            self.attn_fn = scaled_multihead_dot_product_attention
        else:
            raise ValueError(f'{attn_impl=} is an invalid setting.')

        self.out_proj = FC_CLASS_REGISTRY[fc_type](
            self.d_model,
            self.d_model,
            **fc_kwargs,
        )
        self.out_proj._is_residual = True

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attn_bias: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb_w_meta_info: Optional[dict] = None,
        is_causal: bool = True,
        needs_weights: bool = False,
        alibi_slopes: Optional[torch.Tensor] = None,
        flash_attn_padding_info: Optional[dict[str, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[
            torch.Tensor, torch.Tensor]]]:
        qkv = self.Wqkv(x)

        if self.clip_qkv:
            qkv = qkv.clamp(min=-self.clip_qkv, max=self.clip_qkv)

        query, key, value = qkv.split(
            [
                self.d_model,
                self.kv_n_heads * self.head_dim,
                self.kv_n_heads * self.head_dim,
            ],
            dim=2,
        )

        key_padding_mask = attention_mask

        if self.qk_ln:
            # Applying layernorm to qk
            dtype = query.dtype
            query = self.q_ln(query).to(dtype)
            key = self.k_ln(key).to(dtype)

        if rotary_emb_w_meta_info is not None:
            rotary_emb = rotary_emb_w_meta_info['rotary_emb']
            seq_len = rotary_emb_w_meta_info['seq_len']
            offset_info = rotary_emb_w_meta_info['offset_info']
            bsz, seqlen = query.shape[:2]
            query = query.view(bsz, seqlen, -1, self.head_dim)
            key = key.view(bsz, seqlen, -1, self.head_dim)

            if rotary_emb_w_meta_info['impl'] == 'dail':
                value = value.view(bsz, seqlen, -1, self.head_dim)

                kv = torch.stack([key, value], dim=2)
                query, kv = rotary_emb(query,
                                       kv,
                                       seqlen_offset=offset_info,
                                       max_seqlen=seq_len)
                [key, value] = torch.unbind(kv, dim=2)

                value = value.view(bsz, seqlen, self.kv_n_heads * self.head_dim)
            elif rotary_emb_w_meta_info['impl'] == 'hf':
                (cos, sin) = rotary_emb(value, seq_len)
                if is_transformers_version_gte('4.36'):
                    query, key = apply_rotary_pos_emb(query,
                                                      key,
                                                      cos,
                                                      sin,
                                                      offset_info,
                                                      unsqueeze_dim=2)
                else:
                    query = query.transpose(1, 2)
                    key = key.transpose(1, 2)
                    query, key = apply_rotary_pos_emb(query, key, cos, sin,
                                                      offset_info)
                    query = query.transpose(1, 2)
                    key = key.transpose(1, 2)

            query = query.view(bsz, seqlen, self.d_model)
            key = key.view(bsz, seqlen, self.kv_n_heads * self.head_dim)

        extra_attn_kwargs = {}
        if self.attn_impl == 'flash':
            if flash_attn_padding_info is not None:
                key_padding_mask = None
            extra_attn_kwargs = {
                'should_repeat_kv_for_gqa': not is_flash_v2_installed(),
                'sliding_window_size': self.sliding_window_size,
                'alibi_slopes': alibi_slopes,
                'flash_attn_padding_info': flash_attn_padding_info,
            }

        context, attn_weights, past_key_value = self.attn_fn(
            query,
            key,
            value,
            self.n_heads,
            self.kv_n_heads,
            past_key_value=past_key_value,
            softmax_scale=self.softmax_scale,
            attn_bias=attn_bias,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            dropout_p=self.attn_dropout_p,
            training=self.training,
            needs_weights=needs_weights,
            **extra_attn_kwargs,
        )

        return self.out_proj(context), attn_weights, past_key_value


class MultiheadAttention(GroupedQueryAttention):
    """Multi-head self attention.

    Using torch or triton attention implementation enables user to also use
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
        norm_type: str = 'low_precision_layernorm',
        fc_type: str = 'torch',
        device: Optional[str] = None,
        bias: bool = True,
        sliding_window_size: int = -1,
    ):
        super().__init__(
            d_model=d_model,
            n_heads=n_heads,
            kv_n_heads=n_heads,  # for MHA, same # heads as kv groups
            attn_impl=attn_impl,
            clip_qkv=clip_qkv,
            qk_ln=qk_ln,
            softmax_scale=softmax_scale,
            attn_pdrop=attn_pdrop,
            norm_type=norm_type,
            fc_type=fc_type,
            device=device,
            bias=bias,
            sliding_window_size=sliding_window_size,
        )


class MultiQueryAttention(GroupedQueryAttention):
    """Multi-Query self attention.

    Using torch or triton attention implementation enables user to also use
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
        norm_type: str = 'low_precision_layernorm',
        fc_type: str = 'torch',
        device: Optional[str] = None,
        bias: bool = True,
        sliding_window_size: int = -1,
    ):
        super().__init__(
            d_model=d_model,
            n_heads=n_heads,
            kv_n_heads=1,  # for MQA, 1 head
            attn_impl=attn_impl,
            clip_qkv=clip_qkv,
            qk_ln=qk_ln,
            softmax_scale=softmax_scale,
            attn_pdrop=attn_pdrop,
            norm_type=norm_type,
            fc_type=fc_type,
            device=device,
            bias=bias,
            sliding_window_size=sliding_window_size,
        )


def attn_bias_shape(
        attn_impl: str, n_heads: int, seq_len: int, alibi: bool,
        prefix_lm: bool, causal: bool,
        use_sequence_id: bool) -> Optional[tuple[int, int, int, int]]:
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
    attn_impl: str,
    attn_bias: torch.Tensor,
    n_heads: int,
    seq_len: int,
    causal: bool = False,
    alibi: bool = False,
    alibi_bias_max: int = 8,
) -> Optional[torch.Tensor]:
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


def gen_slopes(n_heads: int,
               alibi_bias_max: int = 8,
               device: Optional[torch.device] = None,
               return_1d: bool = False) -> torch.Tensor:
    _n_heads = 2**math.ceil(math.log2(n_heads))
    m = torch.arange(1, _n_heads + 1, dtype=torch.float32, device=device)
    m = m.mul(alibi_bias_max / _n_heads)
    slopes = (1. / torch.pow(2, m))

    if _n_heads != n_heads:
        # if n_heads is not a power of two,
        # Huggingface and FasterTransformer calculate slopes normally,
        # then return this strided concatenation of slopes
        slopes = torch.concat([slopes[1::2], slopes[::2]])[:n_heads]
    if return_1d:
        return slopes
    return slopes.view(1, n_heads, 1, 1)


def build_alibi_bias(
    n_heads: int,
    seq_len: int,
    full: bool = False,
    alibi_bias_max: int = 8,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
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
    'grouped_query_attention': GroupedQueryAttention
}
