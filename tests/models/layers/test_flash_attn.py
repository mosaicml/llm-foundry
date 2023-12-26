# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

from llmfoundry.models.layers import attention
from llmfoundry.models.layers.attention import (flash_attn_fn,
                                                is_flash_v2_installed,
                                                triton_flash_attn_fn)
from llmfoundry.models.mpt.modeling_mpt import gen_alibi_slopes


@pytest.mark.gpu
@pytest.mark.skipif(
    not is_flash_v2_installed(),
    reason='GQA natively only supported by Flash Attention after v2.')
@pytest.mark.parametrize('kv_n_heads', [1, 4, 8])
def test_gqa_kv_repetition(kv_n_heads: int):
    # Test that flash attention v2 with GQA (kv_n_heads < n_heads) works the same
    # whether we repeat the kv_n_heads explicitly or flash attention v2 handles it on its own.
    d = 128
    n_heads = 8
    seqlen_1 = 6
    bsz = 2

    query_1 = torch.randn(bsz, seqlen_1, n_heads * d).to(torch.bfloat16).cuda()
    query_1.requires_grad = True
    key_1 = torch.randn(bsz, seqlen_1, kv_n_heads * d).to(torch.bfloat16).cuda()
    key_1.requires_grad = True
    value_1 = torch.randn(bsz, seqlen_1,
                          kv_n_heads * d).to(torch.bfloat16).cuda()
    value_1.requires_grad = True

    output_1, _, _ = flash_attn_fn(query=query_1,
                                   key=key_1,
                                   value=value_1,
                                   n_heads=n_heads,
                                   kv_n_heads=kv_n_heads,
                                   past_key_value=None,
                                   softmax_scale=1 / math.sqrt(d),
                                   attn_bias=None,
                                   key_padding_mask=None,
                                   is_causal=True,
                                   dropout_p=0.0,
                                   training=False,
                                   needs_weights=False,
                                   multiquery=False,
                                   attention_mask_in_length=None,
                                   should_repeat_kv_for_gqa=True)

    output_1.sum().backward()

    query_2 = query_1.detach().clone()
    query_2.requires_grad = True
    key_2 = key_1.detach().clone()
    key_2.requires_grad = True
    value_2 = value_1.detach().clone()
    value_2.requires_grad = True

    output_2, _, _ = flash_attn_fn(query=query_2,
                                   key=key_2,
                                   value=value_2,
                                   n_heads=n_heads,
                                   kv_n_heads=kv_n_heads,
                                   past_key_value=None,
                                   softmax_scale=1 / math.sqrt(d),
                                   attn_bias=None,
                                   key_padding_mask=None,
                                   is_causal=True,
                                   dropout_p=0.0,
                                   training=False,
                                   needs_weights=False,
                                   multiquery=False,
                                   attention_mask_in_length=None,
                                   should_repeat_kv_for_gqa=False)

    output_2.sum().backward()
    assert torch.allclose(output_1, output_2)
    assert torch.allclose(query_1.grad, query_2.grad)  # type: ignore
    assert torch.allclose(key_1.grad, key_2.grad)  # type: ignore
    assert torch.allclose(value_1.grad, value_2.grad)  # type: ignore


@pytest.mark.gpu
@pytest.mark.skipif(
    not is_flash_v2_installed(v2_version='v2.1.2'),
    reason=
    'Using sequence id with flash attention requires flash attention v2.1.2 or higher.'
)
def test_seq_id_masking_FA_v2():
    # Test that flash attention v2 with sequence id masking works correctly.
    d = 128
    n_heads = 4
    kv_n_heads = 4
    seqlen_1 = 6
    bsz = 2

    query_1 = torch.randn(bsz, seqlen_1, n_heads * d).to(torch.bfloat16).cuda()
    query_1.requires_grad = True
    key_1 = torch.randn(bsz, seqlen_1, kv_n_heads * d).to(torch.bfloat16).cuda()
    key_1.requires_grad = True
    value_1 = torch.randn(bsz, seqlen_1,
                          kv_n_heads * d).to(torch.bfloat16).cuda()
    value_1.requires_grad = True

    seq_ranges = [
        (0, 3), (3, 5), (5, 6)
    ]  # Each batch has 3 sequences of length 3, 2, and 1 respectively.
    attention_mask_in_length_1 = torch.tensor([[3, 2, 1, 0, 0, 0],
                                               [3, 2, 1, 0, 0,
                                                0]]).to(torch.int64).cuda()

    output_1, _, _ = flash_attn_fn(
        query=query_1,
        key=key_1,
        value=value_1,
        n_heads=n_heads,
        kv_n_heads=kv_n_heads,
        past_key_value=None,
        softmax_scale=1 / math.sqrt(d),
        attn_bias=None,
        key_padding_mask=None,
        is_causal=True,
        dropout_p=0.0,
        training=False,
        needs_weights=False,
        multiquery=False,
        attention_mask_in_length=attention_mask_in_length_1)

    output_1.sum().backward()

    for seq_range in seq_ranges:
        query_2 = query_1.detach().clone()[:, seq_range[0]:seq_range[1], :]
        query_2.requires_grad = True
        key_2 = key_1.detach().clone()[:, seq_range[0]:seq_range[1], :]
        key_2.requires_grad = True
        value_2 = value_1.detach().clone()[:, seq_range[0]:seq_range[1], :]
        value_2.requires_grad = True

        output_2, _, _ = flash_attn_fn(query=query_2,
                                       key=key_2,
                                       value=value_2,
                                       n_heads=n_heads,
                                       kv_n_heads=kv_n_heads,
                                       past_key_value=None,
                                       softmax_scale=1 / math.sqrt(d),
                                       attn_bias=None,
                                       key_padding_mask=None,
                                       is_causal=True,
                                       dropout_p=0.0,
                                       training=False,
                                       needs_weights=False,
                                       multiquery=False,
                                       attention_mask_in_length=None)

        output_2.sum().backward()
        assert torch.allclose(output_1[:, seq_range[0]:seq_range[1], :],
                              output_2)
        assert torch.allclose(
            query_1.grad[:, seq_range[0]:seq_range[1], :],  # type: ignore
            query_2.grad)  # type: ignore
        assert torch.allclose(
            key_1.grad[:, seq_range[0]:seq_range[1], :],  # type: ignore
            key_2.grad)  # type: ignore
        assert torch.allclose(
            value_1.grad[:, seq_range[0]:seq_range[1], :],  # type: ignore
            value_2.grad)  # type: ignore


@pytest.mark.gpu
@pytest.mark.skipif(
    not is_flash_v2_installed(v2_version='v2.3.0'),
    reason=
    'Sliding window attention only supported by Flash Attention after v2.3.0.')
@pytest.mark.parametrize('sliding_window_size', [1, 4, 8])
def test_sliding_window(sliding_window_size: int):
    # Test that sliding window attention works as expected.
    dtype = torch.bfloat16
    device = 'cuda'
    d = 128
    n_heads = 8
    seqlen_1 = 8
    bsz = 2

    query_1 = torch.randn(bsz, seqlen_1, n_heads * d).to(dtype=dtype,
                                                         device=device)
    query_1.requires_grad = True
    key_1 = torch.randn(bsz, seqlen_1, n_heads * d).to(dtype=dtype,
                                                       device=device)
    key_1.requires_grad = True
    value_1 = torch.randn(bsz, seqlen_1, n_heads * d).to(dtype=dtype,
                                                         device=device)
    value_1.requires_grad = True

    output_1, _, _ = flash_attn_fn(query=query_1,
                                   key=key_1,
                                   value=value_1,
                                   n_heads=n_heads,
                                   kv_n_heads=n_heads,
                                   past_key_value=None,
                                   softmax_scale=1 / math.sqrt(d),
                                   attn_bias=None,
                                   key_padding_mask=None,
                                   is_causal=True,
                                   dropout_p=0.0,
                                   training=False,
                                   needs_weights=False,
                                   multiquery=False,
                                   attention_mask_in_length=None,
                                   should_repeat_kv_for_gqa=True,
                                   sliding_window_size=sliding_window_size)

    output_1.sum().backward()

    query_2 = query_1.detach().clone()
    query_2.requires_grad = True
    key_2 = key_1.detach().clone()
    key_2.requires_grad = True
    value_2 = value_1.detach().clone()
    value_2.requires_grad = True

    attn_bias_2 = torch.zeros(1, 1, seqlen_1, seqlen_1).to(dtype=dtype,
                                                           device=device)

    window_mask_2 = torch.tril(
        torch.ones(seqlen_1, seqlen_1), diagonal=-(sliding_window_size + 1)).to(
            dtype=dtype, device=device) * torch.finfo(attn_bias_2.dtype).min
    attn_bias_2 = attn_bias_2 + window_mask_2
    output_2, _, _ = triton_flash_attn_fn(
        query=query_2,
        key=key_2,
        value=value_2,
        n_heads=n_heads,
        kv_n_heads=n_heads,
        past_key_value=None,
        softmax_scale=1 / math.sqrt(d),
        attn_bias=attn_bias_2,
        key_padding_mask=None,
        is_causal=True,
        dropout_p=0.0,
        training=False,
        needs_weights=False,
        multiquery=False,
    )

    output_2.sum().backward()

    assert torch.allclose(output_1, output_2)
    assert torch.norm(query_2.grad - query_1.grad  # type: ignore
                     ) <= 1e-2 + 1e-2 * torch.norm(query_2.grad)
    assert torch.norm(key_2.grad - key_1.grad  # type: ignore
                     ) <= 1e-2 + 1e-2 * torch.norm(key_2.grad)
    assert torch.norm(value_2.grad - value_1.grad  # type: ignore
                     ) <= 1e-2 + 1e-2 * torch.norm(value_2.grad)


@pytest.mark.gpu
@pytest.mark.skipif(
    not is_flash_v2_installed(v2_version='v2.4.2'),
    reason='ALiBi only supported by Flash Attention after v2.4.2.')
@pytest.mark.parametrize('n_heads', [1, 6, 8])
def test_alibi_bias(n_heads: int):
    # Test that sliding window attention works as expected.
    dtype = torch.bfloat16
    device = 'cuda'
    d = 128
    seqlen_1 = 8
    bsz = 2

    query_1 = torch.randn(bsz, seqlen_1, n_heads * d).to(dtype=dtype,
                                                         device=device)
    query_1.requires_grad = True
    key_1 = torch.randn(bsz, seqlen_1, n_heads * d).to(dtype=dtype,
                                                       device=device)
    key_1.requires_grad = True
    value_1 = torch.randn(bsz, seqlen_1, n_heads * d).to(dtype=dtype,
                                                         device=device)
    value_1.requires_grad = True
    attn_bias_1 = gen_alibi_slopes(n_heads=n_heads,
                                   alibi_bias_max=8,
                                   device=torch.device(device))
    output_1, _, _ = flash_attn_fn(query=query_1,
                                   key=key_1,
                                   value=value_1,
                                   n_heads=n_heads,
                                   kv_n_heads=n_heads,
                                   past_key_value=None,
                                   softmax_scale=1 / math.sqrt(d),
                                   attn_bias=attn_bias_1,
                                   key_padding_mask=None,
                                   is_causal=True,
                                   dropout_p=0.0,
                                   training=False,
                                   needs_weights=False,
                                   multiquery=False,
                                   attention_mask_in_length=None,
                                   should_repeat_kv_for_gqa=True)

    output_1.sum().backward()

    query_2 = query_1.detach().clone()
    query_2.requires_grad = True
    key_2 = key_1.detach().clone()
    key_2.requires_grad = True
    value_2 = value_1.detach().clone()
    value_2.requires_grad = True

    def gen_bias():
        causal = True
        bs = attention.attn_bias_shape('triton',
                                       n_heads,
                                       seqlen_1,
                                       True,
                                       prefix_lm=False,
                                       use_sequence_id=False,
                                       causal=causal)

        attn_bias = torch.zeros(*bs, device=device)
        attn_bias = attention.build_attn_bias(
            'triton',
            attn_bias,
            n_heads,
            seqlen_1,
            causal=causal,
            alibi=True,
            alibi_bias_max=8,
        )
        return attn_bias

    attn_bias_2 = gen_bias()

    output_2, _, _ = triton_flash_attn_fn(
        query=query_2,
        key=key_2,
        value=value_2,
        n_heads=n_heads,
        kv_n_heads=n_heads,
        past_key_value=None,
        softmax_scale=1 / math.sqrt(d),
        attn_bias=attn_bias_2,
        key_padding_mask=None,
        is_causal=True,
        dropout_p=0.0,
        training=False,
        needs_weights=False,
        multiquery=False,
    )

    output_2.sum().backward()

    assert torch.allclose(output_1, output_2)
    assert torch.norm(query_2.grad - query_1.grad  # type: ignore
                     ) <= 1e-2 + 1e-2 * torch.norm(query_2.grad)
    assert torch.norm(key_2.grad - key_1.grad  # type: ignore
                     ) <= 1e-2 + 1e-2 * torch.norm(key_2.grad)
    assert torch.norm(value_2.grad - value_1.grad  # type: ignore
                     ) <= 1e-2 + 1e-2 * torch.norm(value_2.grad)
