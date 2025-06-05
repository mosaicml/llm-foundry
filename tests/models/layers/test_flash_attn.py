# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Optional

import pytest
import torch
from packaging import version
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from llmfoundry.models.layers.attention import (
    attention_implementations,
    attn_bias_shape,
    build_attn_bias,
    check_alibi_support,
    gen_slopes,
    is_flash_v2_installed,
    scaled_multihead_dot_product_attention,
)
from llmfoundry.models.layers.flex_attn_utils import FLEX_ATTN_COMPILE
from llmfoundry.models.mpt.modeling_mpt import gen_flash_attn_padding_info

compiled_flex_attention = flex_attention
compiled_create_block_mask = create_block_mask
if FLEX_ATTN_COMPILE:
    compiled_flex_attention = torch.compile(flex_attention)
    compiled_create_block_mask = torch.compile(create_block_mask)


@pytest.mark.gpu
@pytest.mark.skipif(
    not is_flash_v2_installed(),
    reason='GQA natively only supported by Flash Attention after v2.',
)
@pytest.mark.parametrize('attn_impl', ['flash', 'flex'])
@pytest.mark.parametrize('kv_n_heads', [1, 4, 8])
def test_gqa_kv_repetition(attn_impl: str, kv_n_heads: int):
    # Test that flash attention v2 with GQA (kv_n_heads < n_heads) works the same
    # whether we repeat the kv_n_heads explicitly or flash attention v2 handles it on its own.
    if attn_impl == 'flex' and version.parse(
        torch.__version__.split('.dev')[0],
    ) < version.parse('2.5.1'):
        pytest.skip(
            'FlexAttention is not supported in torch version {torch.__version__}<2.5.1.',
        )
    d = 128
    n_heads = 8
    seqlen_1 = 6 if attn_impl != 'flex' else 128  # FlexAttention requires seqlen to be a multiple of 128 (to compute gradients I think). More info: https://pytorch.org/blog/flexattention/#limitations-and-future-work
    bsz = 2

    query_1 = torch.randn(bsz, seqlen_1, n_heads * d).to(torch.bfloat16).cuda()
    query_1.requires_grad = True
    key_1 = torch.randn(bsz, seqlen_1, kv_n_heads * d).to(torch.bfloat16).cuda()
    key_1.requires_grad = True
    value_1 = torch.randn(bsz, seqlen_1,
                          kv_n_heads * d).to(torch.bfloat16).cuda()
    value_1.requires_grad = True

    extra_attn_kwargs = {}
    if attn_impl == 'flash':
        extra_attn_kwargs = {
            'flash_attn_padding_info':
                gen_flash_attn_padding_info(
                    bsz,
                    seqlen_1,
                    0,
                    query_1.device,
                    None,
                    None,
                ),
            'should_repeat_kv_for_gqa':
                True,
        }
    elif attn_impl == 'flex':
        extra_attn_kwargs = {
            'compiled_flex_attention': compiled_flex_attention,
            'compiled_create_block_mask': compiled_create_block_mask,
            'flex_attn_compile': FLEX_ATTN_COMPILE,
            'sequence_id_info': {},
        }

    output_1, _, _ = attention_implementations.get(attn_impl)(
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
        **extra_attn_kwargs,
    )

    output_1.sum().backward()

    query_2 = query_1.detach().clone()
    query_2.requires_grad = True
    key_2 = key_1.detach().clone()
    key_2.requires_grad = True
    value_2 = value_1.detach().clone()
    value_2.requires_grad = True
    extra_attn_kwargs = {}
    if attn_impl == 'flash':
        extra_attn_kwargs = {
            'flash_attn_padding_info':
                gen_flash_attn_padding_info(
                    bsz,
                    seqlen_1,
                    0,
                    query_2.device,
                    None,
                    None,
                ),
            'should_repeat_kv_for_gqa':
                False,
        }
    elif attn_impl == 'flex':
        extra_attn_kwargs = {
            'compiled_flex_attention': compiled_flex_attention,
            'compiled_create_block_mask': compiled_create_block_mask,
            'flex_attn_compile': FLEX_ATTN_COMPILE,
            'sequence_id_info': {},
        }

    output_2, _, _ = attention_implementations.get(attn_impl)(
        query=query_2,
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
        **extra_attn_kwargs,
    )

    output_2.sum().backward()
    assert torch.allclose(output_1, output_2)
    assert torch.allclose(query_1.grad, query_2.grad)  # type: ignore
    assert torch.allclose(key_1.grad, key_2.grad)  # type: ignore
    assert torch.allclose(value_1.grad, value_2.grad)  # type: ignore


@pytest.mark.gpu
@pytest.mark.skipif(
    not is_flash_v2_installed(v2_version='v2.1.2'),
    reason=
    'Using sequence id with flash attention requires flash attention v2.1.2 or higher.',
)
@pytest.mark.parametrize('attn_impl', ['flash', 'flex'])
def test_seq_id_masking_FA_v2(attn_impl: str):
    # Test that flash attention v2 with sequence id masking works correctly.
    if attn_impl == 'flex' and version.parse(
        torch.__version__.split('.dev')[0],
    ) < version.parse('2.5.1'):
        pytest.skip(
            'FlexAttention is not supported in torch version {torch.__version__}<2.5.1.',
        )
    d = 128  # TODO: Compiled FlexAttention works for d=16 with seqlen=6, but not for d=128 with seqlen=6. For seqlen=128, all d's in [16, 32, 64, 128, 256] work. Probably because this is not yet fixed: https://pytorch.org/blog/flexattention/#limitations-and-future-work
    n_heads = 4
    kv_n_heads = 4
    seqlen_1 = 128
    bsz = 2

    query_1 = torch.randn(bsz, seqlen_1, n_heads * d).to(torch.bfloat16).cuda()
    query_1.requires_grad = True
    key_1 = torch.randn(bsz, seqlen_1, kv_n_heads * d).to(torch.bfloat16).cuda()
    key_1.requires_grad = True
    value_1 = torch.randn(bsz, seqlen_1,
                          kv_n_heads * d).to(torch.bfloat16).cuda()
    value_1.requires_grad = True

    seq_ranges = [
        (0, 3),
        (3, 5),
        (5, 6),
    ]  # Each batch has 3 sequences of length 3, 2, and 1 respectively.
    attention_mask_in_length_1 = torch.tensor([
        [3, 2, 1] + [0] * (seqlen_1 - 3),
        [3, 2, 1] + [0] * (seqlen_1 - 3),
    ]).to(torch.int64).cuda()
    sequence_id = torch.tensor([[0, 0, 0, 1, 1, 2] + [-1] *
                                (seqlen_1 - 6), [0, 0, 0, 1, 1, 2] + [-1] *
                                (seqlen_1 - 6)],).to(torch.int64).cuda()

    flash_attn_padding_info_1 = gen_flash_attn_padding_info(
        bsz,
        seqlen_1,
        0,
        query_1.device,
        attention_mask_in_length_1,
        None,
    )
    extra_attn_kwargs = {}
    if attn_impl == 'flash':
        extra_attn_kwargs['flash_attn_padding_info'] = flash_attn_padding_info_1
    elif attn_impl == 'flex':
        extra_attn_kwargs = {
            'compiled_flex_attention': compiled_flex_attention,
            'compiled_create_block_mask': compiled_create_block_mask,
            'flex_attn_compile': FLEX_ATTN_COMPILE,
            'sequence_id_info': {
                'sequence_id': sequence_id,
            },
        }
    output_1, _, _ = attention_implementations.get(attn_impl)(
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
        **extra_attn_kwargs,
    )

    output_1.sum().backward()

    for seq_range in seq_ranges:
        query_2 = query_1.detach().clone()[:, seq_range[0]:seq_range[1], :]
        query_2.requires_grad = True
        key_2 = key_1.detach().clone()[:, seq_range[0]:seq_range[1], :]
        key_2.requires_grad = True
        value_2 = value_1.detach().clone()[:, seq_range[0]:seq_range[1], :]
        value_2.requires_grad = True

        flash_attn_padding_info_2 = gen_flash_attn_padding_info(
            bsz,
            seq_range[1] - seq_range[0],
            0,
            query_2.device,
            None,
            None,
        )
        attn_impl = 'flash'
        extra_attn_kwargs = {
            'flash_attn_padding_info': flash_attn_padding_info_2,
        }
        output_2, _, _ = attention_implementations.get(attn_impl)(
            query=query_2,
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
            **extra_attn_kwargs,
        )

        output_2.sum().backward()
        torch.testing.assert_close(
            output_1[:, seq_range[0]:seq_range[1], :],
            output_2,
        )
        assert torch.allclose(
            query_1.grad[:, seq_range[0]:seq_range[1], :],  # type: ignore
            query_2.grad,  # type: ignore
        )
        assert torch.allclose(
            key_1.grad[:, seq_range[0]:seq_range[1], :],  # type: ignore
            key_2.grad,  # type: ignore
        )
        assert torch.allclose(
            value_1.grad[:, seq_range[0]:seq_range[1], :],  # type: ignore
            value_2.grad,  # type: ignore
        )


@pytest.mark.gpu
@pytest.mark.skipif(
    not check_alibi_support('flash'),
    reason='ALiBi only supported by Flash Attention after v2.4.2.',
)
@pytest.mark.parametrize('attn_impl', ['flash', 'flex'])
@pytest.mark.parametrize('n_heads', [1, 6, 8])
def test_alibi_bias(attn_impl: str, n_heads: int):
    # Test that sliding window attention works as expected.
    if attn_impl == 'flex' and version.parse(
        torch.__version__.split('.dev')[0],
    ) < version.parse('2.5.1'):
        pytest.skip(
            'FlexAttention is not supported in torch version {torch.__version__}<2.5.1.',
        )
    if attn_impl == 'flex' and n_heads != 8:
        pytest.skip(
            'FlexAttention passes the test individually for n_heads=1, 6, and 8, but not when all three are configured.',
        )  # TODO: Investigate why this is the case.
    dtype = torch.bfloat16
    device = 'cuda'
    d = 128
    seqlen_1 = 6 if attn_impl != 'flex' else 128  # TODO: FlexAttention requires seqlen to be a multiple of 128 (to compute gradients I think). More info: https://pytorch.org/blog/flexattention/#limitations-and-future-work
    bsz = 2

    query_1 = torch.randn(bsz, seqlen_1,
                          n_heads * d).to(dtype=dtype, device=device)
    query_1.requires_grad = True
    key_1 = torch.randn(bsz, seqlen_1,
                        n_heads * d).to(dtype=dtype, device=device)
    key_1.requires_grad = True
    value_1 = torch.randn(bsz, seqlen_1,
                          n_heads * d).to(dtype=dtype, device=device)
    value_1.requires_grad = True
    alibi_slopes_1 = gen_slopes(
        n_heads=n_heads,
        alibi_bias_max=8,
        device=torch.device(device),
        return_1d=True,
    )
    extra_attn_kwargs = {}
    if attn_impl == 'flash':
        extra_attn_kwargs = {
            'flash_attn_padding_info':
                gen_flash_attn_padding_info(
                    bsz,
                    seqlen_1,
                    0,
                    query_1.device,
                    None,
                    None,
                ),
            'should_repeat_kv_for_gqa':
                True,
        }
    elif attn_impl == 'flex':
        extra_attn_kwargs = {
            'compiled_flex_attention': compiled_flex_attention,
            'compiled_create_block_mask': compiled_create_block_mask,
            'flex_attn_compile': FLEX_ATTN_COMPILE,
            'sequence_id_info': {},
        }
    output_1, _, _ = attention_implementations.get(attn_impl)(
        query=query_1,
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
        alibi_slopes=alibi_slopes_1,
        **extra_attn_kwargs,
    )

    output_1.sum().backward()

    query_2 = query_1.detach().clone()
    query_2.requires_grad = True
    key_2 = key_1.detach().clone()
    key_2.requires_grad = True
    value_2 = value_1.detach().clone()
    value_2.requires_grad = True

    def gen_bias():
        causal = True
        bs = attn_bias_shape(
            'torch',
            n_heads,
            seqlen_1,
            True,
            use_sequence_id=False,
            causal=causal,
        )

        attn_bias = torch.zeros(*bs, device=device)
        attn_bias = build_attn_bias(
            'torch',
            attn_bias,
            n_heads,
            seqlen_1,
            causal=causal,
            alibi=True,
            alibi_bias_max=8,
        )
        return attn_bias

    attn_bias_2 = gen_bias()

    output_2, _, _ = scaled_multihead_dot_product_attention(
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
    )

    output_2.sum().backward()

    _assert_approx_equal(output_1, output_2)
    assert (query_2.grad is not None) and (query_1.grad is not None)
    _assert_approx_equal(query_1.grad, query_2.grad)
    assert (key_2.grad is not None) and (key_1.grad is not None)
    _assert_approx_equal(key_1.grad, key_2.grad)
    assert (value_2.grad is not None) and (value_1.grad is not None)
    _assert_approx_equal(value_1.grad, value_2.grad)


@pytest.mark.gpu
@pytest.mark.skipif(
    not is_flash_v2_installed(v2_version='v2.6.2'),
    reason=
    'attn_logit_softcapping only supported by Flash Attention after v2.6.2.',
)
@pytest.mark.parametrize('attn_impl', ['flash', 'flex'])
@pytest.mark.parametrize(
    'attn_logit_softcapping',
    [None, 0.1, 1.0, 10.0, 100.0],
)
def test_attn_logit_softcapping(
    attn_impl: str,
    attn_logit_softcapping: Optional[float],
):
    # Test that attn_logit_softcapping in attention works as expected.
    if attn_impl == 'flex' and version.parse(
        torch.__version__.split('.dev')[0],
    ) < version.parse('2.5.1'):
        pytest.skip(
            'FlexAttention is not supported in torch version {torch.__version__}<2.5.1.',
        )
    if attn_impl == 'flex' and attn_logit_softcapping is not None:
        if int(attn_logit_softcapping) != attn_logit_softcapping:
            pytest.skip(
                'FlexAttention does not support attn_logit_softcapping with float softcap temperature.',
            )
        else:
            attn_logit_softcapping = int(attn_logit_softcapping)

    dtype = torch.bfloat16
    device = 'cuda'
    d = 128
    seqlen_1 = 8 if attn_impl != 'flex' else 128  # FlexAttention requires seqlen to be a multiple of 128 (to compute gradients I think). More info: https://pytorch.org/blog/flexattention/#limitations-and-future-work
    bsz = 2
    n_heads = 4

    query_1 = torch.randn(bsz, seqlen_1,
                          n_heads * d).to(dtype=dtype, device=device)
    query_1.requires_grad = True
    key_1 = torch.randn(bsz, seqlen_1,
                        n_heads * d).to(dtype=dtype, device=device)
    key_1.requires_grad = True
    value_1 = torch.randn(bsz, seqlen_1,
                          n_heads * d).to(dtype=dtype, device=device)
    value_1.requires_grad = True
    extra_attn_kwargs = {}
    if attn_impl == 'flash':
        extra_attn_kwargs = {
            'flash_attn_padding_info':
                gen_flash_attn_padding_info(
                    bsz,
                    seqlen_1,
                    0,
                    query_1.device,
                    None,
                    None,
                ),
            'should_repeat_kv_for_gqa':
                True,
        }
    elif attn_impl == 'flex':
        extra_attn_kwargs = {
            'compiled_flex_attention': compiled_flex_attention,
            'compiled_create_block_mask': compiled_create_block_mask,
            'flex_attn_compile': FLEX_ATTN_COMPILE,
            'sequence_id_info': {},
        }
    output_1, _, _ = attention_implementations.get(attn_impl)(
        query=query_1,
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
        attn_logit_softcapping=attn_logit_softcapping,
        **extra_attn_kwargs,
    )
    output_1.sum().backward()

    query_2 = query_1.detach().clone()
    query_2.requires_grad = True
    key_2 = key_1.detach().clone()
    key_2.requires_grad = True
    value_2 = value_1.detach().clone()
    value_2.requires_grad = True
    output_2, _, _ = scaled_multihead_dot_product_attention(
        query=query_2,
        key=key_2,
        value=value_2,
        n_heads=n_heads,
        kv_n_heads=n_heads,
        past_key_value=None,
        softmax_scale=1 / math.sqrt(d),
        key_padding_mask=None,
        is_causal=True,
        dropout_p=0.0,
        training=False,
        needs_weights=False,
        attn_logit_softcapping=attn_logit_softcapping,
    )
    output_2.sum().backward()

    _assert_approx_equal(output_1, output_2)
    assert (query_2.grad is not None) and (query_1.grad is not None)
    _assert_approx_equal(query_1.grad, query_2.grad)
    assert (key_2.grad is not None) and (key_1.grad is not None)
    _assert_approx_equal(key_1.grad, key_2.grad)
    assert (value_2.grad is not None) and (value_1.grad is not None)
    _assert_approx_equal(value_1.grad, value_2.grad)


def _assert_approx_equal(
    value1: torch.Tensor,
    value2: torch.Tensor,
    atol: float = 1e-2,
    rtol: float = 1e-2,
):
    actual_difference = torch.norm(value2 - value1)
    allowed_difference = atol + rtol * torch.norm(value2)
    assert actual_difference < allowed_difference, f'{actual_difference=}, {allowed_difference=}'
