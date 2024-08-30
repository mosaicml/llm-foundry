# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
from composer.utils import reproducibility

from llmfoundry.models.layers.attention import (
    attention_implementations,
    scaled_multihead_dot_product_attention,
)
from llmfoundry.models.layers.layer_builders import build_attention_layer
from llmfoundry.models.mpt.modeling_mpt import gen_flash_attn_padding_info


@pytest.mark.parametrize(
    'attn_name',
    ['multihead_attention', 'grouped_query_attention', 'multiquery_attention'],
)
@pytest.mark.parametrize('dim', [1024])
def test_unfused_wqkv(attn_name: str, dim: int):
    d_head = 128
    n_heads = dim // d_head

    generic_attn_kwargs = {
        'd_model': dim,
        'n_heads': n_heads,
        'fc_type': {
            'name': 'torch',
        },
        'device': 'cpu',
        'attn_pdrop': 0.0,
        'attn_impl': 'torch',
        'qk_ln': False,
        'qk_gn': False,
        'clip_qkv': None,
        'softmax_scale': None,
        'sliding_window_size': -1,
    }

    if attn_name == 'grouped_query_attention':
        kv_n_heads = 2
        generic_attn_kwargs['kv_n_heads'] = kv_n_heads
    elif attn_name == 'multiquery_attention':
        kv_n_heads = 1
    elif attn_name == 'multihead_attention':
        kv_n_heads = n_heads
    else:
        raise ValueError(f'Unknown attention name: {attn_name}')

    attn_config_fused = generic_attn_kwargs.copy()
    attn_config_fused['fused_qkv'] = True

    attn_config_unfused = generic_attn_kwargs.copy()
    attn_config_unfused['fused_qkv'] = False

    attn_fused = build_attention_layer(
        name=attn_name,
        attn_kwargs=attn_config_fused,
    )
    attn_unfused = build_attention_layer(
        name=attn_name,
        attn_kwargs=attn_config_unfused,
    )

    # Make sure unfused attention has the same params as the fused one.
    fused_wqkv = attn_fused.Wqkv.weight.detach().clone()
    kv_heads_len = (fused_wqkv.shape[0] - dim) // 2
    Wq_shape_before = (attn_unfused.Wq.weight.shape, attn_unfused.Wq.bias.shape)
    Wk_shape_before = (attn_unfused.Wk.weight.shape, attn_unfused.Wk.bias.shape)
    Wv_shape_before = (attn_unfused.Wv.weight.shape, attn_unfused.Wv.bias.shape)

    attn_unfused.Wq.weight.data = fused_wqkv[:dim, :]
    attn_unfused.Wk.weight.data = fused_wqkv[dim:dim + kv_heads_len, :]
    attn_unfused.Wv.weight.data = fused_wqkv[dim + kv_heads_len:, :]
    attn_unfused.out_proj.weight.data = attn_fused.out_proj.weight
    attn_unfused.Wq.bias.data = attn_fused.Wqkv.bias[:dim]
    attn_unfused.Wk.bias.data = attn_fused.Wqkv.bias[dim:dim + kv_heads_len]
    attn_unfused.Wv.bias.data = attn_fused.Wqkv.bias[dim + kv_heads_len:]
    attn_unfused.out_proj.bias.data = attn_fused.out_proj.bias

    # Make sure initialization fuse splits are as expected.
    all_fuse_splits = (
        0,
        [i * d_head for i in range(1, n_heads + 2 * kv_n_heads)],
    )
    q_fuse_splits = (0, [i * d_head for i in range(1, n_heads)])
    kv_fuse_splits = (0, [i * d_head for i in range(1, kv_n_heads)])

    assert attn_fused.Wqkv._fused == all_fuse_splits
    assert attn_unfused.Wq._fused == q_fuse_splits
    assert attn_unfused.Wk._fused == kv_fuse_splits
    assert attn_unfused.Wv._fused == kv_fuse_splits

    assert torch.allclose(
        attn_fused.Wqkv.weight,
        torch.cat(
            [
                attn_unfused.Wq.weight,
                attn_unfused.Wk.weight,
                attn_unfused.Wv.weight,
            ],
            dim=0,
        ),
    )
    assert torch.allclose(
        attn_fused.Wqkv.bias,
        torch.cat(
            [
                attn_unfused.Wq.bias,
                attn_unfused.Wk.bias,
                attn_unfused.Wv.bias,
            ],
            dim=0,
        ),
    )
    assert torch.allclose(
        attn_fused.out_proj.weight,
        attn_unfused.out_proj.weight,
    )
    assert torch.allclose(attn_fused.out_proj.bias, attn_unfused.out_proj.bias)

    assert Wq_shape_before == (
        attn_unfused.Wq.weight.shape,
        attn_unfused.Wq.bias.shape,
    )
    assert Wk_shape_before == (
        attn_unfused.Wk.weight.shape,
        attn_unfused.Wk.bias.shape,
    )
    assert Wv_shape_before == (
        attn_unfused.Wv.weight.shape,
        attn_unfused.Wv.bias.shape,
    )

    x1 = torch.randn(1, 1, dim)
    x2 = x1.detach().clone()
    x1.requires_grad = True
    x2.requires_grad = True

    out_fused, _, _ = attn_fused(x1)
    out_unfused, _, _ = attn_unfused(x2)

    assert torch.allclose(out_fused, out_unfused)

    # Dummy loss function is simply the sum.
    loss_fused = out_fused.sum()
    loss_fused.backward()

    loss_unfused = out_unfused.sum()
    loss_unfused.backward()

    assert isinstance(x1.grad, torch.Tensor)
    assert isinstance(x2.grad, torch.Tensor)
    assert torch.allclose(x1.grad, x2.grad)
    combined_grad = torch.concat(
        [
            attn_unfused.Wq.weight.grad,
            attn_unfused.Wk.weight.grad,
            attn_unfused.Wv.weight.grad,
        ],
        dim=0,
    )
    assert isinstance(attn_fused.Wqkv.weight.grad, torch.Tensor)
    assert isinstance(combined_grad, torch.Tensor)
    assert torch.allclose(attn_fused.Wqkv.weight.grad, combined_grad)


@pytest.mark.gpu
@pytest.mark.parametrize('sliding_window_size', [1, 4, 8])
@pytest.mark.parametrize('attn_impl', ['flash', 'torch'])
def test_sliding_window(sliding_window_size: int, attn_impl: str):
    # Test that sliding window attention works as expected.
    dtype = torch.bfloat16
    device = 'cuda'
    d = 128
    n_heads = 8
    seqlen_1 = 8
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

    attn_extra_kwargs = {}
    if attn_impl == 'flash':
        attn_extra_kwargs = {
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
        sliding_window_size=sliding_window_size,
        **attn_extra_kwargs,
    )

    output_1.sum().backward()

    query_2 = query_1.detach().clone()
    query_2.requires_grad = True
    key_2 = key_1.detach().clone()
    key_2.requires_grad = True
    value_2 = value_1.detach().clone()
    value_2.requires_grad = True

    attn_bias_2 = torch.zeros(1, 1, seqlen_1,
                              seqlen_1).to(dtype=dtype, device=device)

    window_mask_2 = torch.tril(
        torch.ones(seqlen_1, seqlen_1),
        diagonal=-(sliding_window_size + 1),
    ).to(dtype=dtype, device=device) * torch.finfo(attn_bias_2.dtype).min
    attn_bias_2 = attn_bias_2 + window_mask_2
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

    print(torch.max(output_1 - output_2))

    _assert_approx_equal(output_1, output_2)
    assert (query_2.grad is not None) and (query_1.grad is not None)
    _assert_approx_equal(query_1.grad, query_2.grad)
    assert (key_2.grad is not None) and (key_1.grad is not None)
    _assert_approx_equal(key_1.grad, key_2.grad)
    assert (value_2.grad is not None) and (value_1.grad is not None)
    _assert_approx_equal(value_1.grad, value_2.grad)


def _assert_approx_equal(value1: torch.Tensor, value2: torch.Tensor):
    assert torch.norm(value2 - value1) <= 1e-2 + 1e-2 * torch.norm(value2)


@pytest.mark.parametrize(
    'attn_name',
    ['multihead_attention', 'grouped_query_attention', 'multiquery_attention'],
)
@pytest.mark.parametrize('dim', [1024])
def test_cross_attn_as_self_attn(attn_name: str, dim: int):
    d_head = 128
    n_heads = dim // d_head

    generic_attn_kwargs = {
        'd_model': dim,
        'n_heads': n_heads,
        'fc_type': {
            'name': 'torch',
        },
        'device': 'cpu',
        'attn_pdrop': 0.0,
        'attn_impl': 'torch',
        'qk_ln': False,
        'qk_gn': False,
        'clip_qkv': None,
        'softmax_scale': None,
        'sliding_window_size': -1,
    }

    if attn_name == 'grouped_query_attention':
        kv_n_heads = 2
        generic_attn_kwargs['kv_n_heads'] = kv_n_heads
    elif attn_name == 'multiquery_attention':
        kv_n_heads = 1
    elif attn_name == 'multihead_attention':
        kv_n_heads = n_heads
    else:
        raise ValueError(f'Unknown attention name: {attn_name}')

    attn_config = generic_attn_kwargs.copy()
    attn_config['fused_qkv'] = False

    attn_layer = build_attention_layer(
        name=attn_name,
        attn_kwargs=attn_config,
    )

    x1 = torch.randn(1, 1, dim)
    x2 = x1.detach().clone()

    out_fused, _, _ = attn_layer(x1)
    out_unfused, _, _ = attn_layer(x1, key_value_states=x2)

    assert torch.allclose(out_fused, out_unfused)


@pytest.mark.parametrize(
    'attn_name',
    ['multihead_attention', 'grouped_query_attention', 'multiquery_attention'],
)
@pytest.mark.parametrize('dim', [1024])
def test_cross_attn_kv_dim(attn_name: str, dim: int):
    d_head = 128
    n_heads = dim // d_head

    generic_attn_kwargs = {
        'd_model': dim,
        'n_heads': n_heads,
        'fc_type': {
            'name': 'torch',
        },
        'device': 'cpu',
        'attn_pdrop': 0.0,
        'attn_impl': 'torch',
        'qk_ln': False,
        'qk_gn': False,
        'clip_qkv': None,
        'softmax_scale': None,
        'sliding_window_size': -1,
    }

    if attn_name == 'grouped_query_attention':
        kv_n_heads = 2
        generic_attn_kwargs['kv_n_heads'] = kv_n_heads
    elif attn_name == 'multiquery_attention':
        kv_n_heads = 1
    elif attn_name == 'multihead_attention':
        kv_n_heads = n_heads
    else:
        raise ValueError(f'Unknown attention name: {attn_name}')

    # layer with only dim passed in
    attn_config = generic_attn_kwargs.copy()
    attn_config['fused_qkv'] = False

    reproducibility.seed_all(42)
    attn_layer_no_kv = build_attention_layer(
        name=attn_name,
        attn_kwargs=attn_config,
    )

    # layer with kv_dim = dim passed in
    attn_config = generic_attn_kwargs.copy()
    attn_config['fused_qkv'] = False
    attn_config['kv_dim'] = dim

    reproducibility.seed_all(42)
    attn_layer_kv = build_attention_layer(
        name=attn_name,
        attn_kwargs=attn_config,
    )

    x1 = torch.randn(1, 1, dim)

    out_fused, _, _ = attn_layer_no_kv(x1)
    out_unfused, _, _ = attn_layer_kv(x1)

    assert torch.allclose(out_fused, out_unfused)
