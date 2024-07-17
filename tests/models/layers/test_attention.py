# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from llmfoundry.models.layers.layer_builders import build_attention_layer


@pytest.mark.parametrize(
    'attn_name',
    ['multihead_attention', 'grouped_query_attention', 'multiquery_attention'],
)
@pytest.mark.parametrize('dim', [1024])
def test_unfused_wqkv(attn_name: str, dim: int):
    d_head = 128

    generic_attn_kwargs = {
        'd_model': dim,
        'n_heads': dim // d_head,
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
        generic_attn_kwargs['kv_n_heads'] = 2

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
