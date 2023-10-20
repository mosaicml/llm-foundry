# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from composer.utils import reproducibility


def allclose_helper(t0: torch.Tensor,
                    t1: torch.Tensor,
                    rtol: float = 1e-2,
                    atol: float = 1e-2):
    return torch.allclose(t0, t1, rtol=rtol, atol=atol)


@pytest.mark.gpu
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('rope_scaling_type',
                         ['no_scaling', 'linear', 'dynamic'])
@pytest.mark.parametrize('tensor_type', ['query', 'key'])
def test_rotation_scaling_factor_1(device: str, dtype: torch.dtype,
                                   rope_scaling_type: str, tensor_type: str):
    """Checks all the rotation embedding techniques (with scaling factor 1)
    produce the expected rotation."""
    from llmfoundry.models.layers.rotary_embedding import (
        DynamicNTKScalingRotaryEmbedding, LinearScalingRotaryEmbedding,
        RotaryEmbedding, apply_rotary_pos_emb)

    reproducibility.seed_all(7)

    rope_head_dim = 8
    assert rope_head_dim % 2 == 0
    rope_theta = 5
    rope_scaling_factor = 1.0

    seq_len = 7
    batch_size = 1
    num_heads = 1
    pos = torch.arange(seq_len, device=device,
                       dtype=torch.long).repeat(batch_size, 1)  #

    # x will test the first half cosine part of the rotation and second half of the sine part
    x = torch.ones((batch_size, seq_len, num_heads, rope_head_dim),
                   device=device,
                   dtype=dtype)
    x[..., rope_head_dim // 2:] = 0.0

    # y will test the first half sine part of the rotation and second half of the cosine part
    y = torch.ones((batch_size, seq_len, num_heads, rope_head_dim),
                   device=device,
                   dtype=dtype)
    y[..., :rope_head_dim // 2] = 0.0

    z = torch.zeros((batch_size, seq_len, num_heads, rope_head_dim),
                    device=device,
                    dtype=dtype)

    def gen_rotary_emb():
        if rope_scaling_type == 'no_scaling':
            rotary_embedding = RotaryEmbedding(rope_head_dim,
                                               max_position_embeddings=seq_len,
                                               base=rope_theta,
                                               device=device)
        elif rope_scaling_type == 'linear':
            rotary_embedding = LinearScalingRotaryEmbedding(
                rope_head_dim,
                max_position_embeddings=seq_len,
                base=rope_theta,
                device=device,
                scaling_factor=rope_scaling_factor)
        elif rope_scaling_type == 'dynamic':
            rotary_embedding = DynamicNTKScalingRotaryEmbedding(
                rope_head_dim,
                max_position_embeddings=seq_len,
                base=rope_theta,
                device=device,
                scaling_factor=rope_scaling_factor)
        else:
            raise ValueError(
                'rope_scaling_type should be one no_scaling, linear, or dynamic'
            )
        return rotary_embedding

    rotary_emb = gen_rotary_emb()
    (cos, sin) = rotary_emb(x, seq_len)
    assert allclose_helper(cos[:, :rope_head_dim // 2],
                           cos[:, rope_head_dim // 2:])
    assert allclose_helper(sin[:, :rope_head_dim // 2],
                           sin[:, rope_head_dim // 2:])

    assert allclose_helper(cos * cos + sin * sin, torch.ones_like(cos))

    if tensor_type == 'query':
        x, _ = apply_rotary_pos_emb(x, z, cos, sin, pos, dim_heads_index=2)
        y, _ = apply_rotary_pos_emb(y, z, cos, sin, pos, dim_heads_index=2)
    elif tensor_type == 'key':
        _, x = apply_rotary_pos_emb(z, x, cos, sin, pos, dim_heads_index=2)
        _, y = apply_rotary_pos_emb(z, y, cos, sin, pos, dim_heads_index=2)

    assert allclose_helper(x[..., :rope_head_dim // 2], y[...,
                                                          rope_head_dim // 2:])
    assert allclose_helper(x[..., rope_head_dim // 2:],
                           -y[..., :rope_head_dim // 2])

    inv_freq = (
        1.0 /
        (rope_theta**(torch.arange(0, rope_head_dim, 2) / rope_head_dim))).to(x)
    t = torch.arange(seq_len).to(x)

    expected_rotation_angles = torch.outer(t, inv_freq)
    assert allclose_helper(x[..., :rope_head_dim // 2].squeeze(),
                           expected_rotation_angles.cos())
    assert allclose_helper(x[..., rope_head_dim // 2:].squeeze(),
                           expected_rotation_angles.sin())
