# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Tuple

import torch
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

from llmfoundry.models.mpt.modeling_mpt import gen_rotary_embedding

rope_config = {
    'rope_theta': 500000.0,
    'rope_impl': 'hf',
    'rope_hf_config': {
        'factor': 8.0,
        'low_freq_factor': 1.0,
        'high_freq_factor': 4.0,
        'original_max_position_embeddings': 8192,
        'type': 'llama3',
    },
}

rope_dail_config = {}


def apply_scaling(freqs: torch.Tensor):
    # Values obtained from grid search
    scale_factor = 8
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192  # original llama3 length

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen -
                      low_freq_factor) / (high_freq_factor - low_freq_factor)
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
    use_scaled: bool = False,
):
    freqs = 1.0 / (theta**(torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    if use_scaled:
        freqs = apply_scaling(freqs)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def test_rope_scaling():
    d_model = 128
    n_heads = 32
    max_seq_len = 65536

    embedding = gen_rotary_embedding(
        d_model=d_model,
        n_heads=n_heads,
        rope_dail_config=rope_dail_config,
        max_seq_len=max_seq_len,
        **rope_config,
    )

    assert isinstance(embedding, LlamaRotaryEmbedding)

    x = torch.randn(1, max_seq_len, d_model)
    position_ids = torch.arange(max_seq_len).unsqueeze(0)

    freqs_cis = precompute_freqs_cis(
        d_model,
        max_seq_len,
        rope_config['rope_theta'],
    )

    rope_embeddings = embedding.forward(x, position_ids)

    # ??? WIP
