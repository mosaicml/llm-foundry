# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

# Code modified from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py

import torch
from torch import nn


class RotaryEmbedding(nn.Module):

    def __init__(self, dim: int, max_position_embeddings: int, base: int,
                 coefficient: float, shift: int, device: torch.device,
                 dtype: torch.dtype):
        super().__init__()

        self.max_position_embeddings = max_position_embeddings

        inv_freq = 1.0 / (coefficient * (base**(
            torch.arange(0 + shift, dim + shift, 2).float().to(device) / dim)))
        t = torch.arange(self.max_position_embeddings).to(inv_freq)

        freqs = torch.einsum('i,j->ij', t, inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()[None, None, :, :].to(dtype)
        self.sin_cached = emb.sin()[None, None, :, :].to(dtype)

    def forward(self, dtype: torch.dtype, device: torch.device, seq_len: int):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_position_embeddings:
            raise ValueError(
                'The sequence length is greater than the maximum sequence length.'
            )

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=dtype, device=device),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=dtype, device=device),
        )


def rotate_half(x: torch.Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor,
                         sin: torch.Tensor, position_ids: torch.Tensor):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(q), k_embed.to(k)
