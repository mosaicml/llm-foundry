# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

# Code modified from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py

import torch
from torch import nn


class RotaryEmbedding(nn.Module):

    def __init__(self, dim: int, base: float):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len_cached = -1

        self.caches_initialized = False
        self.cos_cached = torch.Tensor()
        self.sin_cached = torch.Tensor()

    def _set_cos_sin_cache(self, x: torch.Tensor, seq_len: int):
        self.max_seq_len_cached = seq_len
        inv_freq = self._get_inv_freq(x, seq_len)
        t = self._get_t(x)
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()
        self.sin_cached = emb.sin()
        self.caches_initialized = True

    def _get_t(self, x: torch.Tensor):
        t = torch.arange(self.max_seq_len_cached).to(x)
        return t

    def _get_inv_freq(self, x: torch.Tensor, seq_len: int):
        del seq_len
        inv_freq = (
            1.0 / (self.base**(torch.arange(0, self.dim, 2) / self.dim))).to(x)
        return inv_freq

    @torch.no_grad()
    def forward(self, x: torch.Tensor, seq_len: int):
        # x is only used to get the correct dtype and device
        if (not self.caches_initialized) or seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, x=x)

        return (
            self.cos_cached[:seq_len].to(x),
            self.sin_cached[:seq_len].to(x),
        )


class LinearScalingRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with linear scaling.

    Credits to the Reddit user /u/kaiokendev
    """

    def __init__(self, dim: int, base: float, scaling_factor: float):
        self.scaling_factor = scaling_factor
        super().__init__(dim, base)

    def _get_t(self, x: torch.Tensor):
        t = (torch.arange(self.max_seq_len_cached) / self.scaling_factor).to(x)
        return t


class DynamicNTKScalingRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with Dynamic NTK scaling.

    Credits to the Reddit users /u/bloc97 and /u/emozilla
    """

    def __init__(self, dim: int, base: float, scaling_factor: float,
                 max_position_embeddings: float):
        self.scaling_factor = scaling_factor
        self.max_position_embeddings = max_position_embeddings
        super().__init__(dim, base)

    def _get_inv_freq(self, x: torch.Tensor, seq_len: int):
        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) -
                (self.scaling_factor - 1))**(self.dim / (self.dim - 2))
            inv_freq = (1.0 /
                        (base**(torch.arange(0, self.dim, 2) / self.dim))).to(x)
        else:
            inv_freq = (
                1.0 /
                (self.base**(torch.arange(0, self.dim, 2) / self.dim))).to(x)
        return inv_freq


def rotate_half(x: torch.Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.gpt_neox.modeling_gpt_neox.apply_rotary_pos_emb
def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor,
                         sin: torch.Tensor, position_ids: torch.Tensor):
    cos = cos[position_ids].unsqueeze(
        1)  # [seq_len, dim] -> [batch_size, 1, seq_len, head_dim]
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
