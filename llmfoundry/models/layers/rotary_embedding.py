# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from typing import Union

import torch


# Code modified from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
class RotaryEmbedding(torch.nn.Module):

    def __init__(self, dim: int, max_position_embeddings: int, base: int,
                 device: Union[str, torch.device]):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base**(
            torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        self.max_seq_len_cached = self.max_position_embeddings
        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,  # type: ignore
            dtype=torch.get_default_dtype())  # type: ignore

    def _set_cos_sin_cache(self, seq_len: int, device: Union[str, torch.device],
                           dtype: torch.dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached,
                         device=device,
                         dtype=self.inv_freq.dtype)  # type: ignore

        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached',
                             emb.cos().to(dtype),
                             persistent=False)
        self.register_buffer('sin_cached',
                             emb.sin().to(dtype),
                             persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len,
                                    device=x.device,
                                    dtype=x.dtype)  # type: ignore

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),  # type: ignore
            self.sin_cached[:seq_len].to(dtype=x.dtype),  # type: ignore
        )


class LinearScalingRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with linear scaling.

    Credits to the Reddit user /u/kaiokendev
    """

    def __init__(self, dim: int, max_position_embeddings: int, base: int,
                 scaling_factor: float, device: Union[str, torch.device]):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len: int, device: Union[str, torch.device],
                           dtype: torch.dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached,
                         device=device,
                         dtype=self.inv_freq.dtype)  # type: ignore
        t = t / self.scaling_factor

        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached',
                             emb.cos().to(dtype),
                             persistent=False)
        self.register_buffer('sin_cached',
                             emb.sin().to(dtype),
                             persistent=False)


class DynamicNTKScalingRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with Dynamic NTK scaling.

    Credits to the Reddit users /u/bloc97 and /u/emozilla
    """

    def __init__(self, dim: int, max_position_embeddings: int, base: int,
                 scaling_factor: float, device: Union[str, torch.device]):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len: int, device: Union[str, torch.device],
                           dtype: torch.dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) -
                (self.scaling_factor - 1))**(self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base**(
                torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer('inv_freq', inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached,
                         device=device,
                         dtype=self.inv_freq.dtype)  # type: ignore

        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached',
                             emb.cos().to(dtype),
                             persistent=False)
        self.register_buffer('sin_cached',
                             emb.sin().to(dtype),
                             persistent=False)


def rotate_half(x: torch.Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.gpt_neox.modeling_gpt_neox.apply_rotary_pos_emb
def apply_rotary_pos_emb(q: torch.Tensor,
                         k: torch.Tensor,
                         cos: torch.Tensor,
                         sin: torch.Tensor,
                         position_ids: torch.Tensor,
                         dim_heads_index: int = 1):
    cos = cos[position_ids].unsqueeze(dim_heads_index)
    sin = sin[position_ids].unsqueeze(dim_heads_index)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
