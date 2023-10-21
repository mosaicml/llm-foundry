# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

# This file is copied and modified from
# https://github.com/huggingface/transformers/blob/fe3c8ab1af558b95f67f5fafc0c55f09fd2b09db/src/transformers/models/llama/modeling_llama.py
# See the clearly denoted code blocks for the main modifications (there are a few others like type ignores, and error messages)

import logging
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaAttention

from llmfoundry.models.layers.attention import (
    scaled_multihead_dot_product_attention, triton_flash_attn_fn)

log = logging.getLogger(__name__)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Equivalent of torch.repeat_interleave(x, dim=1,

    repeats=n_rep).

    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim) to
    (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :,
                                  None, :, :].expand(batch, num_key_value_heads,
                                                     n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen,
                                 head_dim)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def get_llama_attention_patch_fn(patch_fn_name: str = 'torch') -> Callable:
    if patch_fn_name == 'torch':
        return llama_attention_patch_torch
    elif patch_fn_name == 'triton':
        return llama_attention_patch_triton
    else:
        raise ValueError(
            f'Unrecognized llama attention patch function: {patch_fn_name}')


def llama_attention_patch_torch(
    self: LlamaAttention,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if use_cache:
        raise NotImplementedError(
            'use_cache is not yet supported when patching Llama attention.')

    bsz, q_len, _ = hidden_states.size()

    if self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads *
                             self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp,
            dim=0)
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [
            F.linear(hidden_states, query_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [
            F.linear(hidden_states, key_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [
            F.linear(hidden_states, value_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        value_states = torch.cat(value_states, dim=-1)
    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads,
                                     self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads,
                                 self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads,
                                     self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                    cos, sin, position_ids)

    ### MAIN MODIFICATIONS START HERE ###
    query_states = query_states.transpose(1, 2).view(
        bsz, q_len, self.num_heads * self.head_dim)
    key_states = key_states.transpose(1, 2).view(
        bsz, q_len, self.num_key_value_heads * self.head_dim)
    value_states = value_states.transpose(1, 2).view(
        bsz, q_len, self.num_key_value_heads * self.head_dim)

    attn_output, attn_weights, _ = scaled_multihead_dot_product_attention(
        query=query_states,
        key=key_states,
        value=value_states,
        n_heads=self.num_heads,
        kv_n_heads=self.num_key_value_heads,
        past_key_value=None,
        softmax_scale=None,
        attn_bias=attention_mask,
        key_padding_mask=None,
        is_causal=False,  # The causal mask is propagated from LLamaForCausalLM
        dropout_p=0,
        training=self.training,
        needs_weights=False,
    )
    ### MAIN MODIFICATIONS END HERE ###

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size //
                                        self.config.pretraining_tp,
                                        dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size //
                                                 self.config.pretraining_tp,
                                                 dim=1)
        attn_output = sum([
            F.linear(attn_output[i], o_proj_slices[i])
            for i in range(self.config.pretraining_tp)
        ])
    else:
        attn_output = self.o_proj(attn_output)

    assert isinstance(attn_output, torch.Tensor)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, None


def llama_attention_patch_triton(
    self: LlamaAttention,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if use_cache:
        raise NotImplementedError(
            'use_cache is not yet supported when patching Llama attention.')
    # output_attentions is not support for triton attention
    if output_attentions:
        raise NotImplementedError(
            'output_attentions is not supported when patching Llama attention with triton attention.'
        )

    bsz, q_len, _ = hidden_states.size()

    if self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads *
                             self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp,
            dim=0)
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [
            F.linear(hidden_states, query_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [
            F.linear(hidden_states, key_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [
            F.linear(hidden_states, value_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        value_states = torch.cat(value_states, dim=-1)
    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads,
                                     self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads,
                                 self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads,
                                     self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                    cos, sin, position_ids)

    ### MAIN MODIFICATIONS START HERE ###
    query_states = query_states.transpose(1, 2).view(
        bsz, q_len, self.num_heads * self.head_dim)
    key_states = key_states.transpose(1, 2).view(
        bsz, q_len, self.num_key_value_heads * self.head_dim)
    value_states = value_states.transpose(1, 2).view(
        bsz, q_len, self.num_key_value_heads * self.head_dim)

    attn_output, _, _ = triton_flash_attn_fn(
        query=query_states,
        key=key_states,
        value=value_states,
        n_heads=self.num_heads,
        kv_n_heads=self.num_key_value_heads,
        past_key_value=None,
        softmax_scale=None,
        attn_bias=attention_mask,
        key_padding_mask=None,
        is_causal=False,  # The causal mask is propagated from LLamaForCausalLM
        dropout_p=0,
        training=self.training,
        needs_weights=False,
    )
    ### MAIN MODIFICATIONS END HERE ###

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size //
                                        self.config.pretraining_tp,
                                        dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size //
                                                 self.config.pretraining_tp,
                                                 dim=1)
        attn_output = sum([
            F.linear(attn_output[i], o_proj_slices[i])
            for i in range(self.config.pretraining_tp)
        ])
    else:
        attn_output = self.o_proj(attn_output)

    assert isinstance(attn_output, torch.Tensor)

    return attn_output, None, None
