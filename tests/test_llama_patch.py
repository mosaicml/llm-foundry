# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
import transformers
from composer.utils import reproducibility
from transformers.models.llama.modeling_llama import LlamaAttention

from llmfoundry.models.layers.llama_attention_monkeypatch import (
    llama_attention_patch_torch, llama_attention_patch_triton)


@pytest.mark.parametrize('patch_fn_name', ['torch', 'triton'])
@pytest.mark.parametrize('explicit_mask', [True, False])
@pytest.mark.parametrize(
    'model_name', ['meta-llama/Llama-2-7b-hf', 'meta-llama/Llama-2-70b-hf'])
@pytest.mark.gpu
def test_patch_equivalence(patch_fn_name: str, explicit_mask: bool,
                           model_name: str):
    if 'HUGGING_FACE_HUB_TOKEN' not in os.environ:
        pytest.skip(
            'The CI cluster does not have access to the Llama models, so skip this test.'
        )

    original_forward = LlamaAttention.forward

    device = 'cuda:0'
    sequence_length = 4096
    model_dim = 4096 if '7b' in model_name else 8192
    batch_size = 2
    if patch_fn_name == 'torch':
        patch_fn = llama_attention_patch_torch
        dtype = torch.float32
        atol = 0.0
        rtol = 0.0
    elif patch_fn_name == 'triton':
        # the huggingface implementation of llama performs the softmax in fp32
        # this can result in fairly large differences for the triton implementation
        # but the torch implementation produces the exact same output so we can confirm
        # the implementation is correct
        patch_fn = llama_attention_patch_triton
        dtype = torch.bfloat16
        atol = 1e-2
        rtol = 1e-2
    else:
        raise ValueError(f'Unknown patch_fn_name: {patch_fn_name}')

    llama_config = transformers.AutoConfig.from_pretrained(model_name,
                                                           use_auth_token=True)

    reproducibility.seed_all(42)
    attention = LlamaAttention(config=llama_config,)
    attention.to(dtype=dtype, device=device)

    rng = torch.Generator(device=device).manual_seed(42)
    hidden_states = torch.randn(batch_size,
                                sequence_length,
                                model_dim,
                                generator=rng,
                                dtype=dtype,
                                device=device)
    causal_mask = torch.full((sequence_length, sequence_length),
                             torch.finfo(torch.float32).min,
                             device=device)
    causal_mask = causal_mask.triu(diagonal=1)
    causal_mask = causal_mask[None,
                              None, :, :].expand(batch_size, 1, sequence_length,
                                                 sequence_length)
    attn_output, _, _ = attention(
        hidden_states=hidden_states,
        attention_mask=causal_mask if explicit_mask else None,
        position_ids=None,
        past_key_value=None,
        use_cache=False,
    )

    reproducibility.seed_all(42)
    LlamaAttention.forward = patch_fn
    attention = LlamaAttention(config=llama_config,)
    attention.to(dtype=dtype, device=device)
    new_output, _, _ = attention(
        hidden_states=hidden_states,
        attention_mask=causal_mask if explicit_mask else None,
        position_ids=None,
        past_key_value=None,
        use_cache=False,
    )

    # Reset the forward function so patches don't persist
    LlamaAttention.forward = original_forward

    assert torch.allclose(attn_output, new_output, atol=atol, rtol=rtol)
