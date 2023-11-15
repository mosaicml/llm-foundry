# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

# NOTE: This is a temporary test file to test that we do not need to repeat kv tensors for gqa on our end for flash_attn_2.
# Will be deleted before merging to main (or not, if people think this should be added).
# To run, simply run `python test_dont_repeat_kv_for_gqa.py`.
import math

import torch

from llmfoundry.models.layers.attention import (flash_attn_fn,
                                                is_flash_v2_installed)

assert is_flash_v2_installed()

d = 128
n_heads = 4
kv_n_heads = 2
seqlen_1 = 6
bsz = 2

query_1 = torch.randn(bsz, seqlen_1, n_heads * d).to(torch.bfloat16).cuda()
query_1.requires_grad = True
key_1 = torch.randn(bsz, seqlen_1, kv_n_heads * d).to(torch.bfloat16).cuda()
key_1.requires_grad = True
value_1 = torch.randn(bsz, seqlen_1, kv_n_heads * d).to(torch.bfloat16).cuda()
value_1.requires_grad = True

output_1, _, _ = flash_attn_fn(query=query_1,
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
                               multiquery=False,
                               key_attention_mask_in_length=None,
                               query_attention_mask_in_length=None,
                               should_repeat_kv_for_gqa=True)

output_1.sum().backward()

query_2 = query_1.detach().clone()
query_2.requires_grad = True
key_2 = key_1.detach().clone()
key_2.requires_grad = True
value_2 = value_1.detach().clone()
value_2.requires_grad = True

output_2, _, _ = flash_attn_fn(query=query_2,
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
                               multiquery=False,
                               key_attention_mask_in_length=None,
                               query_attention_mask_in_length=None,
                               should_repeat_kv_for_gqa=False)

output_2.sum().backward()
assert torch.allclose(output_1, output_2)
assert torch.allclose(query_1.grad, query_2.grad)
assert torch.allclose(key_1.grad, key_2.grad)
assert torch.allclose(value_1.grad, value_2.grad)
