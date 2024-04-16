# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from composer.core.precision import get_precision_context
from omegaconf import OmegaConf as om

from llmfoundry.models.layers.attention import is_flash_v2_installed
from llmfoundry.models.layers.layer_builders import build_attention_layer
from llmfoundry.models.mpt.modeling_mpt import (gen_flash_attn_padding_info,
                                                gen_rotary_embedding)


@pytest.mark.gpu
@pytest.mark.parametrize(
    'attn_type',
    ['multihead_attention', 'multiquery_attention', 'grouped_query_attention'])
@pytest.mark.parametrize('seq_len', [1, 233, 2048])
def test_rope_dail_vs_hf(attn_type: str, seq_len: int, device: str = 'cuda'):
    # compare rope rotations for the dail vs hf implementations
    if not is_flash_v2_installed():
        pytest.skip('dail implementation of rope requires flash attention 2.')

    cfg = om.create({
        'attn_impl': 'flash',
        'd_model': 128,
        'n_heads': 4,
        'attn_pdrop': 0,
        'clip_qkv': True,
        'qk_ln': False,
    })

    batch_size = 2
    assert cfg.d_model % cfg.n_heads == 0
    if attn_type == 'grouped_query_attention':
        cfg.kv_n_heads = 2

    attn0 = build_attention_layer(
        name=attn_type,
        attn_kwargs=om.to_container(
            cfg),  # type: ignore (to_container return broad type)
    ).to(device)
    attn1 = build_attention_layer(
        name=attn_type,
        attn_kwargs=om.to_container(
            cfg),  # type: ignore (to_container return broad type)
    ).to(device)

    attn1.load_state_dict(attn0.state_dict())
    x0 = torch.randn(batch_size, seq_len, cfg.d_model).to(device)
    x1 = x0.clone().detach()
    x0.requires_grad = True
    x1.requires_grad = True
    attention_mask = torch.ones(batch_size, seq_len).to(device).bool()

    with get_precision_context('amp_bf16'):
        dail_rope_config = {
            'rope_theta': 10000,
            'rope_impl': 'dail',
            'rope_dail_config': {
                'type': 'original',
                'pos_idx_in_fp32': True,
                'xpos_scale_base': 512,
            }
        }
        hf_rope_config = {
            'rope_theta': 10000,
            'rope_impl': 'hf',
            'rope_hf_config': {
                'type': 'no_scaling',
                'factor': 1.0,
            }
        }

        dail_rope = gen_rotary_embedding(
            rope_head_dim=cfg.d_model // cfg.n_heads,
            rope_impl=dail_rope_config['rope_impl'],
            rope_theta=dail_rope_config['rope_theta'],
            rope_dail_config=dail_rope_config['rope_dail_config'],
            rope_hf_config={},
            max_seq_len=seq_len).to('cuda')
        dail_rope_w_meta_info = {
            'impl': 'dail',
            'rotary_emb': dail_rope,
            'offset_info': 0,
            'seq_len': seq_len,
        }

        hf_rope = gen_rotary_embedding(
            rope_head_dim=cfg.d_model // cfg.n_heads,
            rope_impl=hf_rope_config['rope_impl'],
            rope_theta=hf_rope_config['rope_theta'],
            rope_dail_config={},
            rope_hf_config=hf_rope_config['rope_hf_config'],
            max_seq_len=seq_len).to('cuda')
        pos = torch.arange(seq_len).unsqueeze(0).to(device='cuda')
        # adjust the position indices to account for padding tokens
        pos = torch.clamp(
            pos - torch.cumsum((~attention_mask).to(torch.int32), dim=1),
            min=0,
        )
        hf_rope_w_meta_info = {
            'impl': 'hf',
            'rotary_emb': hf_rope,
            'offset_info': pos,
            'seq_len': seq_len,
        }

        y0, _, _ = attn0(x0,
                         past_key_value=None,
                         attn_bias=None,
                         attention_mask=attention_mask,
                         rotary_emb_w_meta_info=dail_rope_w_meta_info,
                         is_causal=True,
                         flash_attn_padding_info=gen_flash_attn_padding_info(
                             batch_size, seq_len, 0, torch.device(device), None,
                             attention_mask))

        y1, _, _ = attn1(x1,
                         past_key_value=None,
                         attn_bias=None,
                         attention_mask=attention_mask,
                         rotary_emb_w_meta_info=hf_rope_w_meta_info,
                         is_causal=True,
                         flash_attn_padding_info=gen_flash_attn_padding_info(
                             batch_size, seq_len, 0, torch.device(device), None,
                             attention_mask))

        y0 *= attention_mask.unsqueeze(-1)
        y1 *= attention_mask.unsqueeze(-1)

        loss0 = y0.sum()
        loss1 = y1.sum()

    loss0.backward()
    loss1.backward()

    torch.testing.assert_close(y0, y1, rtol=1e-2, atol=1e-2)

    torch_name_param_map = {n: p for n, p in attn1.named_parameters()}
    for n, p in attn0.named_parameters():
        tp = torch_name_param_map[n]
        assert p.grad is not None
        assert tp.grad is not None
        torch.testing.assert_close(p, tp, rtol=1e-2, atol=1e-2)
        # Relaxed to a l2-norm based check.
        assert torch.norm(tp.grad - p.grad) <= 1e-2 + 1e-2 * torch.norm(p.grad)

    assert x0.grad is not None
    assert x1.grad is not None
    # Relaxed to a l2-norm based check.
    assert torch.norm(x0.grad - x1.grad) <= 1e-2 + 1e-2 * torch.norm(x0.grad)
