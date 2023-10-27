# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from composer.core.precision import get_precision_context

from llmfoundry.models.layers.attention import is_flash_v2_installed

if is_flash_v2_installed():
    from flash_attn.layers.rotary import RotaryEmbedding as DAILRotaryEmbedding
from omegaconf import OmegaConf as om
from transformers.models.llama.modeling_llama import \
    LlamaDynamicNTKScalingRotaryEmbedding as HFDynamicNTKScalingRotaryEmbedding
from transformers.models.llama.modeling_llama import \
    LlamaLinearScalingRotaryEmbedding as HFLinearScalingRotaryEmbedding
from transformers.models.llama.modeling_llama import \
    LlamaRotaryEmbedding as HFRotaryEmbedding


def gen_rotary_embedding(rope_head_dim: int, pos_emb_config: dict,
                         max_seq_len: int):
    if pos_emb_config['rope_imp'] == 'dail':
        return DAILRotaryEmbedding(
            dim=rope_head_dim,
            base=pos_emb_config['rope_theta'],
            interleaved=False,
            scale_base=pos_emb_config['rope_dail_config']['xpos_scale_base'] if
            (pos_emb_config['rope_dail_config']['type'] == 'xpos') else None,
            pos_idx_in_fp32=pos_emb_config['rope_dail_config']
            ['pos_idx_in_fp32'],
            device=
            'cpu',  # FSDP does not materialize modules with meta buffers, hence device is set to cpu
        )
    elif pos_emb_config['rope_imp'] == 'hf':
        if pos_emb_config['rope_hf_config']['type'] == 'no_scaling':
            return HFRotaryEmbedding(
                rope_head_dim,
                max_position_embeddings=max_seq_len,
                base=pos_emb_config['rope_theta'],
                device=
                'cpu'  # FSDP does not materialize modules with meta buffers, hence device is set to cpu
            )
        elif pos_emb_config['rope_hf_config']['type'] == 'linear':
            return HFLinearScalingRotaryEmbedding(
                rope_head_dim,
                max_position_embeddings=max_seq_len,
                base=pos_emb_config['rope_theta'],
                scaling_factor=pos_emb_config['rope_hf_config']['factor'],
                device=
                'cpu'  # FSDP does not materialize modules with meta buffers, hence device is set to cpu
            )
        elif pos_emb_config['rope_hf_config']['type'] == 'dynamic':
            return HFDynamicNTKScalingRotaryEmbedding(
                rope_head_dim,
                max_position_embeddings=max_seq_len,
                base=pos_emb_config['rope_theta'],
                scaling_factor=pos_emb_config['rope_hf_config']['factor'],
                device=
                'cpu'  # FSDP does not materialize modules with meta buffers, hence device is set to cpu
            )
        else:
            raise ValueError(
                f'Invalid scaling type: {pos_emb_config["rope_hf_config"]["type"]}'
            )
    else:
        raise ValueError(f'Invalid rope_imp: {pos_emb_config["rope_imp"]}')


@pytest.mark.gpu
@pytest.mark.parametrize('clip_qkv', [True, False])
@pytest.mark.parametrize('qk_ln', [True, False])
@pytest.mark.parametrize(
    'attn_type',
    ['multihead_attention', 'multiquery_attention', 'grouped_query_attention'])
@pytest.mark.parametrize('seq_len', [1, 233, 2048])
def test_rope_dail_vs_hf(clip_qkv: bool,
                         qk_ln: bool,
                         attn_type: str,
                         seq_len: int,
                         device: str = 'cuda'):
    # compare rope rotations for the dail vs hf implementations
    if not is_flash_v2_installed():
        pytest.skip('dail implementation of rope requires flash attention 2.')

    from llmfoundry.models.layers import attention

    cfg = om.create({
        'attn_impl': 'flash',
        'd_model': 128,
        'n_heads': 4,
        'attn_pdrop': 0,
        'clip_qkv': clip_qkv,
        'qk_ln': qk_ln,
    })

    batch_size = 2
    assert cfg.d_model % cfg.n_heads == 0
    if attn_type == 'grouped_query_attention':
        cfg.kv_n_heads = 2

    attn0 = attention.ATTN_CLASS_REGISTRY[attn_type](**cfg).to(device)
    attn1 = attention.ATTN_CLASS_REGISTRY[attn_type](**cfg).to(device)

    attn1.load_state_dict(attn0.state_dict())
    x0 = torch.randn(batch_size, seq_len, cfg.d_model).to(device)
    x1 = x0.clone().detach()
    x0.requires_grad = True
    x1.requires_grad = True
    attention_mask = torch.ones(batch_size, seq_len).to(device).bool()

    with get_precision_context('amp_bf16'):
        dail_rope_config = {
            'rope_theta': 10000,
            'rope_imp': 'dail',
            'rope_dail_config': {
                'type': 'original',
                'pos_idx_in_fp32': True,
                'xpos_scale_base': 512,
            }
        }
        hf_rope_config = {
            'rope_theta': 10000,
            'rope_imp': 'hf',
            'rope_hf_config': {
                'type': 'no_scaling',
                'factor': 1.0,
            }
        }

        dail_rope = gen_rotary_embedding(rope_head_dim=cfg.d_model //
                                         cfg.n_heads,
                                         pos_emb_config=dail_rope_config,
                                         max_seq_len=seq_len).to('cuda')
        dail_rope_w_meta_info = {
            'imp': 'dail',
            'rotary_emb': dail_rope,
            'offset_info': 0,
            'seq_len': seq_len,
        }

        hf_rope = gen_rotary_embedding(rope_head_dim=cfg.d_model // cfg.n_heads,
                                       pos_emb_config=hf_rope_config,
                                       max_seq_len=seq_len).to('cuda')
        pos = torch.arange(seq_len).unsqueeze(0).to(device='cuda')
        # adjust the position indices to account for padding tokens
        pos = torch.clamp(
            pos - torch.cumsum((~attention_mask).to(torch.int32), dim=1),
            min=0,
        )
        hf_rope_w_meta_info = {
            'imp': 'hf',
            'rotary_emb': hf_rope,
            'offset_info': pos,
            'seq_len': seq_len,
        }

        y0, _, _ = attn0(x0,
                         past_key_value=None,
                         attn_bias=None,
                         attention_mask=attention_mask,
                         rotary_emb_w_meta_info=dail_rope_w_meta_info,
                         is_causal=True)

        y1, _, _ = attn1(x1,
                         past_key_value=None,
                         attn_bias=None,
                         attention_mask=attention_mask,
                         rotary_emb_w_meta_info=hf_rope_w_meta_info,
                         is_causal=True)

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
