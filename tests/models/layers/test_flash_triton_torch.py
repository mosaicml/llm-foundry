# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from omegaconf import OmegaConf as om

from llmfoundry.models.layers import attention
from llmfoundry.models.layers.attention import gen_slopes, is_flash_v2_installed
from llmfoundry.models.mpt.modeling_mpt import (apply_sequence_id,
                                                gen_attention_mask_in_length,
                                                gen_rotary_embedding)


def allclose_helper(t0: torch.Tensor,
                    t1: torch.Tensor,
                    rtol: float = 1e-2,
                    atol: float = 1e-2):
    return torch.allclose(t0, t1, rtol=rtol, atol=atol)


@pytest.mark.gpu
@pytest.mark.parametrize('attn_impl_0,attn_impl_1', [
    ('flash', 'triton'),
    ('flash', 'torch'),
    ('triton', 'torch'),
])
@pytest.mark.parametrize('clip_qkv', [True, False])
@pytest.mark.parametrize('qk_ln', [True, False])
@pytest.mark.parametrize('pos_emb_config', [{
    'alibi': False,
    'rope': False
}, {
    'alibi': True,
    'rope': False
}, {
    'alibi': False,
    'rope': True,
    'rope_theta': 10000,
    'rope_impl': 'dail',
    'rope_dail_config': {
        'type': 'original',
        'pos_idx_in_fp32': True,
        'xpos_scale_base': 512,
    },
}, {
    'alibi': False,
    'rope': True,
    'rope_theta': 10000,
    'rope_impl': 'hf',
    'rope_hf_config': {
        'type': 'no_scaling',
        'factor': 1.0,
    },
}])
@pytest.mark.parametrize(
    'attn_type',
    ['multihead_attention', 'multiquery_attention', 'grouped_query_attention'])
@pytest.mark.parametrize('attn_uses_sequence_id', [True, False])
@pytest.mark.parametrize('pad_attention_mask', [True, False])
def test_attn_impl(attn_impl_0: str,
                   attn_impl_1: str,
                   clip_qkv: bool,
                   qk_ln: bool,
                   pos_emb_config: dict,
                   attn_type: str,
                   attn_uses_sequence_id: bool,
                   pad_attention_mask: bool,
                   device: str = 'cuda'):
    """Compare all attn impl with each other.

    Includes testing with and without attn_clip_qkv, attn_qk_ln, alibi, and
    rope.
    """
    alibi = pos_emb_config['alibi']
    rope = pos_emb_config['rope']
    if alibi and (attn_impl_0 == 'flash' or attn_impl_1 == 'flash'
                 ) and not is_flash_v2_installed(v2_version='v2.4.2'):
        pytest.skip('flash attention below v2.4.2 does not support alibi.')
    if rope and (pos_emb_config['rope_impl']
                 == 'dail') and (not is_flash_v2_installed()):
        pytest.skip('dail implementation of rope requires flash attention 2.')

    if attn_uses_sequence_id and (
            attn_impl_0 == 'flash' or attn_impl_1
            == 'flash') and (not is_flash_v2_installed(v2_version='v2.1.2')):
        pytest.skip(
            'Using sequence id with flash attention requires flash attention v2.1.2 or higher.'
        )

    if not (alibi or rope) and attn_uses_sequence_id:
        pytest.skip('attn_uses_sequence_id requires alibi or rope.')

    cfg = om.create({
        'attn_impl': 'flash',
        'd_model': 64,
        'n_heads': 4,
        'attn_pdrop': 0,
        'clip_qkv': clip_qkv,
        'qk_ln': qk_ln,
    })

    n, s, f = 2, 4, cfg.d_model
    assert cfg.d_model % cfg.n_heads == 0
    if attn_type == 'grouped_query_attention':
        cfg.kv_n_heads = 2

    sequence_id = None
    if attn_uses_sequence_id:
        assert n == 2
        assert s >= 4
        sequence_id = torch.LongTensor([[0] * 2 + [1] * (s - 2),
                                        [0] * 4 + [1] * (s - 4)
                                       ]).to(device=device)

    cfg.attn_impl = attn_impl_0
    attn0 = attention.ATTN_CLASS_REGISTRY[attn_type](**cfg).to(device)
    cfg.attn_impl = attn_impl_1
    attn1 = attention.ATTN_CLASS_REGISTRY[attn_type](**cfg).to(device)

    attn1.load_state_dict(attn0.state_dict())

    attention_mask = torch.ones(n, s).to(device).bool()

    if pad_attention_mask:
        # zero out the last third of the attention mask
        # to simulate padding
        attention_mask[:, -s // 3:] = 0

    def gen_bias(attn_impl: str):
        causal = True
        attn_bias = None
        bs = attention.attn_bias_shape(attn_impl,
                                       cfg.n_heads,
                                       s,
                                       alibi,
                                       prefix_lm=False,
                                       use_sequence_id=attn_uses_sequence_id,
                                       causal=causal)
        if bs is not None:
            attn_bias = torch.zeros(*bs, device=device)
            attn_bias = attention.build_attn_bias(
                attn_impl,
                attn_bias,
                cfg.n_heads,
                s,
                causal=causal,
                alibi=alibi,
                alibi_bias_max=8,
            )
        if attn_impl != 'flash' and attn_uses_sequence_id and sequence_id is not None:
            assert isinstance(attn_bias, torch.Tensor)  # pyright
            attn_bias = apply_sequence_id(
                attn_bias,
                sequence_id,  # type: ignore
                s)

        return attn_bias

    attention_mask_in_length_0 = gen_attention_mask_in_length(
        sequence_id=sequence_id,
        S=s,
        attn_uses_sequence_id=attn_uses_sequence_id,
        attn_impl=attn_impl_0,
        attention_mask=attention_mask)
    attention_mask_in_length_1 = gen_attention_mask_in_length(
        sequence_id=sequence_id,
        S=s,
        attn_uses_sequence_id=attn_uses_sequence_id,
        attn_impl=attn_impl_1,
        attention_mask=attention_mask)

    x0 = torch.randn(n, s, f).to(device)
    x1 = x0.clone().detach()
    x0.requires_grad = True
    x1.requires_grad = True

    with torch.autocast(x0.device.type):
        attn_bias_0 = gen_bias(attn_impl_0)
        alibi_slopes_0 = None
        if alibi and attn_impl_0 == 'flash':
            alibi_slopes_0 = gen_slopes(n_heads=cfg.n_heads,
                                        alibi_bias_max=8,
                                        device=torch.device(device),
                                        return_1d=True)
        rotary_emb_w_meta_info = None
        if rope:
            rotary_embedding = gen_rotary_embedding(
                rope_head_dim=cfg.d_model // cfg.n_heads,
                rope_impl=pos_emb_config['rope_impl'],
                rope_theta=pos_emb_config['rope_theta'],
                rope_dail_config=pos_emb_config.get('rope_dail_config', {}),
                rope_hf_config=pos_emb_config.get('rope_hf_config', {}),
                max_seq_len=s).to(device)
            pos = torch.arange(s).unsqueeze(0).to(device=device)
            # adjust the position indices to account for padding tokens
            pos = torch.clamp(
                pos - torch.cumsum((~attention_mask).to(torch.int32), dim=1),
                min=0,
            )
            rotary_emb_w_meta_info = {
                'impl':
                    pos_emb_config['rope_impl'],
                'rotary_emb':
                    rotary_embedding,
                'offset_info':
                    pos if (pos_emb_config['rope_impl'] == 'hf') else 0,
                'seq_len':
                    s,
            }

        y0, _, _ = attn0(x0,
                         past_key_value=None,
                         attn_bias=attn_bias_0,
                         attention_mask=attention_mask,
                         rotary_emb_w_meta_info=rotary_emb_w_meta_info,
                         is_causal=True,
                         attention_mask_in_length=attention_mask_in_length_0,
                         alibi_slopes=alibi_slopes_0)
        attn_bias_1 = gen_bias(attn_impl_1)
        alibi_slopes_1 = None
        if alibi and attn_impl_1 == 'flash':
            alibi_slopes_1 = gen_slopes(n_heads=cfg.n_heads,
                                        alibi_bias_max=8,
                                        device=torch.device(device),
                                        return_1d=True)
        y1, _, _ = attn1(x1,
                         past_key_value=None,
                         attn_bias=attn_bias_1,
                         attention_mask=attention_mask,
                         rotary_emb_w_meta_info=rotary_emb_w_meta_info,
                         is_causal=True,
                         attention_mask_in_length=attention_mask_in_length_1,
                         alibi_slopes=alibi_slopes_1)
        y0 *= attention_mask.unsqueeze(-1)
        y1 *= attention_mask.unsqueeze(-1)

        loss0 = y0.sum()
        loss1 = y1.sum()

    loss0.backward()
    loss1.backward()

    assert allclose_helper(y0, y1)

    torch_name_param_map = {n: p for n, p in attn1.named_parameters()}
    for n, p in attn0.named_parameters():
        tp = torch_name_param_map[n]
        assert p.grad is not None
        assert tp.grad is not None
        assert allclose_helper(p, tp)

        using_hf_rope = pos_emb_config['rope'] and pos_emb_config[
            'rope_impl'] == 'hf'

        # special case that (likely) fails due to numerics
        if clip_qkv and qk_ln and using_hf_rope and attn_type == 'grouped_query_attention':
            assert allclose_helper(p.grad, tp.grad, atol=2.e-2, rtol=2.e-2)
        else:
            assert allclose_helper(p.grad, tp.grad)

    assert x0.grad is not None
    assert x1.grad is not None
    assert allclose_helper(x0.grad, x1.grad)


@pytest.mark.gpu
@pytest.mark.parametrize('attn_impl', ['flash', 'triton', 'torch'])
def test_vs_mha(attn_impl: str, device: str = 'cuda'):
    """Compare diff attn_impl to torch.nn.MultiheadAttention."""
    from llmfoundry.models.layers import attention

    cfg = om.create({
        'attn_impl': attn_impl,
        'd_model': 64,
        'n_heads': 2,
        'attn_pdrop': 0,
        'clip_qkv': False,
        'qk_ln': False,
    })

    n, s, f = 2, 16, cfg.d_model

    mmhsa = attention.MultiheadAttention(**cfg).to(device)
    tmhsa = torch.nn.MultiheadAttention(
        embed_dim=cfg.d_model,
        num_heads=cfg.n_heads,
        dropout=cfg.attn_pdrop,
        bias=True,
        batch_first=True,
        device=device,
    )

    def gen_tca_mask():
        # generate causal mask for torch attn
        ms = (s, s)
        attn_mask = torch.empty(*ms).to(device)
        attn_mask.fill_(float('-inf'))
        attn_mask.masked_fill_(attn_mask.to(torch.bool).fill_(1).tril_(), 0.)
        return attn_mask

    # clone weights
    tmhsa.in_proj_weight.data = mmhsa.Wqkv.weight.data.clone().detach()
    tmhsa.in_proj_bias.data = mmhsa.Wqkv.bias.data.clone().detach()
    tmhsa.out_proj.weight.data = mmhsa.out_proj.weight.data.clone().detach()
    tmhsa.out_proj.bias.data = mmhsa.out_proj.bias.data.clone().detach()

    attention_mask = torch.ones(n, s).to(device).bool()
    x0 = torch.randn(n, s, f).to(device)
    x1 = x0.clone().detach()
    x0.requires_grad = True
    x1.requires_grad = True

    with torch.autocast(x0.device.type):
        y0, _, _ = mmhsa(x0,
                         past_key_value=None,
                         attn_bias=None,
                         attention_mask=attention_mask,
                         is_causal=True)
        y1, _ = tmhsa(x1,
                      x1,
                      x1,
                      attn_mask=gen_tca_mask(),
                      key_padding_mask=~attention_mask,
                      need_weights=True)
        y0 *= attention_mask.unsqueeze(-1)
        y1 *= attention_mask.unsqueeze(-1)

        loss0 = y0.sum()
        loss1 = y1.sum()

    loss0.backward()
    loss1.backward()

    assert y0 is not None
    assert y1 is not None
    assert tmhsa.out_proj.bias.grad is not None
    assert mmhsa.out_proj.bias.grad is not None
    assert tmhsa.out_proj.weight.grad is not None
    assert mmhsa.out_proj.weight.grad is not None
    assert tmhsa.in_proj_bias.grad is not None
    assert mmhsa.Wqkv.bias.grad is not None
    assert tmhsa.in_proj_weight.grad is not None
    assert mmhsa.Wqkv.weight.grad is not None
    assert x0.grad is not None
    assert x1.grad is not None

    assert allclose_helper(y0, y1)

    assert allclose_helper(tmhsa.out_proj.bias.grad, mmhsa.out_proj.bias.grad)
    assert allclose_helper(tmhsa.out_proj.weight.grad,
                           mmhsa.out_proj.weight.grad)
    assert allclose_helper(tmhsa.in_proj_bias.grad, mmhsa.Wqkv.bias.grad)
    assert allclose_helper(tmhsa.in_proj_weight.grad, mmhsa.Wqkv.weight.grad)

    assert allclose_helper(x0.grad, x1.grad)


@pytest.mark.gpu
@pytest.mark.parametrize('attn_impl', ['flash', 'triton', 'torch'])
@pytest.mark.parametrize('n_heads', [16, 8])
@pytest.mark.parametrize('kv_n_heads', [4, 2, 1])
def test_grouped_attention_heads(attn_impl: str,
                                 n_heads: int,
                                 kv_n_heads: int,
                                 device: str = 'cuda'):
    """Ensure grouped_query_attention runs w/ diff n_heads & kv_n_heads."""
    from llmfoundry.models.layers import attention

    cfg = om.create({
        'attn_impl': attn_impl,
        'd_model': 256,
        'n_heads': n_heads,
        'attn_pdrop': 0,
        'clip_qkv': False,
        'qk_ln': False,
        'kv_n_heads': kv_n_heads
    })

    n, s, f = 2, 4, cfg.d_model

    mmhsa = attention.GroupedQueryAttention(**cfg).to(device)

    attention_mask = torch.ones(n, s).to(device).bool()
    x0 = torch.randn(n, s, f).to(device)
    x0.requires_grad = True

    with torch.autocast(x0.device.type):
        y0, _, _ = mmhsa(x0,
                         past_key_value=None,
                         attn_bias=None,
                         attention_mask=attention_mask,
                         is_causal=True)
        y0 *= attention_mask.unsqueeze(-1)

        loss0 = y0.sum()

    loss0.backward()


def test_grouped_query_invalid_heads():
    """Check indivisble combinations of grouped_query_attention."""
    from llmfoundry.models.layers import attention

    cfg = om.create({
        'attn_impl': 'torch',
        'd_model': 256,
        'n_heads': 16,
        'attn_pdrop': 0,
        'clip_qkv': False,
        'qk_ln': False,
        'kv_n_heads': 3
    })

    expected_error = 'Each Q head should get the same number of KV heads, so n_heads must be divisible by kv_n_heads'

    with pytest.raises(ValueError, match=expected_error):
        _ = attention.GroupedQueryAttention(**cfg)

    cfg.kv_n_heads = 17

    expected_error = 'The number of KV heads should be less than or equal to Q heads'

    with pytest.raises(ValueError, match=expected_error):
        _ = attention.GroupedQueryAttention(**cfg)

    cfg.kv_n_heads = 0

    expected_error = 'kv_n_heads should be greater than zero'

    with pytest.raises(ValueError, match=expected_error):
        _ = attention.GroupedQueryAttention(**cfg)
