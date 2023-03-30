# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from composer.utils import reproducibility
from omegaconf import OmegaConf as om


def allclose_helper(t0, t1, rtol=1e-2, atol=1e-2):
    return torch.allclose(t0, t1, rtol=rtol, atol=atol)


@pytest.mark.gpu
@pytest.mark.parametrize('attn_impl_0', ['flash', 'triton', 'torch'])
@pytest.mark.parametrize('attn_impl_1', ['flash', 'triton', 'torch'])
@pytest.mark.parametrize('attn_clip_qkv', [True, False])
@pytest.mark.parametrize('attn_qk_ln', [True, False])
@pytest.mark.parametrize('alibi', [True, False])
def test_attn_impl(attn_impl_0,
                   attn_impl_1,
                   attn_clip_qkv,
                   attn_qk_ln,
                   alibi,
                   device='cuda'):
    """Compare all attn impl with each other.

    Includes testing with and without attn_clip_qkv, attn_qk_ln, and alibi.
    """
    from examples.llm.src.models.layers import attention  # type: ignore

    if alibi and (attn_impl_0 == 'flash' or attn_impl_1 == 'flash'):
        pytest.xfail('flash attn does not support alibi')

    reproducibility.seed_all(7)

    cfg = om.create({
        'attn_impl': 'flash',
        'd_model': 256,
        'n_heads': 2,
        'attn_pdrop': 0,
        'attn_clip_qkv': attn_clip_qkv,
        'attn_qk_ln': attn_qk_ln,
    })

    n, s, f = 2, 16, cfg.d_model

    cfg.attn_impl = attn_impl_0
    attn0 = attention.MultiheadAttention(**cfg).to(device)
    cfg.attn_impl = attn_impl_1
    attn1 = attention.MultiheadAttention(**cfg).to(device)

    attn1.load_state_dict(attn0.state_dict())

    attention_mask = torch.ones(n, s).to(device).bool()

    def gen_bias(attn_impl):
        causal = True
        attn_bias = None
        bs = attention.attn_bias_shape(attn_impl,
                                       cfg.n_heads,
                                       s,
                                       alibi,
                                       prefix_lm=False,
                                       use_sequence_id=False,
                                       causal=causal)
        if bs is not None:
            attn_bias = torch.zeros(*bs, device=device)
            attn_bias = attention.attn_bias(attn_impl,
                                            attn_bias,
                                            cfg.n_heads,
                                            s,
                                            causal=causal,
                                            alibi=alibi,
                                            alibi_bias_max=8)

        return attn_bias

    x0 = torch.randn(n, s, f).to(device)
    x1 = x0.clone().detach()
    x0.requires_grad = True
    x1.requires_grad = True

    with torch.autocast(x0.device.type):
        attn_bias = gen_bias(attn0.attn_impl)
        y0, _, _ = attn0(x0,
                         past_key_value=None,
                         attn_bias=attn_bias,
                         attention_mask=attention_mask,
                         is_causal=True)
        attn_bias = gen_bias(attn1.attn_impl)
        y1, _, _ = attn1(x1,
                         past_key_value=None,
                         attn_bias=attn_bias,
                         attention_mask=attention_mask,
                         is_causal=True)
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
        assert allclose_helper(p, tp)
        assert allclose_helper(p.grad, tp.grad)

    assert allclose_helper(x0.grad, x1.grad)


@pytest.mark.gpu
@pytest.mark.parametrize('attn_impl', ['flash', 'triton', 'torch'])
def test_vs_mha(attn_impl, device='cuda'):
    """Compare diff attn_impl to torch.nn.MultiheadAttention."""
    from examples.llm.src.models.layers import attention  # type: ignore

    reproducibility.seed_all(17)

    cfg = om.create({
        'attn_impl': attn_impl,
        'd_model': 256,
        'n_heads': 2,
        'attn_pdrop': 0,
        'attn_clip_qkv': False,
        'attn_qk_ln': False,
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

    assert allclose_helper(y0, y1)

    assert allclose_helper(tmhsa.out_proj.bias.grad, mmhsa.out_proj.bias.grad)
    assert allclose_helper(tmhsa.out_proj.weight.grad,
                           mmhsa.out_proj.weight.grad)
    assert allclose_helper(tmhsa.in_proj_bias.grad, mmhsa.Wqkv.bias.grad)
    assert allclose_helper(tmhsa.in_proj_weight.grad, mmhsa.Wqkv.weight.grad)

    assert allclose_helper(x0.grad, x1.grad)
