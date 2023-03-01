# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from composer.utils import reproducibility
from omegaconf import OmegaConf as om


def allclose_helper(t0, t1, rtol=1e-2, atol=1e-2):
    return torch.allclose(t0, t1, rtol=rtol, atol=atol)


@pytest.mark.gpu
def test_flash_torch(device='cuda'):
    from examples.llm.src.models.layers.attention import (  # type: ignore
        FlashCausalAttention, TorchCausalAttention)

    reproducibility.seed_all(7)

    cfg = om.create({
        'd_model': 256,
        'n_heads': 2,
        'attn_pdrop': 0,
    })

    n, s, f = 2, 16, cfg.d_model

    fca = FlashCausalAttention(cfg).to(device)
    tca = TorchCausalAttention(cfg).to(device)

    def gen_tca_mask():
        ms = TorchCausalAttention.mask_shape(cfg.n_heads, s, False)
        attn_mask = torch.empty(*ms).to(device)
        TorchCausalAttention.attn_mask_(attn_mask, cfg.n_heads, s)
        return attn_mask

    # clone weights
    tca.mhsa.in_proj_weight.data = fca.mhsa.Wqkv.weight.data.clone().detach()
    tca.mhsa.in_proj_bias.data = fca.mhsa.Wqkv.bias.data.clone().detach()
    tca.mhsa.out_proj.weight.data = fca.mhsa.out_proj.weight.data.clone(
    ).detach()
    tca.mhsa.out_proj.bias.data = fca.mhsa.out_proj.bias.data.clone().detach()

    key_padding_mask = torch.ones(n, s).to(device).bool()
    x0 = torch.randn(n, s, f).to(device)
    x1 = x0.clone().detach()
    x0.requires_grad = True
    x1.requires_grad = True

    with torch.autocast(x0.device.type):
        y0, _ = fca(x0, key_padding_mask, attn_mask=None)
        y1, _ = tca(x1, key_padding_mask, attn_mask=gen_tca_mask())
        y0 *= key_padding_mask.unsqueeze(-1)
        y1 *= key_padding_mask.unsqueeze(-1)

        loss0 = y0.sum()
        loss1 = y1.sum()

    loss0.backward()
    loss1.backward()

    assert allclose_helper(y0, y1)

    assert allclose_helper(tca.mhsa.out_proj.bias.grad,
                           fca.mhsa.out_proj.bias.grad)
    assert allclose_helper(tca.mhsa.out_proj.weight.grad,
                           fca.mhsa.out_proj.weight.grad)
    assert allclose_helper(tca.mhsa.in_proj_bias.grad, fca.mhsa.Wqkv.bias.grad)
    assert allclose_helper(tca.mhsa.in_proj_weight.grad,
                           fca.mhsa.Wqkv.weight.grad)

    assert allclose_helper(x0.grad, x1.grad)


@pytest.mark.gpu
@pytest.mark.parametrize('attn_clip_qkv,attn_qk_ln', [
    (False, False),
    (False, True),
    (True, False),
    (True, True),
])
def test_flash_triton(attn_clip_qkv, attn_qk_ln, device='cuda'):
    from examples.llm.src.models.layers.attention import (  # type: ignore
        FlashCausalAttention, TritonFlashCausalAttention)

    reproducibility.seed_all(7)

    cfg = om.create({
        'd_model': 256,
        'n_heads': 2,
        'attn_pdrop': 0,
        'attn_clip_qkv': attn_clip_qkv,
        'attn_qk_ln': attn_qk_ln,
    })

    n, s, f = 2, 16, cfg.d_model

    fca = FlashCausalAttention(cfg).to(device)
    tfca = TritonFlashCausalAttention(cfg).to(device)
    # clone weights
    if cfg.attn_qk_ln or cfg.attn_clip_qkv:
        tfca.Wqkv.weight.data = fca.W_qkv.weight.data.clone().detach()
        tfca.Wqkv.bias.data = fca.W_qkv.bias.data.clone().detach()
        tfca.out_proj.weight.data = fca.out_proj.weight.data.clone().detach()
        tfca.out_proj.bias.data = fca.out_proj.bias.data.clone().detach()
        if cfg.attn_qk_ln:
            tfca.q_ln.weight.data = fca.q_ln.weight.data.clone().detach()
            tfca.q_ln.bias.data = fca.q_ln.bias.data.clone().detach()
            tfca.k_ln.weight.data = fca.k_ln.weight.data.clone().detach()
            tfca.k_ln.bias.data = fca.k_ln.bias.data.clone().detach()
    else:
        tfca.mhsa.Wqkv.weight.data = fca.mhsa.Wqkv.weight.data.clone().detach()
        tfca.mhsa.Wqkv.bias.data = fca.mhsa.Wqkv.bias.data.clone().detach()
        tfca.mhsa.out_proj.weight.data = fca.mhsa.out_proj.weight.data.clone(
        ).detach()
        tfca.mhsa.out_proj.bias.data = fca.mhsa.out_proj.bias.data.clone(
        ).detach()

    key_padding_mask = torch.ones(n, s).to(device)
    x0 = torch.randn(n, s, f).to(device)
    x1 = x0.clone().detach()
    x0.requires_grad = True
    x1.requires_grad = True

    with torch.autocast(x0.device.type):
        y0, _ = fca(x0, key_padding_mask, attn_mask=None)
        y1, _ = tfca(x1, key_padding_mask, attn_mask=None)
        y0 *= key_padding_mask.unsqueeze(-1)
        y1 *= key_padding_mask.unsqueeze(-1)

        loss0 = y0.sum()
        loss1 = y1.sum()

    loss0.backward()
    loss1.backward()

    assert allclose_helper(y0, y1)

    if cfg.attn_qk_ln or cfg.attn_clip_qkv:
        assert allclose_helper(tfca.out_proj.bias.grad, fca.out_proj.bias.grad)
        assert allclose_helper(tfca.out_proj.weight.grad,
                               fca.out_proj.weight.grad)
        if cfg.attn_qk_ln:
            assert allclose_helper(tfca.q_ln.bias.grad, fca.q_ln.bias.grad)
            assert allclose_helper(tfca.q_ln.weight.grad, fca.q_ln.weight.grad)
            assert allclose_helper(tfca.k_ln.bias.grad, fca.k_ln.bias.grad)
            assert allclose_helper(tfca.k_ln.weight.grad, fca.k_ln.weight.grad)
        assert allclose_helper(tfca.Wqkv.bias.grad, fca.W_qkv.bias.grad)
        assert allclose_helper(tfca.Wqkv.weight.grad, fca.W_qkv.weight.grad)
    else:
        assert allclose_helper(tfca.mhsa.out_proj.bias.grad,
                               fca.mhsa.out_proj.bias.grad)
        assert allclose_helper(tfca.mhsa.out_proj.weight.grad,
                               fca.mhsa.out_proj.weight.grad)
        assert allclose_helper(tfca.mhsa.Wqkv.bias.grad,
                               fca.mhsa.Wqkv.bias.grad)
        assert allclose_helper(tfca.mhsa.Wqkv.weight.grad,
                               fca.mhsa.Wqkv.weight.grad)

    assert allclose_helper(x0.grad, x1.grad)
