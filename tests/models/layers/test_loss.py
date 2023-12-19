# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

# Copied and modified from https://github.com/Dao-AILab/flash-attention/blob/713bd3aa9ad518ecdb5fd41078550c25ebd58e1f/tests/losses/test_cross_entropy.py

import pytest
import torch
from flash_attn.losses.cross_entropy import CrossEntropyLoss


@pytest.mark.gpu
def test_cross_entropy_loss():
    batch_size = 4
    seqlen = 2048
    vocab_size = 100352
    dtype = torch.bfloat16
    device = 'cuda'
    rtol, atol = (1e-3, 1e-4)
    # set seed
    torch.random.manual_seed(0)
    x_pt = torch.randn(batch_size * seqlen,
                       vocab_size,
                       device=device,
                       dtype=dtype,
                       requires_grad=True)
    x = x_pt.detach().clone().requires_grad_()
    y = torch.randint(0,
                      vocab_size, (batch_size * seqlen,),
                      dtype=torch.long,
                      device=device)
    y[torch.randperm(batch_size * seqlen)[:10]] = -100
    model_pt = torch.nn.CrossEntropyLoss()
    model = CrossEntropyLoss()
    out = model(x, y)
    x_pt_scaled = x_pt.float()
    out_pt = model_pt(x_pt_scaled, y)
    assert torch.allclose(out, out_pt, rtol=1e-5, atol=1e-6)

    g = torch.randn_like(out)
    out_pt.backward(g)
    out.backward(g)
    assert torch.allclose(
        x.grad,  # type: ignore
        x_pt.grad,  # type: ignore
        rtol=rtol,
        atol=atol)
