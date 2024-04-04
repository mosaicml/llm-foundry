# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from typing import List, Union

import pytest
import torch
from composer.core.precision import get_precision_context

from llmfoundry.models.layers.attention import is_flash_v2_installed


@pytest.mark.gpu
@pytest.mark.parametrize('normalized_shape', [32, 128, 4096])
def test_rmsnorm_triton_vs_eager(normalized_shape: Union[int, List[int]],
                                 device: str = 'cuda'):
    # Compare Triton and PyTorch Eager implementations of RMSNorm
    if not is_flash_v2_installed():
        pytest.skip(
            'triton implementation of rmsnorm requires flash attention 2.')

    from llmfoundry.models.layers import norm

    batch_size = 2

    cfg = {
        'normalized_shape': normalized_shape,
        'device': device,
    }

    eager_rmsnorm = norm.NORM_CLASS_REGISTRY['rmsnorm'](**cfg)
    triton_rmsnorm = norm.NORM_CLASS_REGISTRY['triton_rmsnorm'](**cfg)

    triton_rmsnorm.load_state_dict(eager_rmsnorm.state_dict())

    if isinstance(normalized_shape, int):
        input_shape = [batch_size, normalized_shape]
    else:
        input_shape = tuple([batch_size, *normalized_shape])

    x0 = torch.randn(size=input_shape, device=device)
    x1 = x0.clone().detach()
    x0.requires_grad = True
    x1.requires_grad = True

    with get_precision_context('amp_bf16'):
        y0 = eager_rmsnorm(x0)
        y1 = triton_rmsnorm(x1)

        loss0 = y0.sum()
        loss1 = y1.sum()

    loss0.backward()
    loss1.backward()

    rtol = 1e-6
    atol = 1e-6

    torch.testing.assert_close(y0, y1, rtol=rtol, atol=atol)

    p0 = eager_rmsnorm.weight
    p1 = triton_rmsnorm.weight

    # weight check
    torch.testing.assert_close(p0, p1, rtol=rtol, atol=atol)
    # weight gradient check
    assert p0.grad is not None
    assert p1.grad is not None
    assert torch.norm(p0.grad - p1.grad) <= atol + rtol * torch.norm(p0.grad)

    # input gradient check
    assert x0.grad is not None
    assert x1.grad is not None
    # Relaxed to a l2-norm based check.
    assert torch.norm(x0.grad - x1.grad) <= atol + rtol * torch.norm(x0.grad)
