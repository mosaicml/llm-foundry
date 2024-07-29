# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from llmfoundry.models.layers.ffn import quickgelu_activation
from llmfoundry.models.layers.layer_builders import build_ffn


@pytest.mark.gpu
def test_quickgelu_activation():
    d_model = 32
    expansion_ratio = 1
    no_bias = True
    ffn_config = {
        'ffn_act_fn': {
            'name': 'quick_gelu',
        },
        'ffn_type': 'mptmlp',
    }
    rank: int = dist.get_rank()
    device_str = f'cuda:{rank}'
    device: torch.device = torch.device(device_str)

    ffn1 = build_ffn(
        name=ffn_config['ffn_type'],
        d_model=d_model,
        expansion_ratio=expansion_ratio,
        device=device_str,
        bias=not no_bias,
        ffn_kwargs=ffn_config,
    )
    assert (
        ffn1.act == quickgelu_activation
    ), f'Expected quick_gelu activation function, got {ffn1.act}'

    ffn_config = {
        'ffn_act_fn': {
            'name': 'gelu',
        },
        'ffn_type': 'mptmlp',
    }
    ffn2 = build_ffn(
        name=ffn_config['ffn_type'],
        d_model=d_model,
        expansion_ratio=expansion_ratio,
        device=device_str,
        bias=not no_bias,
        ffn_kwargs=ffn_config,
    )

    def num_params(model: nn.Module) -> int:
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        return sum([p.numel() for p in model_parameters])

    ffn1_numparams = num_params(ffn1)
    ffn2_numparams = num_params(ffn2)
    assert (
        ffn1_numparams == ffn2_numparams
    ), 'Only activation paths should have changed, re-check modeling!'

    input_ = torch.rand(1, d_model, device=device)
    output1 = ffn1(input_)
    output2 = ffn2(input_)
    assert (
        output1.numel() == output2.numel()
    ), 'Only activation paths should have changed, re-check modeling!'
    assert (
        not torch.allclose(output1, output2)
    ), 'Functions are different, outputs should not match!'
