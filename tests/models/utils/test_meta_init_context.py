# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from typing import Callable

import pytest
import torch
import torch.nn as nn

from llmfoundry.models.utils import init_empty_weights, init_on_device


class ModelWithIntParameter(nn.Module):

    def __init__(self):
        super().__init__()
        self.int_param = nn.Parameter(
            torch.tensor(0, dtype=torch.int64),
            requires_grad=False,
        )
        self.float_param = nn.Parameter(torch.randn(10), requires_grad=True)

    def forward(self, x: torch.Tensor):
        return x


def test_init_empty_weights(
    build_tiny_mpt: Callable,
):
    # Initialize a model on CPU for comparison
    cpu_model = build_tiny_mpt(loss_fn='torch_crossentropy')

    with init_empty_weights():
        meta_model = build_tiny_mpt(loss_fn='torch_crossentropy')

    for (cpu_name, cpu_param), (meta_name, meta_param) in zip(
        cpu_model.named_parameters(),
        meta_model.named_parameters(),
    ):
        assert cpu_name == meta_name, f'Parameter names do not match: {cpu_name} vs {meta_name}'
        assert cpu_param.shape == meta_param.shape, f'Shape mismatch for {cpu_name}: {cpu_param.shape} vs {meta_param.shape}'
        assert meta_param.device == torch.device(
            'meta',
        ), f'Parameter {meta_name} is not on meta device'
        assert cpu_param.dtype == meta_param.dtype, f'Dtype mismatch for {cpu_name}: {cpu_param.dtype} vs {meta_param.dtype}'
        assert cpu_param.requires_grad == meta_param.requires_grad, f'requires_grad mismatch for {cpu_name}'


def test_init_empty_weights_with_buffers():
    with init_empty_weights(include_buffers=True):
        model = nn.BatchNorm1d(10)

    assert model.running_mean is not None
    assert model.running_var is not None
    assert model.num_batches_tracked is not None

    assert model.running_mean.device == torch.device('meta')
    assert model.running_var.device == torch.device('meta')
    assert model.num_batches_tracked.device == torch.device('meta')


@pytest.mark.gpu
def test_init_on_device(
    build_tiny_mpt: Callable,
):
    device = torch.device('cuda')
    with init_on_device(device):
        model = build_tiny_mpt()

    for name, param in model.named_parameters():
        assert param.device.type == device.type, f'Parameter {name} is not on a CUDA device'
        assert param.dtype == torch.float32, f'Parameter {name} is not float32'


@pytest.mark.gpu
def test_init_on_device_with_buffers():
    device = torch.device('cuda')
    with init_on_device(device, include_buffers=True):
        model = nn.BatchNorm1d(10)

    assert model.running_mean is not None
    assert model.running_var is not None
    assert model.num_batches_tracked is not None

    assert model.running_mean.device.type == device.type
    assert model.running_var.device.type == device.type
    assert model.num_batches_tracked.device.type == device.type


@pytest.mark.gpu
def test_init_on_device_int_parameter():
    device = torch.device('cuda')
    with init_on_device(device):
        model = ModelWithIntParameter()

    assert model.int_param.device.type == device.type, 'Int parameter should be on a CUDA device'
    assert model.int_param.dtype == torch.int64, 'Parameter dtype should be int64'
    assert not model.int_param.requires_grad, 'Int parameter should not require gradients'

    assert model.float_param.device.type == device.type, 'Float parameter should be on a CUDA device'
    assert model.float_param.dtype == torch.float32, 'Float parameter should be float32'
    assert model.float_param.requires_grad, 'Float parameter should require gradients'


if __name__ == '__main__':
    pytest.main([__file__])
