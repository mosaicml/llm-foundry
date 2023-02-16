# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from composer.models import ComposerClassifier

from examples.resnet_imagenet.model import build_composer_resnet


@pytest.mark.parametrize(
    'model_name',
    ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
@pytest.mark.parametrize('loss_name', ['binary_cross_entropy', 'cross_entropy'])
@pytest.mark.parametrize('num_classes', [10, 100])
def test_model_builder(model_name, loss_name, num_classes):
    model = build_composer_resnet(model_name, loss_name, num_classes)
    assert isinstance(model, ComposerClassifier)

    rand_input = torch.randn(1, 3, 64, 64)
    rand_label = torch.randint(0, num_classes - 1, (1,))
    output = model((rand_input, rand_label))
    assert output.shape == (1, num_classes)
    assert output.dtype == torch.float
