# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from composer.models import ComposerClassifier

from ..model import build_composer_deeplabv3


@pytest.mark.parametrize('num_classes', [10, 150])
@pytest.mark.parametrize('dice_weight', [0.0, 1.25])
@pytest.mark.parametrize('cross_entropy_weight', [1.0, 0.375])
def test_model_builder(num_classes, dice_weight, cross_entropy_weight):
    model = build_composer_deeplabv3(num_classes=num_classes,
                                     dice_weight=dice_weight,
                                     cross_entropy_weight=cross_entropy_weight)
    assert isinstance(model, ComposerClassifier)

    rand_input = torch.randn(2, 3, 64, 64)
    rand_target = torch.randint(low=0, high=num_classes - 1, size=(2, 64, 64))
    output = model((rand_input, rand_target))
    assert output.shape == (2, num_classes, 64, 64)
    assert output.dtype == torch.float
