# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import sys

import pytest
import torch

from ..data import build_imagenet_dataspec, check_dataloader
from ..tests.utils import SynthClassificationDirectory

# TODO: streaming dataset and dataloader testing


@pytest.mark.parametrize('is_train', [True, False])
def test_dataloader_builder(is_train, batch_size=2, crop_size=16):
    with SynthClassificationDirectory() as datadir:
        imagenet_dataspec = build_imagenet_dataspec(datadir,
                                                    is_streaming=False,
                                                    batch_size=batch_size,
                                                    is_train=is_train,
                                                    shuffle=False,
                                                    crop_size=crop_size)

        for batch in imagenet_dataspec.dataloader:
            # Check the image and label shapes
            assert batch[0].shape == torch.Size(
                [batch_size, 3, crop_size, crop_size])
            assert batch[1].shape == torch.Size([batch_size])


def test_check_dataloader():
    with SynthClassificationDirectory() as datadir:
        tmp_argv = sys.argv.copy()
        sys.argv = ['script.py', datadir]
        check_dataloader()
        sys.argv = tmp_argv
