# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch.utils.data import DataLoader

from examples.resnet_cifar.data import build_cifar10_dataspec
from examples.resnet_cifar.tests.utils import SynthClassificationDirectory

# TODO: streaming dataset and dataloader testing


@pytest.mark.parametrize('is_train', [True, False])
def test_dataloader_builder(is_train, batch_size=2):
    with SynthClassificationDirectory() as datadir:
        cifar_dataspec = build_cifar10_dataspec(data_path=datadir,
                                                is_streaming=True,
                                                local=datadir,
                                                batch_size=batch_size,
                                                is_train=is_train,
                                                download=False)

        assert isinstance(cifar_dataspec.dataloader, DataLoader)
        dataloader = cifar_dataspec.dataloader
        print(len(dataloader))
        assert len(dataloader) == 8

        for batch in dataloader:
            # Check the image and label shapes
            assert batch[0].shape == torch.Size([batch_size, 3, 32, 32])
            assert batch[1].shape == torch.Size([batch_size])
