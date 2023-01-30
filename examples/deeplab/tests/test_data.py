# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import sys
import tempfile

import numpy as np
import pytest
import torch
from PIL import Image

from ..data import build_ade20k_dataspec, check_dataloader


class SynthADE20KDirectory:

    def __enter__(self):
        path = create_synthetic_ade20k()
        self.path = path  # type: ignore (reportUninitializedInstanceVariable)
        return self.path

    def __exit__(self, exc_type, exc_value, traceback):
        shutil.rmtree(self.path)


def create_synthetic_ade20k():
    tmp_dirname = tempfile.mkdtemp()

    ade20k_dir = os.path.join(tmp_dirname, 'ADEChallengeData2016')
    annotations_dir = os.path.join(ade20k_dir, 'annotations')
    images_dir = os.path.join(ade20k_dir, 'images')

    train_images_path = os.path.join(images_dir, 'training')
    train_annotations_path = os.path.join(annotations_dir, 'training')

    val_images_path = os.path.join(images_dir, 'validation')
    val_annotations_path = os.path.join(annotations_dir, 'validation')

    os.makedirs(ade20k_dir)
    os.makedirs(annotations_dir)
    os.makedirs(images_dir)

    os.makedirs(train_images_path)
    os.makedirs(train_annotations_path)

    os.makedirs(val_images_path)
    os.makedirs(val_annotations_path)

    # Make train and val directories with a few classes, then add images
    num_samples = 4
    num_classes = 150
    height = 32
    width = 32
    for i in range(num_samples):
        train_image = np.random.rand(height, width, 3) * 255
        trian_target = np.random.randint(low=0,
                                         high=num_classes - 1,
                                         size=(height, width))

        train_image = Image.fromarray(
            train_image.astype('uint8')).convert('RGB')
        train_image.save(
            os.path.join(train_images_path, f'ADE_train_0000000{i}.jpg'))

        train_target = Image.fromarray(trian_target.astype('uint8'))
        train_target.save(
            os.path.join(train_annotations_path, f'ADE_train_0000000{i}.png'))

        val_image = np.random.rand(height, width, 3) * 255
        val_target = np.random.randint(low=0,
                                       high=num_classes - 1,
                                       size=(height, width))

        val_image = Image.fromarray(val_image.astype('uint8')).convert('RGB')
        val_image.save(os.path.join(val_images_path, f'ADE_val_0000000{i}.jpg'))

        val_target = Image.fromarray(val_target.astype('uint8'))
        val_target.save(
            os.path.join(val_annotations_path, f'ADE_val_0000000{i}.png'))

    return tmp_dirname


@pytest.mark.parametrize('split', ['train', 'val'])
def test_dataloader_builder(split, batch_size=2, base_size=32, final_size=32):
    with SynthADE20KDirectory() as datadir:
        ade20k_dataspec = build_ade20k_dataspec(datadir,
                                                is_streaming=False,
                                                batch_size=batch_size,
                                                split=split,
                                                shuffle=False,
                                                base_size=base_size,
                                                final_size=final_size)

        for batch in ade20k_dataspec.dataloader:
            # Check the image and label shapes
            image, target = batch
            assert image.shape == torch.Size(
                [batch_size, 3, final_size, final_size])
            assert target.shape == torch.Size(
                [batch_size, final_size, final_size])


def test_check_dataloader():
    with SynthADE20KDirectory() as datadir:
        tmp_argv = sys.argv.copy()
        sys.argv = ['script.py', datadir]
        check_dataloader()
        sys.argv = tmp_argv
