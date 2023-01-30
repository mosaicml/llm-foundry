# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import itertools
import os
import shutil
import tempfile

import numpy as np
from PIL import Image


class SynthClassificationDirectory(object):

    def __enter__(self):
        path = create_synthetic_imagenet()
        self.path = path  # type: ignore (reportUninitializedInstanceVariable)
        return self.path

    def __exit__(self, exc_type, exc_value, traceback):
        shutil.rmtree(self.path)


def create_synthetic_imagenet():
    tmp_dirname = tempfile.mkdtemp()

    # Make train and val directories with a few classes, then add images
    splits = ['train', 'val']
    labels = ['dog', 'cat', 'person']
    n_img_per_class = 4
    for split, label in itertools.product(splits, labels):
        dirpath = os.path.join(tmp_dirname, split, label)
        os.makedirs(dirpath)

        for i in range(n_img_per_class):
            arr = np.random.rand(32, 32, 3) * 255
            img = Image.fromarray(arr.astype('uint8')).convert('RGB')
            img.save(os.path.join(dirpath, f'img{i}.jpeg'))

    return tmp_dirname
