# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
import shutil

import numpy as np
import streaming
from PIL import Image

class SynthClassificationDirectory(object):

    def __enter__(self):
        self.path = create_synthetic_cifar()
        return self.path

    def __exit__(self, exc_type, exc_value, traceback):
        shutil.rmtree(self.path)

def create_synthetic_cifar(n_samples = 16):
    tmp_dirname = tempfile.mkdtemp()
    fields = {
        'x': 'pil',
        'y': 'int',
    }

    for split in ['train', 'test']:
        dirname = os.path.join(tmp_dirname, split)
        hashes = ['sha1' ,'xxh64']
        size_limit = 1 << 25
        with streaming.MDSWriter(dirname=dirname, columns=fields, hashes=hashes, size_limit=size_limit) as out:
            for i in range(n_samples):
                arr = np.random.rand(32, 32, 3) * 255
                img = Image.fromarray(arr.astype('uint8')).convert('RGB')
                y = i % 4
                out.write({'x': img, 'y': y})

    return tmp_dirname
