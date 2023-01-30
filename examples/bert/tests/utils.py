# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import os
import random
import shutil
import tempfile
from typing import Any

import numpy as np
import streaming


class SynthTextDirectory(object):

    def __enter__(self):
        path = create_synthetic_text_dataset()
        self.path = path  # type: ignore (reportUninitializedInstanceVariable)
        return self.path

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any):
        del exc_type, exc_value, traceback  # Unused
        shutil.rmtree(self.path)


def create_synthetic_text_dataset(n_samples: int = 16):
    tmp_dirname = tempfile.mkdtemp()

    for split in ['train', 'val']:
        dirname = os.path.join(tmp_dirname, split)
        hashes = ['sha1', 'xxh64']
        size_limit = 1 << 25
        with streaming.MDSWriter(dirname=dirname,
                                 columns={'text': 'str'},
                                 hashes=hashes,
                                 size_limit=size_limit) as out:
            for _ in range(n_samples):
                n_letters = np.random.randint(low=5, high=256)
                letter_str = ' '.join([
                    random.choice('abcdefghijklmnopqrstuvwxyz')
                    for _ in range(n_letters)
                ])
                out.write({'text': letter_str})

    return tmp_dirname
