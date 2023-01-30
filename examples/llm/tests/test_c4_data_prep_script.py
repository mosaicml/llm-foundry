# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
from argparse import Namespace

from examples.common.convert_c4 import main


def test_download_script_from_api():
    # test calling it directly
    path = os.path.join(os.getcwd(), 'my-copy-c4-1')
    shutil.rmtree(path, ignore_errors=True)
    main(
        Namespace(**{
            'splits': ['val'],
            'out_root': './my-copy-c4-1',
            'compression': None,
        }))
    assert os.path.exists(path)
    shutil.rmtree(path, ignore_errors=False)


def test_download_script_from_cmdline():
    # test calling it via the cmd line interface
    path = os.path.join(os.getcwd(), 'my-copy-c4-2')
    shutil.rmtree(path, ignore_errors=True)
    os.system(
        'python ../common/convert_c4.py --out_root ./my-copy-c4-2 --splits val')
    assert os.path.exists(path)
    shutil.rmtree(path, ignore_errors=False)
