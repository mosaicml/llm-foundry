# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import os
from argparse import Namespace
from pathlib import Path

from scripts.data_prep.convert_dataset_hf import main as main_hf


def test_download_script_from_api(tmp_path: Path):
    # test calling it directly
    path = os.path.join(tmp_path, 'my-copy-c4-1')
    main_hf(
        Namespace(
            **{
                'dataset': 'c4',
                'data_subset': 'en',
                'splits': ['val_xsmall'],
                'out_root': path,
                'compression': None,
                'concat_tokens': None,
                'bos_text': None,
                'eos_text': None,
                'no_wrap': False,
                'num_workers': None,
            },
        ),
    )
    assert os.path.exists(path)
