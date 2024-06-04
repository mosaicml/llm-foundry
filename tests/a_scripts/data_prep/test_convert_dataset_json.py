# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import os
from argparse import Namespace
from pathlib import Path

from scripts.data_prep.convert_dataset_json import main as main_json


def test_json_script_from_api(tmp_path: Path):
    # test calling it directly
    path = os.path.join(tmp_path, 'my-copy-arxiv-1')
    main_json(
        Namespace(
            **{
                'path': 'scripts/data_prep/example_data/arxiv.jsonl',
                'out_root': path,
                'compression': None,
                'split': 'train',
                'concat_tokens': None,
                'bos_text': None,
                'eos_text': None,
                'no_wrap': False,
                'num_workers': None,
            },
        ),
    )
    assert os.path.exists(path)
