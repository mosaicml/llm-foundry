# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

from llmfoundry.command_utils import convert_dataset_json


def test_json_script_from_api(tmp_path: Path):
    # test calling it directly
    path = os.path.join(tmp_path, 'my-copy-arxiv-1')
    convert_dataset_json(
        path='scripts/data_prep/example_data/arxiv.jsonl',
        out_root=path,
        compression=None,
        split='train',
        concat_tokens=None,
        bos_text='',
        eos_text='',
        no_wrap=False,
        num_workers=None,
    )
    assert os.path.exists(path)
