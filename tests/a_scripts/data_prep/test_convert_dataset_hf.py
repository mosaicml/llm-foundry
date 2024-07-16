# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

from llmfoundry.command_utils import convert_dataset_hf


def test_download_script_from_api(tmp_path: Path):
    # test calling it directly
    path = os.path.join(tmp_path, 'my-copy-c4-1')
    convert_dataset_hf(
        dataset='c4',
        data_subset='en',
        splits=['val_xsmall'],
        out_root=path,
        compression=None,
        concat_tokens=None,
        bos_text='',
        eos_text='',
        no_wrap=False,
        num_workers=None,
        tokenizer=None,
        tokenizer_kwargs={},
    )
    assert os.path.exists(path)
