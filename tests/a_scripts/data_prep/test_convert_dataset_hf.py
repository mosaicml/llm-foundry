# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path
from unittest.mock import patch

import datasets as hf_datasets

from llmfoundry.command_utils import convert_dataset_hf


def test_download_script_from_api(
    tmp_path: Path,
    tiny_text_hf_dataset: hf_datasets.Dataset,
):
    with patch('datasets.load_dataset') as mock_load_dataset:
        mock_load_dataset.return_value = tiny_text_hf_dataset

        path = os.path.join(tmp_path, 'my-copy-c4-1')
        convert_dataset_hf(
            dataset='allenai/c4',
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
        mock_load_dataset.assert_called_once()
