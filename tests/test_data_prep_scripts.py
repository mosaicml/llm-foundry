# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import sys
from argparse import Namespace

# Add repo root to path so we can import scripts and test it
repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(repo_dir)
from scripts.data_prep.convert_dataset_hf import main as main_hf
from scripts.data_prep.convert_dataset_json import main as main_json


def test_download_script_from_api():
    # test calling it directly
    path = os.path.join(os.getcwd(), 'my-copy-c4-1')
    shutil.rmtree(path, ignore_errors=True)
    main_hf(
        Namespace(
            **{
                'dataset': 'c4',
                'data_subset': 'en',
                'splits': ['val_xsmall'],
                'out_root': './my-copy-c4-1',
                'compression': None,
                'concat_tokens': None,
                'bos_text': None,
                'eos_text': None,
                'no_wrap': False,
                'num_workers': None
            }))
    assert os.path.exists(path)
    shutil.rmtree(path, ignore_errors=False)


def test_json_script_from_api():
    # test calling it directly
    path = os.path.join(os.getcwd(), 'my-copy-c4-3')
    shutil.rmtree(path, ignore_errors=True)
    main_json(
        Namespace(
            **{
                'path': 'scripts/data_prep/example_data/arxiv.jsonl',
                'out_root': './my-copy-c4-3',
                'compression': None,
                'split': 'train',
                'concat_tokens': None,
                'bos_text': None,
                'eos_text': None,
                'no_wrap': False,
                'num_workers': None
            }))
    assert os.path.exists(path)
    shutil.rmtree(path, ignore_errors=False)
