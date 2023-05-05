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
from scripts.data_prep.convert_dataset_csv import main as main_csv


def test_download_script_from_api():
    # test calling it directly
    path = os.path.join(os.getcwd(), 'my-copy-c4-1')
    shutil.rmtree(path, ignore_errors=True)
    main_hf(
        Namespace(
            **{
                'dataset': 'c4',
                'data_subset': 'en',
                'splits': ['val_small'],
                'out_root': './my-copy-c4-1',
                'compression': None,
                'concat_tokens': None,
                'bos_text': None,
                'eos_text': None,
                'no_wrap': False
            }))
    assert os.path.exists(path)
    shutil.rmtree(path, ignore_errors=False)


def test_download_script_from_cmdline():
    # test calling it via the cmd line interface
    path = os.path.join(os.getcwd(), 'my-copy-c4-2')
    shutil.rmtree(path, ignore_errors=True)
    print(os.getcwd())
    os.system(
        'python scripts/data_prep/convert_dataset_hf.py --dataset c4 --data_subset en --out_root ./my-copy-c4-2 --splits val_xsmall'
    )
    assert os.path.exists(path)
    shutil.rmtree(path, ignore_errors=False)


def test_json_script_from_api():
    # test calling it directly
    path = os.path.join(os.getcwd(), 'my-copy-c4-1')
    shutil.rmtree(path, ignore_errors=True)
    main_json(
        Namespace(
            **{
                'path': 'scripts/example_data/arxiv.jsonl',
                'out_root': './my-copy-c4-1',
                'compression': None,
                'concat_tokens': None,
                'bos_text': None,
                'eos_text': None,
                'no_wrap': False
            }))
    assert os.path.exists(path)
    shutil.rmtree(path, ignore_errors=False)


def test_json_script_from_cmdline():
    # test calling it via the cmd line interface
    path = os.path.join(os.getcwd(), 'my-copy-c4-2')
    shutil.rmtree(path, ignore_errors=True)
    print(os.getcwd())
    os.system(
        'python scripts/data_prep/convert_dataset_json.py --path scripts/example_data/arxiv.jsonl --out_root ./my-copy-c4-2'
    )
    assert os.path.exists(path)
    shutil.rmtree(path, ignore_errors=False)

def test_csv_script_from_api():
    # test calling it directly
    path = os.path.join(os.getcwd(), 'my-copy-c4-1')
    shutil.rmtree(path, ignore_errors=True)
    main_csv(
        Namespace(
            **{
                'path': 'scripts/example_data/arxiv.csv',
                'out_root': './my-copy-c4-1',
                'compression': None,
                'concat_tokens': None,
                'bos_text': None,
                'eos_text': None,
                'no_wrap': False
            }))
    assert os.path.exists(path)
    shutil.rmtree(path, ignore_errors=False)


def test_csv_script_from_cmdline():
    # test calling it via the cmd line interface
    path = os.path.join(os.getcwd(), 'my-copy-c4-2')
    shutil.rmtree(path, ignore_errors=True)
    print(os.getcwd())
    os.system(
        'python scripts/data_prep/convert_dataset_csv.py --path scripts/example_data/arxiv.csv --out_root ./my-copy-c4-2'
    )
    assert os.path.exists(path)
    shutil.rmtree(path, ignore_errors=False)

