# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import os
import sys

# Add repo root to path so we can import scripts and test it
repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(repo_dir)
import pathlib
from glob import glob
from multiprocessing.pool import Pool
from typing import Callable, Iterable, List, Optional
from unittest.mock import Mock, patch

import numpy as np
from streaming import StreamingDataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset

from scripts.data_prep.convert_text_to_mds import (download_and_convert, main,
                                                   merge_shard_groups)
from scripts.data_prep.utils import build_dataloader



class MockObjectStore():
    def __init__(self, remote_folder: str, n_text_files: int,
                 text_content: str):
        os.makedirs(remote_folder, exist_ok=True)
        for i in range(n_text_files):
            with open(os.path.join(remote_folder, f'test{i}.txt'), 'w') as f:
                f.write((text_content + ' ') * 500)
        self.remote_folder = remote_folder
        self.n_text_files = n_text_files

    def download_object(self,
                        object_name: str,
                        filename: str,
                        overwrite: bool = False):
        with open(
                os.path.join(self.remote_folder, os.path.basename(object_name)),
                'rb') as remote_file, open(filename, 'wb') as local_file:
            local_file.write(remote_file.read())

    def list_objects(self, prefix: str) -> List[str]:
        return glob(os.path.join(self.remote_folder, '*.txt'))

    def upload_object(self, object_name: str, filename: str):
        with open(
                os.path.join(self.remote_folder, os.path.basename(object_name)),
                'wb') as remote_file, open(filename, 'rb') as local_file:
            remote_file.write(local_file.read())


def _call_convert_text_to_mds(processes: int, tokenizer_name: str) -> None:
    main(
        tokenizer_name=tokenizer_name,
        output_folder=f's3://fake-test-output-path',
        input_folder=f's3://fake-test-input-path',
        concat_tokens=2048,
        eos_text='',
        bos_text='',
        no_wrap=False,
        max_workers=1,
        compression='zstd',
        processes=processes,
        args_str='Namespace()',
        reprocess=False,
    )

# Mock starmap with no multiprocessing
def _mock_starmap(func: Callable, args: Iterable):
    for arg in args:
        func(*arg)

# Build a dataloader with no threading so mock call counts are correct
def _mock_build_dataloader(dataset: Dataset,
                     batch_size: int,
                     num_workers: Optional[int] = None) -> DataLoader:
    return build_dataloader(dataset, batch_size, num_workers=0) 

def _assert_files_exist(prefix: str, files: List[str]):
    for file in files:
        assert os.path.exists(os.path.join(prefix, file))

@patch('scripts.data_prep.convert_text_to_mds.build_dataloader', new=Mock(wraps=_mock_build_dataloader))
@patch.object(Pool, 'starmap', new=Mock(wraps=_mock_starmap))
@patch(
    'scripts.data_prep.convert_text_to_mds.maybe_create_object_store_from_uri')
@patch('scripts.data_prep.convert_text_to_mds.parse_uri')
@patch('scripts.data_prep.convert_text_to_mds.download_and_convert',
       wraps=download_and_convert)
@patch('scripts.data_prep.convert_text_to_mds.merge_shard_groups',
       wraps=merge_shard_groups)
def test_multi_process(merge_shard_groups: Mock,
                       download_and_convert: Mock, parse_uri: Mock,
                       maybe_create_object_store_from_uri: Mock,
                       tmp_path: pathlib.Path):
    remote_folder = os.path.join(tmp_path, 'remote')
    n_text_files = 6
    text_content = 'HELLO WORLD'
    tokenizer_name = 'mosaicml/mpt-7b'

    mock_object_store = Mock(wraps=MockObjectStore(remote_folder, n_text_files, text_content))
    maybe_create_object_store_from_uri.return_value = mock_object_store
    parse_uri.return_value = ('s3', 'fake-test-bucket', str(remote_folder))

    _call_convert_text_to_mds(processes=2, tokenizer_name=tokenizer_name)

    assert download_and_convert.call_count == 2
    assert mock_object_store.download_object.call_count == n_text_files + 1
    assert mock_object_store.upload_object.call_count == 4  # 2 shards + done file + index.json
    merge_shard_groups.assert_called_once()

    object_names_0 = download_and_convert.call_args_list[0][0][0]
    object_names_1 = download_and_convert.call_args_list[1][0][0]

    assert len(
        object_names_0
    ) == n_text_files / 2  # Half of the text files should be called with process 0
    assert len(
        object_names_1
    ) == n_text_files / 2  # Half of the text files should be called with process 1
    assert len(
        set(object_names_0 + object_names_1)
    ) == n_text_files  # There should be n_text_files unique object names

    # Check that correct output files exist
    _assert_files_exist(prefix=remote_folder,
                        files=[
                            'index.json',
                            'done',
                            'shard.00000.mds.zstd',
                            'shard.00001.mds.zstd',
                        ])

    # Check that the dataset can be used and produces samples with the original text_content
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    dataset = StreamingDataset(local=remote_folder)
    sample = dataset.__iter__().__next__()
    assert tokenizer.decode(np.frombuffer(sample['tokens'],
                                          dtype=int)).startswith(text_content)


@patch('scripts.data_prep.convert_text_to_mds.build_dataloader', new=Mock(wraps=_mock_build_dataloader))
@patch(
    'scripts.data_prep.convert_text_to_mds.maybe_create_object_store_from_uri')
@patch('scripts.data_prep.convert_text_to_mds.parse_uri')
@patch('scripts.data_prep.convert_text_to_mds.download_and_convert',
       wraps=download_and_convert)
def test_single_process(download_and_convert: Mock, parse_uri: Mock,
                        maybe_create_object_store_from_uri: Mock,
                        tmp_path: pathlib.Path):
    remote_folder = os.path.join(tmp_path, 'remote')
    n_text_files = 3
    text_content = 'HELLO WORLD'
    tokenizer_name = 'mosaicml/mpt-7b'

    mock_object_store = Mock(
        wraps=MockObjectStore(remote_folder, n_text_files, text_content))
    maybe_create_object_store_from_uri.return_value = mock_object_store
    parse_uri.return_value = ('s3', 'fake-test-bucket', str(remote_folder))

    _call_convert_text_to_mds(processes=1, tokenizer_name=tokenizer_name)

    # Check call counts
    download_and_convert.assert_called_once()
    assert mock_object_store.download_object.call_count == n_text_files + 1  # Downloaded n_text_files files + done file
    assert mock_object_store.upload_object.call_count == 3  # Uploaded done file, index.json, and shard

    # Check that correct output files exist
    _assert_files_exist(prefix=remote_folder,
                        files=[
                            'index.json',
                            'done',
                            'shard.00000.mds.zstd',
                        ])

    _call_convert_text_to_mds(processes=1, tokenizer_name=tokenizer_name)

    # Check call counts
    download_and_convert.assert_called_once(
    )  # No changes because the input files and args are unchanged.
    assert mock_object_store.download_object.call_count == n_text_files + 2  # Downloaded the done file an extra time
    assert mock_object_store.upload_object.call_count == 3  # No changes

    # Create an extra text file and call again.
    mock_object_store = Mock(wraps=MockObjectStore(remote_folder, n_text_files +
                                                   1, text_content))
    maybe_create_object_store_from_uri.return_value = mock_object_store

    _call_convert_text_to_mds(processes=1, tokenizer_name=tokenizer_name)

    # Check call counts
    assert download_and_convert.call_count == 2  # download_and_convert should have been called once more.
    assert mock_object_store.download_object.call_count == n_text_files + 1 + 1  # Downloaded n_text_files + 1 files + done file
    assert mock_object_store.upload_object.call_count == 3  # Uploaded done file, index.json, and shard

    # Check that the dataset can be used and produces samples with the original text_content
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    dataset = StreamingDataset(local=remote_folder)
    sample = dataset.__iter__().__next__()
    assert tokenizer.decode(np.frombuffer(sample['tokens'],
                                          dtype=int)).startswith(text_content)
