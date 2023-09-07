# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import pytest

# Add repo root to path so we can import scripts and test it
repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(repo_dir)
import pathlib
from glob import glob
from typing import Callable, Iterable, List, Optional
from unittest.mock import Mock, patch
from concurrent.futures import ProcessPoolExecutor


import numpy as np
from streaming import StreamingDataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from scripts.data_prep.convert_text_to_mds import (download_and_convert,
                                                   get_done_file_name,
                                                   is_already_processed, main,
                                                   merge_shard_groups,
                                                   write_done_file)


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
        dirname = os.path.dirname(filename)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
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
        max_mds_writer_workers=1,
        compression='zstd',
        processes=processes,
        args_str='Namespace()',
        reprocess=False,
    )


# Mock starmap with no multiprocessing
def _mock_map(func: Callable, args: Iterable)-> Iterable:
    for arg in args:
        yield func(arg)


# Build a dataloader with no threading so mock call counts are correct
def _mock_build_dataloader(dataset: Dataset,
                           batch_size: int,
                           num_workers: Optional[int] = None) -> DataLoader:
    return DataLoader(
        dataset=dataset,
        sampler=None,
        batch_size=batch_size,
        num_workers=0,
    )


def _assert_files_exist(prefix: str, files: List[str]):
    for file in files:
        assert os.path.exists(os.path.join(prefix, file))


@pytest.mark.parametrize('processes', [1, 2, 3])
@patch('scripts.data_prep.convert_text_to_mds.build_dataloader',
       new=Mock(wraps=_mock_build_dataloader))
@patch.object(ProcessPoolExecutor, 'map', new=Mock(wraps=_mock_map))
@patch(
    'scripts.data_prep.convert_text_to_mds.maybe_create_object_store_from_uri')
@patch('scripts.data_prep.convert_text_to_mds.parse_uri')
@patch('scripts.data_prep.convert_text_to_mds.download_and_convert',
       wraps=download_and_convert)
@patch('scripts.data_prep.convert_text_to_mds.merge_shard_groups',
       wraps=merge_shard_groups)
def test_single_and_multi_process(merge_shard_groups: Mock,
                                  download_and_convert: Mock, parse_uri: Mock,
                                  maybe_create_object_store_from_uri: Mock,
                                  tmp_path: pathlib.Path, processes: int):
    remote_folder = os.path.join(tmp_path, 'remote')
    text_content = 'HELLO WORLD'
    tokenizer_name = 'mosaicml/mpt-7b'
    n_text_files = processes * 3

    mock_object_store = Mock(
        wraps=MockObjectStore(remote_folder, n_text_files, text_content))
    maybe_create_object_store_from_uri.return_value = mock_object_store
    parse_uri.return_value = ('s3', 'fake-test-bucket', str(remote_folder))

    _call_convert_text_to_mds(processes=processes,
                              tokenizer_name=tokenizer_name)

    # Check call counts
    assert download_and_convert.call_count == processes  # called once per process
    assert mock_object_store.download_object.call_count == n_text_files + 1  # text files + done file
    assert mock_object_store.upload_object.call_count == processes + 2  # shard per process + done file + index.json

    if processes > 1:
        merge_shard_groups.assert_called_once()

    total_object_names = 0
    for call_args in download_and_convert.call_args_list:
        object_names = call_args[0][0]
        total_object_names += len(object_names)

    assert total_object_names == n_text_files  # We should have processed all the text files

    # Check that correct output files exist
    shards = [f'shard.0000{i}.mds.zstd' for i in range(processes)]
    _assert_files_exist(prefix=remote_folder,
                        files=['index.json', get_done_file_name()] + shards)

    _call_convert_text_to_mds(processes=processes,
                              tokenizer_name=tokenizer_name)

    # Check call counts
    assert download_and_convert.call_count == processes  # No changes because we shoudn't reprocess
    assert mock_object_store.download_object.call_count == n_text_files + 2  # One more done file is downloaded
    assert mock_object_store.upload_object.call_count == processes + 2  # No changes

    # Create an extra text file and call again.
    n_text_files += 1
    mock_object_store = Mock(
        wraps=MockObjectStore(remote_folder, n_text_files, text_content))
    maybe_create_object_store_from_uri.return_value = mock_object_store

    _call_convert_text_to_mds(processes=processes,
                              tokenizer_name=tokenizer_name)

    # Check call counts
    assert download_and_convert.call_count == processes * 2  # called once per process
    assert mock_object_store.download_object.call_count == n_text_files + 1  # text files + done file
    assert mock_object_store.upload_object.call_count == processes + 2  # shard per process + done file + index.json

    # Check that the dataset can be used and produces samples with the original text_content
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    dataset = StreamingDataset(local=remote_folder)
    sample = dataset.__iter__().__next__()
    assert tokenizer.decode(np.frombuffer(sample['tokens'],
                                          dtype=int)).startswith(text_content)


def test_is_already_processed(tmp_path: pathlib.Path):
    tmp_path_str = str(tmp_path)
    fname = get_done_file_name()
    args_str = 'Namespace(x = 5)'
    object_names = ['test0.txt', 'test1.txt']

    assert not is_already_processed(tmp_path_str, fname, args_str,
                                    object_names)  # Done file doesn't exist

    write_done_file(tmp_path_str, fname, args_str, object_names)
    assert is_already_processed(tmp_path_str, fname, args_str,
                                object_names)  # Args and names match

    write_done_file(tmp_path_str, fname, args_str, object_names + ['test2.txt'])
    assert not is_already_processed(tmp_path_str, fname, args_str,
                                    object_names)  # Object names differ

    write_done_file(tmp_path_str, fname, 'Namespace()', object_names)
    assert not is_already_processed(tmp_path_str, fname, args_str,
                                    object_names)  # Argument strings differ
