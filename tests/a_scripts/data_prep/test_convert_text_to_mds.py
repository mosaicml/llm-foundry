# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib
import shutil
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from typing import Callable, Iterable, List
from unittest.mock import Mock, patch

import numpy as np
import pytest
from streaming import StreamingDataset
from transformers import AutoTokenizer

from scripts.data_prep.convert_text_to_mds import (DONE_FILENAME,
                                                   convert_text_to_mds,
                                                   download_and_convert,
                                                   is_already_processed,
                                                   merge_shard_groups,
                                                   write_done_file)


class MockObjectStore():

    def __init__(self, remote_folder: str, n_text_files: int,
                 text_content: str):
        os.makedirs(remote_folder, exist_ok=True)
        for i in range(n_text_files):
            with open(os.path.join(remote_folder, f'test{i}.txt'), 'w') as f:
                f.write(text_content)

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


# Mock starmap with no multiprocessing
def _mock_map(func: Callable, args: Iterable) -> Iterable:
    for arg in args:
        yield func(arg)


def _assert_files_exist(prefix: str, files: List[str]):
    for file in files:
        assert os.path.exists(os.path.join(prefix, file))


@pytest.mark.parametrize('processes', [1, 2, 3])
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
    text_content = 'HELLO WORLD ' * 500
    tokenizer_name = 'mosaicml/mpt-7b'
    n_text_files = processes * 3
    concat_tokens = 2048

    mock_object_store = Mock(
        wraps=MockObjectStore(remote_folder, n_text_files, text_content))
    maybe_create_object_store_from_uri.return_value = mock_object_store
    parse_uri.return_value = ('s3', 'fake-test-bucket', str(remote_folder))

    def call_convert_text_to_mds(processes: int) -> None:
        convert_text_to_mds(
            tokenizer_name=tokenizer_name,
            output_folder=f's3://fake-test-output-path',
            input_folder=f's3://fake-test-input-path',
            concat_tokens=concat_tokens,
            eos_text='',
            bos_text='',
            no_wrap=False,
            compression='zstd',
            processes=processes,
            args_str='Namespace()',
            reprocess=False,
        )

    call_convert_text_to_mds(processes=processes)

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
                        files=['index.json', DONE_FILENAME] + shards)

    call_convert_text_to_mds(processes=processes)

    # Check call counts
    assert download_and_convert.call_count == processes  # No changes because we shoudn't reprocess
    assert mock_object_store.download_object.call_count == n_text_files + 2  # One more done file is downloaded
    assert mock_object_store.upload_object.call_count == processes + 2  # No changes

    # Create an extra text file and call again.
    n_text_files += 1
    object_store = MockObjectStore(remote_folder, n_text_files, text_content)
    mock_object_store = Mock(wraps=object_store)
    maybe_create_object_store_from_uri.return_value = mock_object_store

    call_convert_text_to_mds(processes=processes)

    # Check call counts
    assert download_and_convert.call_count == processes * 2  # called once per process
    assert mock_object_store.download_object.call_count == n_text_files + 1  # text files + done file
    assert mock_object_store.upload_object.call_count == processes + 2  # shard per process + done file + index.json

    # Compute the expected number of tokens
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokens_per_file = len(tokenizer(text_content)['input_ids'])
    files_per_process = [n_text_files // processes
                        ] * processes  # Distrubte the files equally
    files_per_process[
        0] += n_text_files % processes  # Give one of the processes the remainder
    # expected number of tokens accounts for last tokens dropped by ConcatTokensDataset
    expected_n_tokens = sum([
        ((n_files * tokens_per_file) // concat_tokens) * concat_tokens
        for n_files in files_per_process
    ])

    dataset = StreamingDataset(local=remote_folder, num_canonical_nodes=1)
    n_tokens = 0
    for i in range(dataset.num_samples):
        sample = dataset[i]
        tokens = np.frombuffer(sample['tokens'], dtype=int)
        if i == 0:  # For the first sample, check that the decoded sample matches the text_content
            decoded = tokenizer.decode(tokens)
            assert decoded == text_content[:len(decoded)]
        n_tokens += len(tokens)

    # Check that the number of tokens found while iterating through the dataset is as expected.
    assert n_tokens == expected_n_tokens


def test_local_path(tmp_path: pathlib.Path):
    # Input/output folders
    input_folder = tmp_path / 'input'
    output_folder = tmp_path / 'output'

    def call_convert_text_to_mds(reprocess: bool):
        convert_text_to_mds(
            tokenizer_name='mosaicml/mpt-7b',
            output_folder=str(output_folder),
            input_folder=str(input_folder),
            concat_tokens=1,
            eos_text='',
            bos_text='',
            no_wrap=False,
            compression='zstd',
            processes=1,
            args_str='Namespace()',
            reprocess=reprocess,
        )

    # Create input text data
    os.makedirs(input_folder, exist_ok=True)
    with open(input_folder / 'test.txt', 'w') as f:
        f.write('test')

    # Convert text data to mds
    call_convert_text_to_mds(reprocess=False)

    # Make sure all the files exist as expected.
    assert os.path.exists(output_folder / '.text_to_mds_conversion_done')
    assert os.path.exists(output_folder / 'index.json')
    assert os.path.exists(output_folder / 'shard.00000.mds.zstd')

    # Test reprocessing.
    with pytest.raises(FileExistsError):
        call_convert_text_to_mds(reprocess=True)

    shutil.rmtree(output_folder)

    call_convert_text_to_mds(reprocess=True)


def test_is_already_processed(tmp_path: pathlib.Path):
    tmp_path_str = str(tmp_path)
    args_str = 'Namespace(x = 5)'
    object_names = ['test0.txt', 'test1.txt']

    assert not is_already_processed(tmp_path_str, args_str,
                                    object_names)  # Done file doesn't exist

    write_done_file(tmp_path_str, args_str, object_names)
    assert is_already_processed(tmp_path_str, args_str,
                                object_names)  # Args and names match

    write_done_file(tmp_path_str, args_str, object_names + ['test2.txt'])
    assert not is_already_processed(tmp_path_str, args_str,
                                    object_names)  # Object names differ

    write_done_file(tmp_path_str, 'Namespace()', object_names)
    assert not is_already_processed(tmp_path_str, args_str,
                                    object_names)  # Argument strings differ
