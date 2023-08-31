# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

# Mock get object names and hash, object store object (upload- mock, no functionality; download - write to given location), remote path s3://fake-test-path-input 

# Test single process (download and convert is called once with all argumnets), multi process (download and convert and mds writer is called n times and with the correct arguments, test to make sure json is correct), 
# Test 1 process and remote directory

import pathlib
from typing import List
from unittest.mock import Mock, patch
import os
from scripts.data_prep.convert_text_to_mds import main, download_and_convert
from streaming import StreamingDataset

from transformers import AutoTokenizer
import numpy as np
from glob import glob

class MockObjectStore():
    def __init__(self, remote_folder: str, n_text_files: int, text_content: str):
        os.makedirs(remote_folder, exist_ok=True)
        for i in range(n_text_files):
            with open(os.path.join(remote_folder, f'test{i}.txt'), 'w') as f:
                f.write((text_content + ' ') * 500)
        self.remote_folder = remote_folder
        self.n_text_files = n_text_files

    def download_object(self, object_name: str, filename: str, overwrite: bool=False):
        with open(os.path.join(self.remote_folder, os.path.basename(object_name)), 'rb') as remote_file, open(filename, 'wb') as local_file:
            local_file.write(remote_file.read())

    def list_objects(self, prefix: str) -> List[str]:
        return glob(os.path.join(self.remote_folder, '*.txt'))
    
    def upload_object(self, object_name: str, filename: str):
        with open(os.path.join(self.remote_folder, os.path.basename(object_name)), 'wb') as remote_file, open(filename, 'rb') as local_file:
            remote_file.write(local_file.read())

def _call_convert_text_to_mds(processes: int, tokenizer_name: str) -> None:
    main(tokenizer_name, f's3://fake-test-output-path', f's3://fake-test-input-path', 
         2048, '', '', False, 1, 'zstd', processes, args_str='Namespace()', reprocess=False)

@patch('scripts.data_prep.convert_text_to_mds.maybe_create_object_store_from_uri')
@patch('scripts.data_prep.convert_text_to_mds.parse_uri')
@patch('scripts.data_prep.convert_text_to_mds.download_and_convert', wraps=download_and_convert)
def test_single_process(download_and_convert: Mock, parse_uri: Mock, maybe_create_object_store_from_uri: Mock, tmp_path: pathlib.Path):
    remote_folder = os.path.join(tmp_path, 'remote')
    n_text_files = 3
    text_content = 'HELLO WORLD'
    tokenizer_name = 'mosaicml/mpt-7b'

    maybe_create_object_store_from_uri.return_value = Mock(name='ObjectStore mock', wraps=MockObjectStore(remote_folder, n_text_files, text_content))
    parse_uri.return_value = ('s3', 'fake-test-bucket', str(remote_folder))

    _call_convert_text_to_mds(processes=1, tokenizer_name=tokenizer_name)

    assert os.path.exists(os.path.join(remote_folder, 'shard.00000.mds.zstd'))

    index_json = os.path.join(remote_folder, 'index.json')
    assert os.path.exists(index_json)

    with open(index_json, 'r') as f:
        assert f.read() == '{"shards": [{"column_encodings": ["bytes"], "column_names": ["tokens"], "column_sizes": [null], "compression": "zstd", "format": "mds", "hashes": [], "raw_data": {"basename": "shard.00000.mds", "bytes": 49359, "hashes": {}}, "samples": 3, "size_limit": 67108864, "version": 2, "zip_data": {"basename": "shard.00000.mds.zstd", "bytes": 227, "hashes": {}}}], "version": 2}'

    download_and_convert.assert_called_once()

    _call_convert_text_to_mds(processes=1, tokenizer_name=tokenizer_name)
 
    download_and_convert.assert_called_once() 

    # Create an extra text file and call again.
    maybe_create_object_store_from_uri.return_value = Mock(name='ObjectStore mock', wraps=MockObjectStore(remote_folder, n_text_files + 1, text_content))
    _call_convert_text_to_mds(processes=1, tokenizer_name=tokenizer_name)

    assert download_and_convert.call_count == 2

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    dataset = StreamingDataset(local=remote_folder)
    sample = dataset.__iter__().__next__()
    assert tokenizer.decode(np.frombuffer(sample['tokens'], dtype=int)).startswith(text_content)
