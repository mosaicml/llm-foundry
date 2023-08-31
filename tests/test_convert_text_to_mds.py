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


def _call_convert_text_to_mds(processes: int) -> None:
    main('mosaicml/mpt-7b', f's3://fake-test-output-path', f's3://fake-test-input-path', 
         2048, '', '', False, 1, 'zstd', processes, args_str='Namespace()', reprocess=False)

@patch('scripts.data_prep.convert_text_to_mds.maybe_create_object_store_from_uri')
@patch('scripts.data_prep.convert_text_to_mds.parse_uri')
@patch('scripts.data_prep.convert_text_to_mds.download_and_convert', wraps=download_and_convert)
def test_single_process(download_and_convert: Mock, parse_uri: Mock, maybe_create_object_store_from_uri: Mock, tmp_path: pathlib.Path):
    print('\ntmp path!', tmp_path)
    remote_folder = os.path.join(tmp_path, 'remote')
    os.makedirs(remote_folder)
    for i in range(3):
        with open(os.path.join(remote_folder, f'test{i}.txt'), 'w') as f:
            f.write('TEST CONTENTS!!!' * 500)
            print('writing!', os.path.join(remote_folder, f'test{i}.txt'))

    def download_object(object_name: str, filename: str, overwrite: bool=False):
        print('download_object', object_name, '\n\t', filename)
        with open(os.path.join(remote_folder, os.path.basename(object_name)), 'rb') as remote_file, open(filename, 'wb') as local_file:
            local_file.write(remote_file.read())
    def list_objects(prefix: str) -> List[str]:
        return [os.path.join(prefix, f'test{i}.txt') for i in range(3)]
    def upload_object(object_name: str, filename: str):
        with open(os.path.join(remote_folder, os.path.basename(object_name)), 'wb') as remote_file, open(filename, 'rb') as local_file:
            remote_file.write(local_file.read())
    object_store = Mock(name='ObjectStore mock')
    object_store.download_object = download_object
    object_store.list_objects = list_objects
    object_store.upload_object = upload_object

    maybe_create_object_store_from_uri.return_value = object_store
    parse_uri.return_value = ('s3', 'fake-test-bucket', str(tmp_path))

    _call_convert_text_to_mds(processes=1)

    _call_convert_text_to_mds(processes=1)
    download_and_convert.assert_called_once()

    print(os.listdir(remote_folder))
