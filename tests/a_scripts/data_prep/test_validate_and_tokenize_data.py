# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import Mock, patch, MagicMock, mock_open
from argparse import Namespace
from scripts.data_prep.validate_and_tokenize_data import integrity_check, check_HF_datasets, is_hf_dataset_path, create_om_cfg
from streaming.base.storage.upload import CloudUploader
from transformers import AutoTokenizer

class MockCloudUploader:
    def __init__(self):
        self.remote = "some_remote_path"
        self.local = "some_local_path"

    def list_objects(self):
        return ['shard1.mds', 'shard2.mds']

class MockDatasetInfo:
    def __init__(self):
        self.id = "valid_dataset"
        self.description = "A mock dataset description"

@patch('scripts.data_prep.validate_and_tokenize_data.CloudUploader.get')
@patch('scripts.data_prep.validate_and_tokenize_data.download_file')
@patch('scripts.data_prep.validate_and_tokenize_data.json.load')
@patch('builtins.open', new_callable=mock_open, read_data='{"shards": [{"raw_data": {"basename": "shard1.mds"}}, {"raw_data": {"basename": "shard2.mds"}}]}')
def test_integrity_check(mock_file_open, mock_json_load, mock_download_file, mock_cloud_uploader):
    # Setup mocks
    mock_cloud_uploader.return_value = MockCloudUploader()
    mock_json_load.return_value = {'shards': [{'raw_data': {'basename': 'shard1.mds'}}, {'raw_data': {'basename': 'shard2.mds'}}]}

    # Test case where integrity is valid
    assert integrity_check('mock_dataset_path')

    # Test case where integrity is invalid
    # Modify the mock to simulate a different scenario
    mock_json_load.return_value = {'shards': [{'raw_data': {'basename': 'shard1.mds'}}]} # less shards than expected
    assert not integrity_check('mock_dataset_path')

# Additional tests can be written for cases like remote URL, file not found, etc.



@patch('scripts.data_prep.validate_and_tokenize_data.dataset_info')
@patch('scripts.data_prep.validate_and_tokenize_data.get_dataset_split_names')
def test_check_HF_datasets(mock_get_splits, mock_dataset_info):
    # Setup mocks
    mock_get_splits.return_value = ['train', 'test']
    mock_dataset_info.return_value = MockDatasetInfo()

    # Test valid dataset with valid split
    result, message = check_HF_datasets(['valid_dataset/train'])
    assert result

    # Test valid dataset with invalid split
    result, message = check_HF_datasets(['valid_dataset/invalid_split'])
    assert not result

    # Test invalid dataset
    mock_dataset_info.side_effect = Exception("Dataset not found")
    result, message = check_HF_datasets(['invalid_dataset/train'])
    assert not result

# Additional tests for private datasets, token issues, etc.



def test_is_hf_dataset_path():
    # Valid dataset paths
    assert is_hf_dataset_path('user/dataset/train')
    assert is_hf_dataset_path('user/dataset')

    # Invalid dataset paths
    assert not is_hf_dataset_path('user@dataset/train')
    assert not is_hf_dataset_path('just_dataset_name')
    assert not is_hf_dataset_path('user/dataset/unknown_split/')


@patch('transformers.AutoTokenizer.from_pretrained')
def test_create_om_cfg_instruction_finetune(mock_from_pretrained):
    mock_from_pretrained.return_value = MagicMock(spec=AutoTokenizer)
    args = Namespace(
        task_type='INSTRUCTION_FINETUNE',
        train_data_path='hf_dataset/train',
        model='model_name',
        context_length=512
    )
    cfg, tokenizer = create_om_cfg(args)
    assert cfg.dataset.hf_name == 'hf_dataset/train'
    assert cfg.dataset.max_seq_len == 512

@patch('transformers.AutoTokenizer.from_pretrained')
def test_create_om_cfg_continued_pretrain(mock_from_pretrained):
    mock_from_pretrained.return_value = MagicMock(spec=AutoTokenizer)
    args = Namespace(
        task_type='CONTINUED_PRETRAIN',
        train_data_path='object_store_path',
        model='model_name',
        context_length=512
    )
    cfg, tokenizer = create_om_cfg(args)
    assert cfg.dataset.remote == 'object_store_path'
    assert cfg.dataset.max_seq_len == 512

