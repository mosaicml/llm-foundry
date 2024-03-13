# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from argparse import Namespace
from unittest.mock import patch, MagicMock
from llmfoundry.utils.data_validation_utils import integrity_check, is_uc_delta_table, is_hf_dataset_path, token_counts, create_om_cfg

@patch('llmfoundry.utils.data_validation_utils.CloudUploader')
@patch('llmfoundry.utils.data_validation_utils.download_file')
@patch('llmfoundry.utils.data_validation_utils.os.path.join')
@patch('llmfoundry.utils.data_validation_utils.json.load')
@patch('llmfoundry.utils.data_validation_utils.tempfile.TemporaryDirectory')
def test_remote_file_path_integrity_passes(MockTempDir,
                                           MockJsonLoad,
                                           MockOsPathJoin,
                                           MockDownloadFile,
                                           MockCloudUploader):
    MockCloudUploader.get.return_value = MagicMock(remote='remote_path', local=None, list_objects=lambda: ['file1.mds', 'file2.mds'])
    MockJsonLoad.return_value = {'shards': [{'raw_data': {'basename': 'file1.mds'}}, {'raw_data': {'basename': 'file2.mds'}}]}

    # Assuming the function to be tested is named integrity_check
    result = integrity_check('remote_dataset_path')

    assert(result)


def test_is_uc_delta_table_valid():
    # Test valid UC delta table formats
    valid_names = [
        'catalog.scheme.tablename', 'database.schema.table', 'mycatalog.myschema.mytable'
    ]
    for name in valid_names:
        assert(is_uc_delta_table(name))

def test_is_uc_delta_table_invalid():
    # Test invalid UC delta table formats
    invalid_names = ['folder/file/table', 'catalog.schema', 'catalog.schema.table.extra']
    for name in invalid_names:
        assert(not is_uc_delta_table(name))

def test_is_hf_dataset_path_valid():
    # Test valid Hugging Face dataset paths
    valid_paths = [
        'dataset_name/split_name/train', 'another_dataset_name/another_split_name/validation',
        'dataset_name/with_no_split_name'
    ]
    for path in valid_paths:
        assert(is_hf_dataset_path(path))

def test_is_hf_dataset_path_with_split():
    # Test valid Hugging Face dataset paths
    valid_paths = [
        'dataset_name/split_name/train',
        'another_dataset_name/another_split_name/validation',
    ]
    for path in valid_paths:
        assert is_hf_dataset_path(path) == (True, False)

    in_valid_paths = [
        'dataset_name/split_name/',
        'another_dataset_name/validation',
    ]
    for path in in_valid_paths:
        print('path = ', path)
        assert is_hf_dataset_path(path) == (True, False)

def test_is_hf_dataset_path_invalid():
    # Test invalid Hugging Face dataset paths
    invalid_paths = [
        'dataset_name/split_name/extra_component', 'invalid_dataset_name/', 'no_dataset_name'
    ]
    for path in invalid_paths:
        assert is_hf_dataset_path(path) == (False, False)

def test_is_hf_dataset_path_mixed():
    # Test invalid Hugging Face dataset paths
    url = 'https://huggingface.co/datasets/test_prefix/test_submix/additional/segments'
    mixed_paths = ['dataset_name/split_name/train', 'invalid_dataset_name/', url]
    result = []
    for path in mixed_paths:
        result.append(is_hf_dataset_path(path))

    assert result == [(True, False), (False, False), (False, True)]


@patch('llmfoundry.utils.data_validation_utils.build_finetuning_dataloader')
@patch('os.cpu_count')
def test_token_counts(mock_cpu_count, mock_build_finetuning_dataloader):
    # Mocking the dependencies
    mock_cpu_count.return_value = 4  # Simulate a 4-core CPU
    mock_dataloader = MagicMock()
    mock_dataloader.dataset.map.return_value = [10, 20]  # Mocking expected token lengths
    mock_build_finetuning_dataloader.return_value = MagicMock(dataloader=mock_dataloader)

    # Simulate input arguments
    FT_API_args = Namespace(
        model= 'mosaicml/mpt-7b', # Other examples: 'EleutherAI/gpt-neox-20b',
        train_data_path= 'main.streaming.random_large_table', # Other examples: 'tatsu-lab/alpaca/train', # '/Volumes/main/mosaic_hackathon/managed-volume/IFT/train.jsonl'  # 'mosaicml/dolly_hhrlhf/train'
        task_type='INSTRUCTION_FINETUNE',
        training_duration=3,
        context_length=2048,
    )

    # Calling the function under test
    result = token_counts(FT_API_args)

    # Asserting the expected outcome
    assert result == [10, 20]
