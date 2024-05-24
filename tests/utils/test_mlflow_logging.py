# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from llmfoundry.utils.config_utils import (
    log_dataset_uri,
    _parse_source_dataset,
)

mlflow = pytest.importorskip('mlflow')
from mlflow.data.huggingface_dataset_source import HuggingFaceDatasetSource


def create_config(**kwargs: Any):
    """Helper function to create OmegaConf configurations."""
    return kwargs


def test_parse_source_dataset_delta_table():
    cfg = create_config(
        source_dataset_train='db.schema.train_table',
        source_dataset_eval='db.schema.eval_table',
    )
    expected = [('delta_table', 'db.schema.train_table', 'train'),
                ('delta_table', 'db.schema.eval_table', 'eval')]
    assert _parse_source_dataset(cfg) == expected


def test_parse_source_dataset_uc_volume():
    cfg = create_config(
        source_dataset_train='dbfs:/Volumes/train_data',
        source_dataset_eval='dbfs:/Volumes/eval_data',
    )
    expected = [('uc_volume', '/Volumes/train_data', 'train'),
                ('uc_volume', '/Volumes/eval_data', 'eval')]
    assert _parse_source_dataset(cfg) == expected


def test_parse_source_dataset_hf():
    cfg = create_config(
        train_loader={'dataset': {
            'hf_name': 'huggingface/train_dataset',
        }},
        eval_loader={'dataset': {
            'hf_name': 'huggingface/eval_dataset',
        }},
    )
    expected = [('hf', 'huggingface/train_dataset', 'train'),
                ('hf', 'huggingface/eval_dataset', 'eval')]
    assert _parse_source_dataset(cfg) == expected


def test_parse_source_dataset_remote():
    cfg = create_config(
        train_loader={
            'dataset': {
                'remote': 'https://remote/train_dataset',
                'split': 'train',
            },
        },
        eval_loader={
            'dataset': {
                'remote': 'https://remote/eval_dataset',
                'split': 'eval',
            },
        },
    )
    expected = [('https', 'https://remote/train_dataset/train/', 'train'),
                ('https', 'https://remote/eval_dataset/eval/', 'eval')]
    assert _parse_source_dataset(cfg) == expected


def test_log_dataset_uri():
    cfg = create_config(
        train_loader={'dataset': {
            'hf_name': 'huggingface/train_dataset',
        }},
        eval_loader={'dataset': {
            'hf_name': 'huggingface/eval_dataset',
        }},
        source_dataset_train='huggingface/train_dataset',
        source_dataset_eval='huggingface/eval_dataset',
    )

    with patch('mlflow.log_input') as mock_log_input:
        log_dataset_uri(cfg)
        assert mock_log_input.call_count == 2
        meta_dataset_calls = [
            args[0] for args, _ in mock_log_input.call_args_list
        ]
        assert all(
            isinstance(call.source, HuggingFaceDatasetSource)
            for call in meta_dataset_calls
        ), 'Source types are incorrect'
        # Verify the names
        assert meta_dataset_calls[
            0
        ].name == 'train', f"Expected 'train', got {meta_dataset_calls[0].name}"
        assert meta_dataset_calls[
            1
        ].name == 'eval', f"Expected 'eval', got {meta_dataset_calls[1].name}"


def test_multiple_eval_datasets():
    # Setup a configuration with multiple evaluation datasets
    cfg = {
        'train_loader': {
            'dataset': {
                'hf_name': 'huggingface/train_dataset',
            },
        },
        'eval_loader': [{
            'dataset': {
                'hf_name': 'huggingface/eval_dataset1',
            },
        }, {
            'dataset': {
                'hf_name': 'huggingface/eval_dataset2',
            },
        }],
    }

    expected_data_paths = [('hf', 'huggingface/train_dataset', 'train'),
                           ('hf', 'huggingface/eval_dataset1', 'eval'),
                           ('hf', 'huggingface/eval_dataset2', 'eval')]

    # Mock mlflow to avoid any actual logging calls
    with patch('mlflow.data.meta_dataset.MetaDataset') as mock_meta_dataset:
        mock_meta_dataset.side_effect = lambda source, name: MagicMock()
        data_paths = _parse_source_dataset(cfg)
        assert sorted(data_paths) == sorted(
            expected_data_paths,
        ), 'Data paths did not match expected'


@pytest.fixture
def mock_mlflow_classes():
    with patch('mlflow.data.http_dataset_source.HTTPDatasetSource') as http_source, \
         patch('mlflow.data.huggingface_dataset_source.HuggingFaceDatasetSource') as hf_source, \
         patch('mlflow.data.delta_dataset_source.DeltaDatasetSource') as delta_source, \
         patch('mlflow.data.uc_volume_dataset_source.UCVolumeDatasetSource') as uc_source:
        yield http_source, hf_source, delta_source, uc_source
