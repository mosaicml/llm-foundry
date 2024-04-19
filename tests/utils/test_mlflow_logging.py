# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
from typing import Any
from omegaconf import OmegaConf

from llmfoundry.utils.config_utils import log_dataset_uri, parse_source_dataset

mlflow = pytest.importorskip('mlflow')


def create_config(**kwargs: Any):
    """Helper function to create OmegaConf configurations."""
    return OmegaConf.create(kwargs)


def test_parse_source_dataset_delta_table():
    cfg = create_config(source_dataset_train='db.schema.train_table',
                        source_dataset_eval='db.schema.eval_table')
    expected = {('delta_table', 'db.schema.train_table', 'train'),
                ('delta_table', 'db.schema.eval_table', 'eval')}
    assert parse_source_dataset(cfg) == expected


def test_parse_source_dataset_uc_volume():
    cfg = create_config(source_dataset_train='/Volumes/train_data',
                        source_dataset_eval='/Volumes/eval_data')
    expected = {('uc_volume', '/Volumes/train_data', 'train'),
                ('uc_volume', '/Volumes/eval_data', 'eval')}
    assert parse_source_dataset(cfg) == expected


def test_parse_source_dataset_hf():
    cfg = create_config(
        train_loader={'dataset': {
            'hf_name': 'huggingface/train_dataset'
        }},
        eval_loader={'dataset': {
            'hf_name': 'huggingface/eval_dataset'
        }})
    expected = {('hf', 'huggingface/train_dataset', 'train'),
                ('hf', 'huggingface/eval_dataset', 'eval')}
    assert parse_source_dataset(cfg) == expected


def test_parse_source_dataset_remote():
    cfg = create_config(
        train_loader={'dataset': {
            'remote': 'https://remote/train_dataset'
        }},
        eval_loader={'dataset': {
            'remote': 'https://remote/eval_dataset'
        }})
    expected = {('https', 'https://remote/train_dataset', 'train'),
                ('https', 'https://remote/eval_dataset', 'eval')}
    assert parse_source_dataset(cfg) == expected


def test_parse_source_dataset_local():
    cfg = create_config(
        train_loader={'dataset': {
            'local': '/local/train_dataset'
        }},
        eval_loader={'dataset': {
            'local': '/local/eval_dataset'
        }})
    expected = {('local', '/local/train_dataset', 'train'),
                ('local', '/local/eval_dataset', 'eval')}
    assert parse_source_dataset(cfg) == expected


@pytest.mark.usefixtures('mock_mlflow_classes')
def test_log_dataset_uri_all_sources():
    cfg = create_config(
        train_loader={'dataset': {
            'hf_name': 'huggingface/train_dataset'
        }},
        eval_loader={'dataset': {
            'hf_name': 'huggingface/eval_dataset'
        }},
        source_dataset_train='db.schema.train_table',
        source_dataset_eval='/Volumes/eval_data')

    with patch('mlflow.data.meta_dataset.MetaDataset'):
        with patch('mlflow.log_input') as mock_log_input:
            log_dataset_uri(cfg)
            assert mock_log_input.call_count == 2


@pytest.fixture
def mock_mlflow_classes():
    with patch('mlflow.data.http_dataset_source.HTTPDatasetSource') as http_source, \
         patch('mlflow.data.huggingface_dataset_source.HuggingFaceDatasetSource') as hf_source, \
         patch('mlflow.data.delta_dataset_source.DeltaDatasetSource') as delta_source, \
         patch('mlflow.data.uc_volume_dataset_source.UCVolumeDatasetSource') as uc_source:
        yield http_source, hf_source, delta_source, uc_source
