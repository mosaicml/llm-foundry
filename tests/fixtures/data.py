# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
import os
import random
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import datasets as hf_datasets
import numpy as np
from composer.utils import dist
from omegaconf import DictConfig
from pytest import fixture
from streaming import MDSWriter
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from llmfoundry.data.finetuning.dataloader import build_finetuning_dataloader
from tests.data_utils import make_tiny_ft_dataset


@fixture()
def tiny_text_dataset_path(tmp_path: Path) -> Path:
    rng = random.Random(42)
    out_dir = tmp_path / 'test-text-data'
    columns = {'tokens': 'ndarray:int32'}
    for split in ['train', 'val']:
        with MDSWriter(
            columns=columns,
            out=os.path.join(out_dir, split),
            compression=None,
        ) as out:
            for _ in range(100):
                tokens = np.array(
                    rng.sample(range(0, 100), 100),
                    dtype=np.int32,
                )
                sample = {
                    'tokens': tokens,
                }
                out.write(sample)

    return out_dir


@fixture
def tiny_text_hf_dataset():
    assets_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'assets',
    )
    text_data_path = os.path.join(assets_dir, 'text_data.jsonl')

    ds = hf_datasets.load_dataset(
        'json',
        data_files=text_data_path,
        split='train',
    )

    return ds


@fixture
def tiny_ft_dataset_path(tmp_path: Path, dataset_size: int = 4) -> Path:
    """Creates a tiny dataset and returns the path."""
    tiny_dataset_path = tmp_path / 'test-ift-data-small'
    tiny_dataset_path.mkdir(exist_ok=True)
    tiny_dataset_file = tiny_dataset_path / 'train.jsonl'
    if dist.get_world_size() == 1 or dist.get_global_rank() == 0:
        make_tiny_ft_dataset(path=str(tiny_dataset_file), size=dataset_size)
    return tiny_dataset_path


@fixture
def tiny_ft_dataloader_cfg(
    tiny_ft_dataset_path: Path,
    max_seq_len: int = 128,
) -> dict[str, Any]:
    return {
        'dataset': {
            'hf_name': str(tiny_ft_dataset_path),
            'split': 'train',
            'max_seq_len': max_seq_len,
            'decoder_only_format': True,
            'allow_pad_trimming': False,
            'packing_ratio': None,
            'shuffle': True,
        },
        'drop_last': False,
        'num_workers': 4,
        'pin_memory': False,
        'prefetch_factor': 2,
        'persistent_workers': False,
        'timeout': 0,
    }


@fixture
@patch('os.cpu_count', MagicMock(return_value=1))
def tiny_ft_dataloader(
    mpt_tokenizer: PreTrainedTokenizerBase,
    tiny_ft_dataloader_cfg: dict[str, Any],
    device_batch_size: int = 1,
) -> DataLoader:
    dataloader_cfg = DictConfig(tiny_ft_dataloader_cfg)

    dataloader = build_finetuning_dataloader(
        **dataloader_cfg,
        tokenizer=mpt_tokenizer,
        device_batch_size=device_batch_size,
    ).dataloader

    assert isinstance(dataloader, DataLoader)
    return dataloader
