# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import MagicMock, patch

from composer.utils import dist
from omegaconf import DictConfig
from pytest import fixture
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from llmfoundry.data.finetuning.dataloader import build_finetuning_dataloader
from tests.data_utils import make_tiny_ft_dataset


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
@patch('os.cpu_count', MagicMock(return_value=1))
def tiny_ft_dataloader(tiny_ft_dataset_path: Path,
                       mpt_tokenizer: PreTrainedTokenizerBase,
                       max_seq_len: int = 128,
                       device_batch_size: int = 1) -> DataLoader:
    dataloader_cfg = DictConfig({
        'name': 'finetuning',
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
        'timeout': 0
    })

    dataloader = build_finetuning_dataloader(
        **dataloader_cfg,
        tokenizer=mpt_tokenizer,
        device_batch_size=device_batch_size,
    ).dataloader

    assert isinstance(dataloader, DataLoader)
    return dataloader
