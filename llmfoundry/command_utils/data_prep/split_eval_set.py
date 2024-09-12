# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import os
import json
from enum import Enum

import datasets
from llmfoundry.data.finetuning.tasks import download_hf_dataset_if_needed


class SupportedDataFormats(Enum):
    REMOTE_JSONL = "jsonl"  # UC JSONL
    DELTA_JSONL = "delta_jsonl"  # Delta table preprocessed to JSONL
    HF = "huggingface"


def validate_data_path(data_path: str) -> None:
    """
    Validates the data path and returns the format of the data.

    Args:
        data_path (str): Path to the training dataset
    """


def split_eval_set_from_args() -> None:
    """
    Args:
        data_path_folder (str): Path to the training dataset folder
        data_path_split (str): Data split
        output_path (str): Directory to save the split dataset
        eval_split_ratio (float): Ratio of the dataset to use for evaluation. The remainder will be used for training
        max_eval_samples (int): Maximum number of samples to include in the eval set. If None, all eval_split_ratio * train_dataset_size samples will be used
        seed (int): Random seed for splitting the dataset
    """
    pass
