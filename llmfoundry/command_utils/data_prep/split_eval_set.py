# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import re
import json
import contextlib
import datasets as hf_datasets
import numpy as np
from typing import Optional

from composer.utils import get_file
from llmfoundry.data.finetuning.tasks import maybe_safe_download_hf_data


DELTA_JSONL_REGEX = re.compile(r"^tmp-t$")
REMOTE_OBJECT_STORE_FILE_REGEX = re.compile(
    r"^((s3|oci|gs):\/\/|dbfs:\/Volumes\/)[/a-zA-Z0-9 ()_\-.]+$"
)
HF_REGEX = re.compile(r"^[/a-zA-Z0-9 ()_\-.]+$")

TEMP_DIR = "tmp-split"

log = logging.getLogger(__name__)

import sys

log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler(sys.stdout))


def maybe_download_data_as_json(data_path_folder: str, data_path_split: str) -> str:
    """
    Prepares dataset as a local JSONL file. Downloads from remote object store or HF if necessary.

    This function is intended to be invoked by DBX Finetuning.
    Thus, it assumes the provided data is in one of three formats:
        1. A Delta table converted to JSONL at 'tmp-t/{data_path_split}-00000-of-00001.jsonl`
           using the 'llmfoundry.scripts.convert_delta_to_json.py' script.
        2. A JSONL stored as a remote object store file (e.g. S3, OCI, GCS)
        3. A Hugging Face dataset

    Args:
        data_path_folder (str): Path to the training dataset folder
        data_path_split (str): Data split

    Returns:
        str: Path to the training dataset
    """
    os.makedirs(TEMP_DIR, exist_ok=True)

    if DELTA_JSONL_REGEX.match(data_path_folder):
        data_path = os.path.join(data_path_folder, f"{data_path_split}-00000-of-00001.jsonl")
        if not os.path.exists(data_path):
            # TODO: error handling
            raise FileNotFoundError(f"File {data_path} does not exist.")

    if REMOTE_OBJECT_STORE_FILE_REGEX.match(data_path_folder):
        log.info(
            f"Downloading dataset from remote object store: {data_path_folder}{data_path_split}.jsonl"
        )
        remote_path = f"{data_path_folder}/{data_path_split}.jsonl"
        data_path = os.path.join(TEMP_DIR, f"{data_path_split}.jsonl")
        try:
            get_file(remote_path, data_path, overwrite=True)
        except FileNotFoundError as e:
            # TODO: error handling
            raise e

    elif HF_REGEX.match(data_path_folder):
        log.info(
            f"Downloading dataset from Hugging Face: {data_path_folder} with split {data_path_split}"
        )
        # TODO: maybe add support for HF kwargs
        local_hf_path = maybe_safe_download_hf_data(data_path_folder)
        # convert dataset split to JSONL
        dataset = hf_datasets.load_dataset(
            local_hf_path,
            split=data_path_split,
        )
        data_path = os.path.join(TEMP_DIR, f"{data_path_split}.jsonl")
        with open(data_path, "w") as f:
            for example in dataset:
                f.write(json.dumps(example) + "\n")

    else:
        # TODO: error handling
        raise ValueError(
            f"Unrecognized data_path_folder: {data_path_folder}. Must be a Delta table, remote object store file, or Hugging Face dataset."
        )

    if not os.path.exists(data_path):
        # TODO: error handling
        raise FileNotFoundError(f"File {data_path} does not exist.")

    return data_path


@contextlib.contextmanager
def temp_seed(seed: int):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def _split_examples(
    data_path: str,
    output_path: str,
    eval_split_ratio: float,
    max_eval_samples: Optional[int],
    seed: Optional[int] = None,
) -> None:
    """
    Splits the dataset into training and evaluation sets.

    Args:
        data_path (str): Path to the training dataset (local jsonl file)
        eval_split_ratio (float): Ratio of the dataset to use for evaluation. The remainder will be used for training
        max_eval_samples (int): Maximum number of samples to include in the eval set. If None, all eval_split_ratio * train_dataset_size samples will be used
        seed (int): Random seed for splitting the dataset
    """
    # first pass: count total number of lines and determine sample size
    total_lines = 0
    with open(data_path, "r") as infile:
        for _ in infile:
            total_lines += 1
    sample_size = int(eval_split_ratio * total_lines)
    if max_eval_samples is not None:
        sample_size = min(sample_size, max_eval_samples)

    with temp_seed(seed) if seed is not None else contextlib.nullcontext():
        random_numbers = np.random.rand(total_lines)
        sample_indices = set(np.argsort(random_numbers)[:sample_size])

    # second pass: sample indices
    with open(data_path, "r") as infile, open(
        os.path.join(output_path, "train.jsonl"), "w"
    ) as train_outfile, open(os.path.join(output_path, "eval.jsonl"), "w") as eval_outfile:
        for idx, line in enumerate(infile):
            if idx in sample_indices:
                eval_outfile.write(line)
            else:
                train_outfile.write(line)

    log.info(
        f"Split {data_path} into train set of size {total_lines - sample_size} and eval set of size {sample_size}."
    )


def split_eval_set_from_args(
    data_path_folder: str,
    data_path_split: str,
    output_path: str,
    eval_split_ratio: float,
    max_eval_samples: Optional[int] = None,
    seed: Optional[int] = None,
) -> None:
    """
    A wrapper for split_eval_set that parses arguments

    Args:
        data_path_folder (str): Path to the training dataset folder
        data_path_split (str): Data split
        output_path (str): Directory to save the split dataset
        eval_split_ratio (float): Ratio of the dataset to use for evaluation. The remainder will be used for training
        max_eval_samples (int): Maximum number of samples to include in the eval set. If None, all eval_split_ratio * train_dataset_size samples will be used
        seed (int): Random seed for splitting the dataset
    """
    os.makedirs(output_path, exist_ok=True)
    data_path = maybe_download_data_as_json(data_path_folder, data_path_split)
    _split_examples(data_path, output_path, eval_split_ratio, max_eval_samples, seed)
