# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import logging
import os
import re
from typing import Optional

import composer.utils as utils
import numpy as np

log = logging.getLogger(__name__)

DELTA_JSONL_REGEX = re.compile(r'^tmp-t$')
REMOTE_OBJECT_STORE_FILE_REGEX = re.compile(
    r'^((s3|oci|gs):\/\/|dbfs:\/Volumes\/)[/a-zA-Z0-9 ()_\-.]+$',
)


def get_dataset_format(data_path_folder: str) -> str:
    """Determine the format of the dataset from the provided data path.

    Args:
        data_path_folder (str): Path to the training dataset folder

    Returns:
        str: The format of the dataset
    """
    if DELTA_JSONL_REGEX.match(data_path_folder):
        return 'delta'
    if REMOTE_OBJECT_STORE_FILE_REGEX.match(data_path_folder):
        return 'remote_object_store'
    return 'unknown'


TEMP_DIR = 'tmp-split'


def maybe_download_data_as_json(
    data_path_folder: str, data_path_split: str
) -> str:
    """Prepares dataset as a local JSONL file. Downloads from remote object
    store if necessary.

    This function is intended to be invoked by DBX Finetuning.
    Thus, it assumes the provided data is:
        1. A Delta table converted to JSONL at 'tmp-t/{data_path_split}-00000-of-00001.jsonl`
           using the 'llmfoundry.scripts.convert_delta_to_json.py' script.
        2. A JSONL stored as a remote object store file (e.g. S3, OCI, GCS)

    Args:
        data_path_folder (str): Path to the training dataset folder
        data_path_split (str): Data split

    Returns:
        str: Path to the training dataset
    """
    os.makedirs(TEMP_DIR, exist_ok=True)

    dataset_format = get_dataset_format(data_path_folder)

    if dataset_format == 'delta':
        log.info(
            f'Dataset is converted from Delta table. Using local file {data_path_folder}'
        )
        data_path = os.path.join(
            data_path_folder, f'{data_path_split}-00000-of-00001.jsonl'
        )

    elif dataset_format == 'remote_object_store':
        log.info(
            f'Downloading dataset from remote object store: {data_path_folder}{data_path_split}.jsonl',
        )
        remote_path = f'{data_path_folder}/{data_path_split}.jsonl'
        data_path = os.path.join(TEMP_DIR, f'{data_path_split}.jsonl')
        utils.get_file(remote_path, data_path, overwrite=True)

    else:
        raise ValueError(
            f'Encountered unknown data path format when splitting dataset: {data_path_folder} with split {data_path_split}',
        )

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f'Expected dataset file at {data_path} for splitting, but it does not exist.',
        )

    return data_path


@contextlib.contextmanager
def temp_seed(seed: int):
    log.info(f'Setting random seed to {seed}')
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def split_examples(
    data_path: str,
    output_path: str,
    eval_split_ratio: float,
    max_eval_samples: Optional[int] = None,
    seed: Optional[int] = None,
) -> None:
    """Splits the dataset into training and evaluation sets.

    Args:
        data_path (str): Path to the training dataset (local jsonl file)
        output_path (str): Directory to save the split dataset
        eval_split_ratio (float): Ratio of the dataset to use for evaluation. The remainder will be used for training
        max_eval_samples (int): Maximum number of samples to include in the eval set. If None, all eval_split_ratio * train_dataset_size samples will be used
        seed (int): Random seed for splitting the dataset
    """
    os.makedirs(output_path, exist_ok=True)

    # first pass: count total number of lines and determine sample size
    total_lines = 0
    with open(data_path, 'r') as infile:
        for _ in infile:
            total_lines += 1
    sample_size = int(eval_split_ratio * total_lines)
    if max_eval_samples is not None:
        sample_size = min(sample_size, max_eval_samples)

    with temp_seed(seed) if seed is not None else contextlib.nullcontext():
        random_numbers = np.random.rand(total_lines)
        sample_indices = set(np.argsort(random_numbers)[:sample_size])

    # second pass: sample indices
    with open(data_path, 'r') as infile, open(
        os.path.join(output_path, 'train.jsonl'),
        'w',
    ) as train_outfile, open(
        os.path.join(output_path, 'eval.jsonl'), 'w'
    ) as eval_outfile:
        for idx, line in enumerate(infile):
            if idx in sample_indices:
                eval_outfile.write(line)
            else:
                train_outfile.write(line)

    log.info(
        f'Split {data_path} into train set of size {total_lines - sample_size} and eval set of size {sample_size}.',
    )


def split_eval_set_from_args(
    data_path_folder: str,
    data_path_split: str,
    output_path: str,
    eval_split_ratio: float,
    max_eval_samples: Optional[int] = None,
    seed: Optional[int] = None,
) -> None:
    """A wrapper for split_eval_set that parses arguments.

    Args:
        data_path_folder (str): Path to the training dataset folder
        data_path_split (str): Data split
        output_path (str): Directory to save the split dataset
        eval_split_ratio (float): Ratio of the dataset to use for evaluation. The remainder will be used for training
        max_eval_samples (int): Maximum number of samples to include in the eval set. If None, all eval_split_ratio * train_dataset_size samples will be used
        seed (int): Random seed for splitting the dataset
    """
    data_path = maybe_download_data_as_json(data_path_folder, data_path_split)
    split_examples(
        data_path, output_path, eval_split_ratio, max_eval_samples, seed
    )
