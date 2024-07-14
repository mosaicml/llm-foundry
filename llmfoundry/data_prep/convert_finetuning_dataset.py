# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import platform
import warnings
from typing import Any, Callable, Dict, Iterable, Optional, Union

import datasets as hf_datasets
import psutil
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from streaming import MDSWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from llmfoundry.data.finetuning.collator import validate_target_settings
from llmfoundry.data.finetuning.tasks import (
    _get_example_type,
    dataset_constructor,
    is_valid_ift_example,
    tokenize_formatted_example,
)
from llmfoundry.utils.builders import build_tokenizer

HFDataset = Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]


def build_dataloader(
    dataset: HFDataset,
    batch_size: int,
    num_workers: Optional[int] = None,
) -> DataLoader:
    if num_workers is None:
        # Multiple workers is only supported on linux machines
        if 'linux' in platform.platform().lower():
            num_workers = max(1, psutil.cpu_count())
        else:
            num_workers = 0

    # If using multiple workers, configure each worker to prefetch as many samples as it can, up to
    # the aggregate device batch size
    # If not using workers, the torch DataLoader expects the default value for prefetch_factor,
    # which non-intuitively must be 2.
    # If on macOS, PyTorch requires prefetch_factor set to None since num_workers is always zero
    if 'macos' in platform.platform().lower() and num_workers == 0:
        prefetch_factor = None
    else:
        prefetch_factor = max(
            1,
            2 * batch_size // num_workers,
        ) if num_workers > 0 else 2

    return DataLoader(
        dataset=dataset,
        sampler=None,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )


def generate_samples(
    loader: DataLoader,
    truncate_num_samples: Optional[int] = None,
) -> Iterable[Dict[str, bytes]]:
    """Generator over samples of a dataloader.

    Args:
       loader (DataLoader): A dataloader emitting batches like {key: [sample0_bytes, sample1_bytes, sample2_bytes, ...]}
       truncate_num_samples (Optional[int]): An optional # of samples to stop at.

    Yields:
        Sample dicts.
    """
    n_samples = 0
    for batch in loader:
        keys = list(batch.keys())
        current_bs = len(batch[keys[0]])
        for idx in range(current_bs):
            if truncate_num_samples is not None and n_samples == truncate_num_samples:
                return
            n_samples += 1
            yield {k: v[idx] for k, v in batch.items()}


def get_columns_and_format(
    dataset: HFDataset,
    tokenizing: bool,
    preprocessing_fn: Callable,
):
    ex = preprocessing_fn(next(iter(dataset)))
    example_type = _get_example_type(ex)
    if tokenizing:
        return {'turns': 'json'}, example_type
    if example_type == 'chat':
        # Chat format
        return {'messages': 'json'}, example_type
    else:
        # Prompt-response format
        return {'prompt': 'str', 'response': 'str'}, example_type


def convert_finetuning_dataset(
    dataset: str,
    data_subset: Optional[str],
    splits: list[str],
    preprocessor: Optional[str],
    data_files: list[str],
    skip_preprocessing: bool,
    out_root: str,
    local: Optional[str],
    compression: Optional[str],
    num_workers: Optional[int],
    tokenizer: Optional[str],
    tokenizer_kwargs: dict[str, Any],
    max_seq_len: int,
    target_prompts: str,
    target_responses: str,
    encoder_decoder: bool,
) -> None:
    """Converts Finetuning datasets to MDS format.
    Args:
        dataset (str): Name of the dataset (e.g., first argument to `datasets.load_dataset`, for jsonl data format, it is `json`).
        data_subset (Optional[str]): Subset of data to use.
        splits (list[str]): Comma-separated list of dataset splits
        preprocessor (Optional[str]): Name or import path of function used to preprocess (reformat) the dataset.
        data_files (list[str]): Data file for each split. Comma-separated.
        skip_preprocessing (bool): Whether to skip preprocessing.
        out_root (str): Root path of output directory where MDS shards will be stored. Can be a remote URI.
        local (Optional[str]): Root path of local directory if you want to keep a local copy when out_root is remote.
        compression (Optional[str]): Name of compression algorithm to use.
        num_workers (Optional[int]): Number of workers.
        tokenizer (Optional[str]): Tokenizer used for processing.
        tokenizer_kwargs (dict[str, Any]): Keyword arguments for tokenizer initialization in JSON format.
        max_seq_len (int): Maximum sequence length.
        target_prompts (str): Policy for when to use prompts as training targets.
        target_responses (str): Policy for which responses to treat as training targets.
        encoder_decoder (bool): Set if the data are intended to be used to train an encoder-decoder model.

    Raises:
        ValueError: If the target settings are invalid.
    """
    if skip_preprocessing:
        preprocessing_fn = lambda x: x  # Just an identity function
    else:
        preprocessor_str = preprocessor
        preprocessing_fn = dataset_constructor.get_preprocessing_fn_from_str(
            preprocessor=preprocessor_str,
            dataset_name=dataset,
        )
        if preprocessing_fn is None:
            raise ValueError(
                '`preprocessor` was not set and no preprocessing function ' +\
                'has been registered for `dataset`. If this was intentional ' +\
                '(e.g., because your dataset is already correctly formatted), ' +\
                'include the "--skip-preprocessing" flag to avoid this error.',
            )

    # Make sure the target settings are valid
    validate_target_settings(
        target_prompts=target_prompts,
        target_responses=target_responses,
        decoder_only_format=not encoder_decoder,
    )

    tokenizer = None
    tokenizer_kwargs = tokenizer_kwargs
    tokenizer_kwargs.update({'model_max_length': max_seq_len})
    if tokenizer:
        tokenizer = build_tokenizer(tokenizer, tokenizer_kwargs)

    for i, split_name in enumerate(splits):
        data_file = None
        if len(data_files) > 0:
            data_file = data_files[i]
        dataset = hf_datasets.load_dataset(
            path=dataset,
            name=data_subset,
            split=split_name,
            data_files=data_file,
            streaming=True,
        )
        # Determine the output columns
        columns, example_type = get_columns_and_format(
            dataset=dataset,
            tokenizing=tokenizer is not None,
            preprocessing_fn=preprocessing_fn,
        )
        # Prepare the iterables
        if example_type == 'chat':
            samples = iter(dataset)
        else:
            loader = build_dataloader(
                dataset=dataset,
                batch_size=512,
                num_workers=num_workers,
            )
            samples = generate_samples(loader)

        # Write samples
        print(f'Converting {split_name} to MDS format...')
        out = os.path.join(out_root, split_name)
        if local is not None:
            out = (os.path.join(local, split_name), out)
            keep_local = True
        else:
            keep_local = False
        with MDSWriter(
            columns=columns,
            out=out,
            compression=compression,
            keep_local=keep_local,
        ) as out:
            examples_removed = 0
            for sample in tqdm(samples, desc=split_name):
                formatted_sample = preprocessing_fn(sample)
                assert isinstance(formatted_sample, dict)

                # Use the _get_example_type utility to confirm that the formatted sample
                # can be interpreted by the tokenization code
                try:
                    example_type = _get_example_type(formatted_sample)
                except Exception as e:
                    raise ValueError(
                        'Encountered an error when checking example for proper formatting. ' +\
                        f'example={formatted_sample}',
                    ) from e
                if tokenizer is not None:
                    sample = tokenize_formatted_example(
                        formatted_sample,
                        tokenizer=tokenizer,
                    )
                    if not is_valid_ift_example(
                        max_seq_len,
                        target_prompts=target_prompts,
                        target_responses=target_responses,
                        decoder_only_format=not encoder_decoder,
                        example=sample,
                    ):
                        examples_removed += 1
                        continue

                    sample_to_write = {'turns': []}
                    for turn in sample['turns']:
                        turn_to_write = {}
                        for key in ['input_ids', 'labels']:
                            turn_to_write[key] = list(turn[key])
                        sample_to_write['turns'].append(turn_to_write)
                    out.write(sample_to_write)
                else:
                    if example_type == 'prompt_response':
                        encoded_sample = {}
                        for key in ['prompt', 'response']:
                            value = formatted_sample[key]
                            assert isinstance(value, str)
                            encoded_sample[key] = value.encode('utf-8')
                        out.write(encoded_sample)
                    else:
                        out.write(formatted_sample)

        if tokenizer is not None and examples_removed > 0:
            warnings.warn(
                f'Dropped {examples_removed} examples where the prompt was longer than {max_seq_len}, '
                +
                'the prompt or response was empty, or the response was all padding tokens.',
            )


def convert_finetuning_dataset_from_args(
    dataset: str,
    data_subset: Optional[str],
    splits: list[str],
    preprocessor: Optional[str],
    data_files: list[str],
    skip_preprocessing: bool,
    out_root: str,
    local: Optional[str],
    compression: Optional[str],
    num_workers: Optional[int],
    tokenizer: Optional[str],
    tokenizer_kwargs: Optional[str],
    max_seq_len: int,
    target_prompts: str,
    target_responses: str,
    encoder_decoder: bool,
):
    """A wrapper for `convert_dataset_hf`
    Args:
        dataset (str): Name of the dataset (e.g., first argument to `datasets.load_dataset`, for jsonl data format, it is `json`).
        data_subset (Optional[str]): Subset of data to use.
        splits (list[str]): Comma-separated list of dataset splits
        preprocessor (Optional[str]): Name or import path of function used to preprocess (reformat) the dataset.
        data_files (list[str]): Data file for each split. Comma-separated.
        skip_preprocessing (bool): Whether to skip preprocessing.
        out_root (str): Root path of output directory where MDS shards will be stored. Can be a remote URI.
        local (Optional[str]): Root path of local directory if you want to keep a local copy when out_root is remote.
        compression (Optional[str]): Name of compression algorithm to use.
        num_workers (Optional[int]): Number of workers.
        tokenizer (Optional[str]): Tokenizer used for processing.
        tokenizer_kwargs (Optional[str]): Keyword arguments for tokenizer initialization in JSON format.
        max_seq_len (int): Maximum sequence length.
        target_prompts (str): Policy for when to use prompts as training targets.
        target_responses (str): Policy for which responses to treat as training targets.
        encoder_decoder (bool): Set if the data are intended to be used to train an encoder-decoder model

    Raises:
        ValueError: If the target settings are invalid.
        ValueError: If the output directory already contains the requested splits.
        """
    if os.path.isdir(out_root) and len(
        set(os.listdir(out_root)).intersection(set(splits)),
    ) > 0:
        raise ValueError(
            f'--out_root={out_root} contains {os.listdir(out_root)} which cannot overlap with the requested splits {splits}.',
        )

    if tokenizer_kwargs is not None:
        parsed_tokenizer_kwargs = json.loads(tokenizer_kwargs)
    else:
        parsed_tokenizer_kwargs = {}

    if len(data_files) > 0 and len(data_files,) != len(splits):
        raise ValueError(
            f'If data_files is set, data_files and splits must have the same length. Got {len(data_files)=} while {len(splits)=}',
        )
    convert_finetuning_dataset(
        dataset=dataset,
        data_subset=data_subset,
        splits=splits,
        preprocessor=preprocessor,
        data_files=data_files,
        skip_preprocessing=skip_preprocessing,
        out_root=out_root,
        local=local,
        compression=compression,
        num_workers=num_workers,
        tokenizer=tokenizer,
        tokenizer_kwargs=parsed_tokenizer_kwargs,
        max_seq_len=max_seq_len,
        target_prompts=target_prompts,
        target_responses=target_responses,
        encoder_decoder=encoder_decoder,
    )
