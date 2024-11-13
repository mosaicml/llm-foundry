# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming dataset conversion scripts for C4 and The Pile."""
import json
import os
import platform
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Optional, Union

import datasets as hf_datasets
import psutil
import torch
from numpy.typing import NDArray
from streaming import MDSWriter
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from llmfoundry.data import ConcatTokensDataset, NoConcatDataset
from llmfoundry.utils.builders import build_tokenizer


class ConcatMode(Enum):
    NO_CONCAT = 'NO_CONCAT'
    CONCAT_TOKENS = 'CONCAT_TOKENS'


@dataclass
class DataSplitConstants:
    hf_split: str
    folder_split: str
    raw_samples: Optional[int]
    truncated_samples: Union[int, None]


@dataclass
class DatasetConstants:
    chars_per_sample: int
    chars_per_token: int
    splits: dict[str, DataSplitConstants] = field(default_factory=dict)

    def __iter__(self):
        for v in self.splits.values():
            yield v


class TrainSmallConstants(DataSplitConstants):

    def __init__(
        self,
        hf_split: str = 'train',
        folder_split: str = 'train_small',
        raw_samples: int = 100000,
        truncated_samples: int = 100000,
    ):
        super().__init__(hf_split, folder_split, raw_samples, truncated_samples)


class ValSmallConstants(DataSplitConstants):

    def __init__(
        self,
        hf_split: str = 'validation',
        folder_split: str = 'val_small',
        raw_samples: int = 10000,
        truncated_samples: int = 10000,
    ):
        super().__init__(hf_split, folder_split, raw_samples, truncated_samples)


class ValXSmallConstants(DataSplitConstants):

    def __init__(
        self,
        hf_split: str = 'validation',
        folder_split: str = 'val_xsmall',
        raw_samples: int = 3000,
        truncated_samples: int = 3000,
    ):
        super().__init__(hf_split, folder_split, raw_samples, truncated_samples)


pileconstants = DatasetConstants(
    chars_per_sample=6212,  # Computed over validation set
    chars_per_token=4,  # OpenAI estimate
)
pileconstants.splits['train'] = DataSplitConstants(
    hf_split='train',
    folder_split='train',
    raw_samples=210607728,
    truncated_samples=None,
)
pileconstants.splits['train_small'] = DataSplitConstants(
    hf_split='train',
    folder_split='train_small',
    raw_samples=100000,
    truncated_samples=100000,
)
pileconstants.splits['val'] = DataSplitConstants(
    hf_split='validation',
    folder_split='val',
    raw_samples=214670,
    truncated_samples=None,
)
pileconstants.splits['val_small'] = DataSplitConstants(
    hf_split='validation',
    folder_split='val_small',
    raw_samples=10000,
    truncated_samples=10000,
)
pileconstants.splits['val_xsmall'] = DataSplitConstants(
    hf_split='validation',
    folder_split='val_xsmall',
    raw_samples=3000,
    truncated_samples=3000,
)

c4constants = DatasetConstants(
    chars_per_sample=2163,  # Computed over validation set
    chars_per_token=4,  # OpenAI estimate
)
c4constants.splits['train'] = DataSplitConstants(
    hf_split='train',
    folder_split='train',
    raw_samples=364868892,
    truncated_samples=None,
)
c4constants.splits['train_small'] = DataSplitConstants(
    hf_split='train',
    folder_split='train_small',
    raw_samples=100000,
    truncated_samples=100000,
)
c4constants.splits['val'] = DataSplitConstants(
    hf_split='validation',
    folder_split='val',
    raw_samples=364608,
    truncated_samples=None,
)
c4constants.splits['val_small'] = DataSplitConstants(
    hf_split='validation',
    folder_split='val_small',
    raw_samples=10000,
    truncated_samples=10000,
)
c4constants.splits['val_xsmall'] = DataSplitConstants(
    hf_split='validation',
    folder_split='val_xsmall',
    raw_samples=3000,
    truncated_samples=3000,
)
c4constants.splits['val_xxsmall'] = DataSplitConstants(
    hf_split='validation',
    folder_split='val_xxsmall',
    raw_samples=100,
    truncated_samples=100,
)

CONSTS = {'allenai/c4': c4constants, 'the_pile': pileconstants}


def build_hf_dataset(
    dataset_name: str,
    split: str,
    mode: ConcatMode,
    max_length: Optional[int] = None,
    bos_text: str = '',
    eos_text: str = '',
    no_wrap: bool = False,
    tokenizer: PreTrainedTokenizerBase = None,
    data_subset: Union[str, None] = None,
) -> IterableDataset:
    """Build an IterableDataset over the HF C4 or pile source data.

    Args:
        dataset_name (str): Dataset name
        split (str): Split name.
        mode (ConcatMode): NO_CONCAT, or CONCAT_TOKENS
        max_length (int): The length of concatenated tokens
        bos_text (str): text to insert at the beginning of each sequence
        eos_text (str): text to insert at the end of each sequence
        no_wrap (bool): if concatenating, whether to wrap text across `max_length` boundaries
        tokenizer (PreTrainedTokenizerBase): if mode is CONCAT_TOKENS, the tokenizer to use
        data_subset (str): Referred to as "name" in HuggingFace datasets.load_dataset.
            Typically "all" (The Pile) or "en" (c4).

    Returns:
        An IterableDataset.
    """
    hf_dataset = hf_datasets.load_dataset(
        path=dataset_name,
        name=data_subset,
        split=split,
        streaming=True,
    )
    if mode == ConcatMode.NO_CONCAT:
        dataset = NoConcatDataset(hf_dataset)
    else:
        if not isinstance(tokenizer, PreTrainedTokenizerBase):
            raise ValueError(
                f'{tokenizer=} must be of type PreTrainedTokenizerBase',
            )
        if max_length is None:
            raise ValueError(f'max_length must be set.')
        if bos_text + eos_text == '':
            test_tokens = tokenizer('test')
            if test_tokens['input_ids'][
                0] != tokenizer.bos_token_id and test_tokens['input_ids'][
                    -1] != tokenizer.eos_token_id:
                tok_error_msg = 'This tokenizer does not insert an EOS nor BOS token. '
                tok_error_msg += 'Concatenating with this tokenizer will result in sequences being '
                tok_error_msg += 'attached without a separating token. Please use another tokenizer, '
                tok_error_msg += 'such as facebook/opt-125m, or specify EOS/BOS text with e.g. '
                tok_error_msg += '--bos_text=<|endoftext|>.'
                raise ValueError(tok_error_msg)
        dataset = ConcatTokensDataset(
            hf_dataset=hf_dataset,
            tokenizer=tokenizer,
            max_length=max_length,
            bos_text=bos_text,
            eos_text=eos_text,
            no_wrap=no_wrap,
        )
    return dataset


def _est_progress_denominator(
    total_samples: int,
    chars_per_sample: int,
    chars_per_token: int,
    mode: ConcatMode,
    max_length: int,
):
    est_tokens_per_sample = chars_per_sample // chars_per_token
    if mode == ConcatMode.NO_CONCAT:
        return total_samples
    elif mode == ConcatMode.CONCAT_TOKENS:
        return total_samples * est_tokens_per_sample // max_length


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: Optional[int],
) -> DataLoader:
    if num_workers is None:
        # Multiple workers is only supported on linux machines
        if 'linux' or 'macos' in platform.platform().lower():
            num_workers = max(1, psutil.cpu_count())
        else:
            num_workers = 0

    # If using multiple workers, configure each worker to prefetch as many samples as it can, up to
    # the aggregate device batch size
    # If not using workers, the torch DataLoader expects the default value for prefetch_factor,
    # which non-intuitively must be 2.
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
) -> Iterable[Union[dict[str, bytes], dict[str, NDArray]]]:
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
            yield {
                k:
                v[idx].numpy() if isinstance(v[idx], torch.Tensor) else v[idx]
                for k, v in batch.items()
            }


def convert_dataset_hf(
    dataset: str,
    data_subset: Optional[str],
    splits: list[str],
    out_root: str,
    compression: Optional[str],
    concat_tokens: Optional[int],
    tokenizer: Optional[str],
    tokenizer_kwargs: dict[str, Any],
    bos_text: str,
    eos_text: str,
    no_wrap: bool,
    num_workers: Optional[int],
) -> None:
    """Converts HuggingFace datasets to MDS format.

    Args:
        dataset (str): Name of the dataset
        data_subset (Optional[str]): Subset of the dataset (e.g., "all" or "en")
        splits (list[str]): Comma-separated list of dataset splits
        out_root (str): Output root directory
        compression (Optional[str]): Compression type
        concat_tokens (Optional[int]): Concatenate tokens up to this many tokens
        tokenizer (Optional[str]): Tokenizer name
        tokenizer_kwargs (dict[str, Any]): Tokenizer keyword arguments
        bos_text (str): BOS text
        eos_text (str): EOS text
        no_wrap (bool): Do not wrap text across max_length boundaries
        num_workers (Optional[int]): Number of workers

    Raises:
        KeyError: If constants are not defined for the split
    """
    try:
        dataset_constants = CONSTS[dataset]
    except KeyError:
        raise ValueError(
            f'Constants for dataset "{dataset}" not found. Currently only "the_pile" and "allenai/c4" are supported.',
        )

    if concat_tokens is not None and tokenizer is not None:
        mode = ConcatMode.CONCAT_TOKENS
        built_tokenizer = build_tokenizer(tokenizer, tokenizer_kwargs)
        # we will enforce length, so suppress warnings about sequences too long for the model
        built_tokenizer.model_max_length = int(1e30)
        columns = {'tokens': 'ndarray:int32'}
    else:
        mode = ConcatMode.NO_CONCAT
        built_tokenizer = None
        columns = {'text': 'str'}

    for split_name in splits:
        try:
            split = dataset_constants.splits[split_name]
        except KeyError:
            raise KeyError(f'Constants not defined for split {split_name}.')
        hf_split = split.hf_split
        folder_split = split.folder_split
        expected_num_samples = split.raw_samples
        truncate_num_samples = split.truncated_samples
        # Only generate the splits requested
        if folder_split not in splits:
            continue

        # Get samples
        hf_dataset = build_hf_dataset(
            dataset_name=dataset,
            data_subset=data_subset,
            split=hf_split,
            mode=mode,
            max_length=concat_tokens,
            bos_text=bos_text,
            eos_text=eos_text,
            no_wrap=no_wrap,
            tokenizer=built_tokenizer,
        )
        loader = build_dataloader(
            dataset=hf_dataset,
            batch_size=512,
            num_workers=num_workers,
        )
        samples = generate_samples(
            loader,
            truncate_num_samples=truncate_num_samples,
        )

        if expected_num_samples is not None and concat_tokens is not None:
            denominator = truncate_num_samples if truncate_num_samples is not None else _est_progress_denominator(
                total_samples=expected_num_samples,
                chars_per_sample=dataset_constants.chars_per_sample,
                chars_per_token=dataset_constants.chars_per_token,
                mode=mode,
                max_length=concat_tokens,
            )
        else:
            denominator = None

        # Write samples
        print(f'Converting {folder_split} to MDS format...')
        print(
            f'Note: the progress bar is based on the dataset length before tokenization, and may finish at a value before 100%.',
        )
        with MDSWriter(
            columns=columns,
            out=os.path.join(out_root, folder_split),
            compression=compression,
        ) as out:
            if denominator is not None:
                for sample in tqdm(
                    samples,
                    desc=folder_split,
                    total=denominator,
                ):
                    out.write(sample)
            else:
                for sample in tqdm(samples, desc=folder_split):
                    out.write(sample)


def convert_dataset_hf_from_args(
    dataset: str,
    data_subset: Optional[str],
    splits: list[str],
    out_root: str,
    compression: Optional[str],
    concat_tokens: Optional[int],
    tokenizer: Optional[str],
    tokenizer_kwargs: Optional[str],
    bos_text: Optional[str],
    eos_text: Optional[str],
    no_wrap: bool,
    num_workers: Optional[int],
) -> None:
    """A wrapper for `convert_dataset_hf` that parses arguments.

    Args:
        dataset (str): Name of the dataset
        data_subset (Optional[str]): Subset of the dataset (e.g., "all" or "en")
        splits (list[str]): Comma-separated list of dataset splits
        out_root (str): Output root directory
        compression (Optional[str]): Compression type
        concat_tokens (Optional[int]): Concatenate tokens up to this many tokens
        tokenizer (Optional[str]): Tokenizer name
        tokenizer_kwargs (Optional[str]): Tokenizer keyword arguments in JSON format
        bos_text (Optional[str]): BOS text
        eos_text (Optional[str]): EOS text
        no_wrap (bool): Do not wrap text across max_length boundaries
        num_workers (Optional[int]): Number of workers

    Raises:
        ValueError: If the output directory already contains the requested splits
        ValueError: If `concat_tokens` is set but `tokenizer` is not
    """
    os.environ['WORLD_SIZE'] = '1'
    if tokenizer_kwargs:
        parsed_tokenizer_kwargs = json.loads(tokenizer_kwargs)
    else:
        parsed_tokenizer_kwargs = {}

    if os.path.isdir(out_root) and len(
        set(os.listdir(out_root)).intersection(set(splits)),
    ) > 0:
        raise ValueError(
            f'--out_root={out_root} contains {os.listdir(out_root)} which cannot overlap with the requested splits {splits}.',
        )

    # Make sure we have needed concat options
    if (
        concat_tokens is not None and isinstance(concat_tokens, int) and
        tokenizer is None
    ):
        raise ValueError(
            'When setting --concat_tokens, you must specify a --tokenizer',
        )

    # now that we have validated them, change BOS/EOS to strings and convert
    convert_dataset_hf(
        dataset=dataset,
        data_subset=data_subset,
        splits=splits,
        out_root=out_root,
        compression=compression,
        concat_tokens=concat_tokens,
        tokenizer=tokenizer,
        tokenizer_kwargs=parsed_tokenizer_kwargs,
        bos_text=bos_text if bos_text else '',
        eos_text=eos_text if eos_text else '',
        no_wrap=no_wrap,
        num_workers=num_workers,
    )
