# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming dataset conversion scripts for json files."""
import os
from enum import Enum
from glob import glob
from typing import Optional

import datasets as hf_datasets
from streaming import MDSWriter
from torch.utils.data import IterableDataset
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from llmfoundry.data import ConcatTokensDataset, NoConcatDataset


class ConcatMode(Enum):
    NO_CONCAT = 'NO_CONCAT'
    CONCAT_TOKENS = 'CONCAT_TOKENS'


def build_hf_dataset(
    path: str,
    split: str,
    mode: ConcatMode,
    max_length: Optional[int] = None,
    bos_text: str = '',
    eos_text: str = '',
    no_wrap: bool = False,
    tokenizer: PreTrainedTokenizerBase = None,
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
    if os.path.isdir(path):
        data_files = glob(f'{path}/*')
    else:
        data_files = path

    hf_dataset = hf_datasets.load_dataset(
        'json',
        data_files=data_files,
        split=split,
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


def convert_dataset_json(
    path: str,
    out_root: str,
    compression: Optional[str],
    concat_tokens: Optional[int],
    split: str,
    tokenizer: Optional[str] = None,
    bos_text: str = '',
    eos_text: str = '',
    no_wrap: bool = False,
    num_workers: Optional[int] = None,
) -> None:
    """Create C4/pile streaming dataset.

    Args:
        path (str): Path to the input data file
        out_root (str): Output root directory
        compression (Optional[str]): Compression type, if any
        concat_tokens (Optional[int]): Convert text to tokens and concatenate up to this many tokens
        split (str): Dataset split to process
        tokenizer (Optional[str]): Tokenizer name
        bos_text (str): Text to insert at the beginning of each sequence
        eos_text (str): Text to insert at the end of each sequence
        no_wrap (bool): Do not wrap text across max_length boundaries
        num_workers (Optional[int]): Number of workers for data loading
    """
    if concat_tokens is not None:
        mode = ConcatMode.CONCAT_TOKENS
        built_tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        # we will enforce length, so suppress warnings about sequences too long for the model
        built_tokenizer.model_max_length = int(1e30)
        columns = {'tokens': 'ndarray:int32'}
    else:
        mode = ConcatMode.NO_CONCAT
        built_tokenizer = None
        columns = {'text': 'str'}

    # Get samples
    dataset = build_hf_dataset(
        path=path,
        split=split,
        mode=mode,
        max_length=concat_tokens,
        bos_text=bos_text,
        eos_text=eos_text,
        no_wrap=no_wrap,
        tokenizer=built_tokenizer,
    )

    print('here')

    # Write samples
    print(f'Converting to MDS format...')
    print(
        f'Note that the progress bar is based on the dataset length before tokenization.',
    )
    print(f'It will finish at a value below 100% if tokenizing')
    with MDSWriter(
        columns=columns,
        out=os.path.join(out_root),
        compression=compression,
    ) as out:
        for sample in tqdm(dataset):
            out.write(sample)


def convert_dataset_json_from_args(
    path: str,
    out_root: str,
    compression: Optional[str],
    concat_tokens: Optional[int],
    split: str,
    tokenizer: Optional[str] = None,
    bos_text: Optional[str] = None,
    eos_text: Optional[str] = None,
    no_wrap: bool = False,
    num_workers: Optional[int] = None,
) -> None:
    """A wrapper for `convert_dataset_json` that parses arguments.

    Args:
        path (str): Path to the input data file
        out_root (str): Output root directory
        compression (Optional[str]): Compression type, if any
        concat_tokens (Optional[int]): Convert text to tokens and concatenate up to this many tokens
        split (str): Dataset split to process
        tokenizer (Optional[str]): Tokenizer name
        bos_text (Optional[str]): Text to insert at the beginning of each sequence
        eos_text (Optional[str]): Text to insert at the end of each sequence
        no_wrap (bool): Do not wrap text across max_length boundaries
        num_workers (Optional[int]): Number of workers for data loading

    Raises:
        ValueError: If the out_root directory exists and contains files that overlap with the requested splits
        ValueError: If concat_tokens is set and a tokenizer is not provided
    """
    if os.path.isdir(out_root) and len(
        set(os.listdir(out_root)).intersection(set(split)),
    ) > 0:
        raise ValueError(
            f'--out_root={out_root} contains {os.listdir(out_root)} which cannot overlap with the requested splits {split}.',
        )

    # Make sure we have needed concat options
    if (
        concat_tokens is not None and isinstance(concat_tokens, int) and
        tokenizer is None
    ):
        ValueError(
            'When setting --concat_tokens, you must specify a --tokenizer',
        )

    # now that we have validated them, change BOS/EOS to strings
    if bos_text is None:
        bos_text = ''
    if eos_text is None:
        eos_text = ''

    convert_dataset_json(
        path=path,
        out_root=out_root,
        compression=compression,
        concat_tokens=concat_tokens,
        split=split,
        tokenizer=tokenizer,
        bos_text=bos_text,
        eos_text=eos_text,
        no_wrap=no_wrap,
        num_workers=num_workers,
    )
