# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Datasets for converting to MDS Shards."""
import os
import warnings
from typing import Dict, Iterable, Union

import datasets as hf_datasets
import numpy as np
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizerBase


class NoConcatDataset(IterableDataset):
    """An IterableDataset that returns text samples for MDSWriter.

    Returns dicts of {'text': bytes}
    """

    def __init__(self, hf_dataset: Union[hf_datasets.IterableDataset,
                                         hf_datasets.Dataset]):
        self.hf_dataset = hf_dataset

    def __iter__(self) -> Iterable[Dict[str, bytes]]:
        for sample in self.hf_dataset:
            # print(sample)
            # convert to bytes to store in MDS binary format
            yield {'text': sample['text'].encode('utf-8')}


class ConcatTokensDataset(IterableDataset):
    """An IterableDataset that returns token samples for MDSWriter.

    Returns dicts of {'tokens': bytes}

    To use data created by this class and written to MDS format:

    ```python
        import torch
        from streaming.base import StreamingDataset
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained('your/tokenizer')
        ds = StreamingDataset(local='mds-data-folder', split='val')

        # note, you need to copy the numpy array because the original is non-writeable
        # and torch does not support non-writeable tensors, so you get a scary warning and
        # if you do try to write to the tensor you get undefined behavior
        tokens = torch.from_numpy(np.frombuffer(ds[0]['tokens'], dtype=np.int64).copy())
        print(tokenizer.decode(tokens))
    ```
    """

    def __init__(
        self,
        hf_dataset: Union[hf_datasets.IterableDataset, hf_datasets.Dataset],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        bos_text: str,
        eos_text: str,
        no_wrap: bool,
    ):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        self.max_length = max_length
        self.bos_text = bos_text
        self.eos_text = eos_text
        self.should_wrap = not no_wrap

        self.bos_tokens = self.tokenizer(self.bos_text,
                                         truncation=False,
                                         padding=False,
                                         add_special_tokens=False)['input_ids']
        if len(self.bos_tokens) > 1:
            warnings.warn(
                f'You specified --concat_tokens with --bos_text, but your BOS text is not tokenizing to one token\
                , instead we got {self.bos_tokens}. Quit if this was in error.')

        self.eos_tokens = self.tokenizer(self.eos_text,
                                         truncation=False,
                                         padding=False,
                                         add_special_tokens=False)['input_ids']
        if len(self.eos_tokens) > 1:
            warnings.warn(
                f'You specified --concat_tokens with --eos_text, but your EOS text is not tokenizing to one token\
                , instead we got {self.eos_tokens}. Quit if this was in error.')

        eos_text_provided = self.eos_text != ''
        bos_text_provided = self.bos_text != ''
        test_text = self.tokenizer('')
        if len(test_text['input_ids']) > 0 and (eos_text_provided or
                                                bos_text_provided):
            message = 'both eos and bos' if eos_text_provided and bos_text_provided else (
                'eos_text' if eos_text_provided else 'bos_text')
            warnings.warn(
                f'The provided tokenizer adds special tokens, but you also specified {message}. This may result '
                'in duplicated special tokens. Please be sure this is what you intend.'
            )

    def __iter__(self) -> Iterable[Dict[str, bytes]]:

        buffer = []
        for sample in self.hf_dataset:
            encoded = self.tokenizer(sample['text'],
                                     truncation=False,
                                     padding=False)
            iids = encoded['input_ids']
            buffer = buffer + self.bos_tokens + iids + self.eos_tokens
            while len(buffer) >= self.max_length:
                concat_sample = buffer[:self.max_length]
                buffer = buffer[self.max_length:] if self.should_wrap else []
                yield {
                    # convert to bytes to store in MDS binary format
                    'tokens': np.asarray(concat_sample).tobytes()
                }
