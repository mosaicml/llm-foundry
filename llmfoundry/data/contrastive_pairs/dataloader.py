# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Dataset and dataloader for contrastive training.

Build a StreamingPairsDataset dataset and dataloader for contrastive training.
"""

import json
from itertools import islice
from typing import Any, Literal, Mapping, Optional, Union

import numpy as np
import torch
from composer.core import DataSpec
from composer.utils import retry
from streaming import Stream, StreamingDataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from llmfoundry import registry
from llmfoundry.data.text_data import (
    ConcatenatedSequenceCollatorWrapper,
    StreamingTextDataset,
)
from llmfoundry.utils.registry_utils import construct_from_registry

ContrastiveSampleType = Literal['one_query_multiple_responses',
                                'one_query_one_response']


def _get_contrastive_sample_type(
    sample: Mapping[str, Any],
) -> ContrastiveSampleType:
    """Get the type of contrastive sample from the sample.

    Args:
        sample (Mapping): A sample from the dataset.

    Returns:
        ContrastiveSampleType: The type of contrastive sample.
    """
    sample_contains_text_a = any(
        key.startswith('text_a') for key in sample.keys()
    )
    sample_contains_text_b = any(
        key.startswith('text_b') for key in sample.keys()
    )

    if sample_contains_text_a and sample_contains_text_b:
        return 'one_query_one_response'
    elif 'query_text' in sample and 'positive_passage' in sample and 'negative_passages' in sample:
        return 'one_query_multiple_responses'
    else:
        raise ValueError(
            'Sample does not contain the required keys for contrastive training. \
                For datasets with one query and one response, the keys must contain \
                "text_a" and "text_b". For datasets with one query and multiple responses, \
                the keys must contain "query_text", "positive_passage", and "negative_passages".',
        )


class StreamingPairsDataset(StreamingTextDataset):
    """Contrastive pairs dataset using MosaicML's StreamingTextDataset.

    Args:
        max_hard_negatives (int, optional): The maximum number of hard negatives to include in the
            contrastive training samples. Defaults to ``None``. If ``None``, all hard negatives are
            included.
        prepend_query (str, optional): Text to prepend to the query text. Defaults to ``''``.
        prepend_passage (str, optional): Text to prepend to the passage text. Defaults to ``''``.
        append_eos_token (bool, optional): Whether to append the EOS token to the query and passage
            text. Defaults to ``False``. Mutually exclusive with ``append_token``.
        append_token (str, optional): Token to append to the query and passage text. Defaults to
            ``''``. Mutually exclusive with ``append_eos_token``.
        shuffle_hard_negatives (bool, optional): Whether to shuffle the hard negatives. Defaults to
            ``False``.
        **kwargs: Additional keyword arguments to pass to the superclass. See ``StreamingTextDataset``
            for more information.
    """

    def __init__(
        self,
        max_hard_negatives: Optional[int] = None,
        prepend_query: str = '',
        prepend_passage: str = '',
        append_eos_token: bool = False,
        append_token: str = '',
        shuffle_hard_negatives: bool = False,
        **kwargs: Any,
    ):

        super().__init__(**kwargs)

        self.max_hard_negatives = max_hard_negatives
        self.prepend_query = prepend_query
        self.prepend_passage = prepend_passage
        self.shuffle_hard_negatives = shuffle_hard_negatives
        self._generator = np.random.default_rng(seed=self.shuffle_seed)
        if append_eos_token:
            if append_token != '':
                raise ValueError(
                    'The arguments append_eos_token and append_token are mutually exclusive.',
                )
            self.append_token = self.tokenizer.eos_token
        else:
            self.append_token = append_token

    def _get_contrastive_samples(
        self,
        query_text: str,
        positive_response: str,
        negative_responses: list[str],
    ) -> dict[str, Union[str, list[str]]]:
        """Flatten contrastive samples into a list of strings.

        Args:
            query_text (str): The query text.
            positive_response (str): The positive response.
            negative_responses (List[str]): The negative responses.

        Returns:
            Dict[str, Union[str, List[str]]]: The contrastive samples, with keys 'query', 'positive', and 'negative'.
        """
        query_text = f'{self.prepend_query}{query_text}{self.append_token}'
        positive_response = f'{self.prepend_passage}{positive_response}{self.append_token}'
        if self.shuffle_hard_negatives:
            self._generator.shuffle(negative_responses)
        negative_responses = negative_responses[:self.max_hard_negatives]
        negative_responses = [
            f'{self.prepend_passage}{response}{self.append_token}'
            for response in negative_responses
        ]
        return {
            'query': query_text,
            'positive': positive_response,
            'negative': negative_responses,
        }

    @retry(BlockingIOError, num_attempts=5, initial_backoff=1.0, max_jitter=0.5)
    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        sample = StreamingDataset.__getitem__(self, idx)
        text_samples = []

        sample_type = _get_contrastive_sample_type(sample)
        if sample_type == 'one_query_one_response':
            text_samples = self._get_contrastive_samples(
                sample['text_a'],
                sample['text_b'],
                [],
            )
        elif sample_type == 'one_query_multiple_responses':
            negative_passages_str = sample['negative_passages']
            text_samples = self._get_contrastive_samples(
                sample['query_text'],
                sample['positive_passage'],
                json.loads(negative_passages_str),
            )
        else:
            raise ValueError(f'Unknown sample type: {sample_type}')

        token_samples = self._tokenize(text_samples)
        return token_samples

    def _tokenize(
        self,
        text_samples: dict[str, Union[str, list[str]]],
    ) -> dict[str, list[int]]:
        if self.tokenizer.pad_token is None:
            raise RuntimeError(
                'If tokenizing on-the-fly, tokenizer must have a pad_token_id',
            )

        text_samples_list = [text_samples['query'], text_samples['positive']]
        text_samples_negatives = text_samples['negative']
        assert isinstance(text_samples_negatives, list)  # pyright type check
        text_samples_list.extend(text_samples_negatives)
        return self.tokenizer(
            text_samples_list,
            truncation=True,
            padding='max_length',
            max_length=self.max_seq_len,
        )


def build_pairs_dataloader(
    dataset: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    device_batch_size: int,
    drop_last: bool,
    num_workers: int,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    timeout: int = 0,
    max_hard_negatives: Optional[int] = None,
) -> DataSpec:
    dataset_cfg = dataset
    streams_dict = dataset.pop('streams', None)
    eos_token_id = dataset.pop('eos_token_id', None)
    bos_token_id = dataset.pop('bos_token_id', None)

    streams = None
    if streams_dict is not None:
        streams = []
        for stream in streams_dict.values():
            # stream is the streams kwargs
            # fwd all kwargs with **stream allows streaming to check args
            streams.append(Stream(**stream))

    pairs_dataset = StreamingPairsDataset(
        tokenizer=tokenizer,
        streams=streams,
        batch_size=device_batch_size,
        max_hard_negatives=max_hard_negatives,
        **dataset,
    )

    dataloader_cfg = {
        'name': 'contrastive_pairs',
        'dataset': dataset_cfg,
        'drop_last': drop_last,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'prefetch_factor': prefetch_factor,
        'persistent_workers': persistent_workers,
        'timeout': timeout,
    }

    collate_fn, _ = construct_from_registry(
        name='text_collator',
        registry=registry.collators,
        partial_function=False,
        kwargs={
            'dataloader_cfg': dataloader_cfg,
            'tokenizer': tokenizer,
            'dataset_batch_size': device_batch_size,
        },
    )

    if (eos_token_id is not None) or (bos_token_id is not None):
        # Note: Will raise an error if both are non-None
        collate_fn = ConcatenatedSequenceCollatorWrapper(
            base_collator=collate_fn,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
        )

    def collate_fn_without_labels(batch: list[Any]) -> dict[str, torch.Tensor]:
        # Contrastive learning does not require labels, with some embedding models even erroring out if they are present
        processed_batch: dict[str, torch.Tensor] = collate_fn(batch)
        if 'labels' in processed_batch:
            del processed_batch['labels']

        if 'total_tokens' in processed_batch:
            del processed_batch['total_tokens']
        if 'loss_generating_tokens' in processed_batch:
            del processed_batch['loss_generating_tokens']
        return processed_batch

    dl = DataLoader(
        pairs_dataset,
        collate_fn=collate_fn_without_labels,
        batch_size=device_batch_size,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        timeout=timeout,
    )

    return DataSpec(dataloader=dl)


# Helpful to test if your dataloader is working locally
# Run `python dataloader.py  --local_path [local] [--remote_path remote, optional]` and verify that batches are printed out
if __name__ == '__main__':
    import argparse

    from llmfoundry.utils.builders import build_tokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--tokenizer',
        type=str,
        default='EleutherAI/gpt-neox-20b',
        help='the name of the tokenizer to use',
    )
    parser.add_argument(
        '--local_path',
        type=str,
        required=True,
        help='the path to the local copy of the dataset',
    )
    parser.add_argument(
        '--remote_path',
        type=str,
        default=None,
        help='the path to the remote copy to stream from (optional)',
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        help='which split of the dataset to use',
    )
    parser.add_argument(
        '--max_seq_len',
        type=int,
        default=32,
        help='max sequence length to test',
    )

    args = parser.parse_args()

    if args.remote_path is not None:
        print(
            f'Reading {args.split} split from {args.local_path} <- streamed from <- {args.remote_path}',
        )
    else:
        print(f'Reading {args.split} split from {args.local_path}')

    cfg = {
        'name': 'contrastive_pairs',
        'dataset': {
            'local': args.local_path,
            'remote': args.remote_path,
            'split': args.split,
            'shuffle': False,
            'max_seq_len': args.max_seq_len,
            'keep_zip': True,  # in case we need compressed files after testing
        },
        'drop_last': False,
        'num_workers': 4,
    }
    device_batch_size = 2

    tokenizer_name = args.tokenizer
    tokenizer_kwargs = {'model_max_length': args.max_seq_len}
    tokenizer = build_tokenizer(tokenizer_name, tokenizer_kwargs)

    loader = build_pairs_dataloader(
        **cfg,
        tokenizer=tokenizer,
        device_batch_size=device_batch_size,
    ).dataloader
    assert isinstance(loader, DataLoader)
    assert isinstance(loader.dataset, StreamingPairsDataset)
    tokenizer = loader.dataset.tokenizer

    for batch_ix, batch in enumerate(islice(loader, 5)):
        print('\n')
        print('#' * 20, f'Batch {batch_ix}', '#' * 20)
        for k, v in batch.items():
            print(k, v.shape, v.dtype)
        for sample_ix, token_sample in enumerate(batch['input_ids']):
            print('-' * 20, f' Sample {sample_ix} ', '-' * 20)
            print(tokenizer.decode(token_sample))
