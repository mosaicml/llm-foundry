# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from llmfoundry.utils.builders import build_dataloader
from tests.data_utils import (
    temporary_contrastive_streaming_dataset,
    temporary_tokenizer,
)


@pytest.mark.parametrize(
    'ds_format',
    ['one_query_one_response', 'one_query_multiple_responses'],
)
def test_pairs_dataloader(ds_format: str):
    with temporary_tokenizer('mosaicml/mpt-7b') as tokenizer, \
        temporary_contrastive_streaming_dataset(ds_format) as data_dir:
        cfg = {
            'name': 'contrastive_pairs',
            'dataset': {
                'remote': data_dir,
                'split': 'train',
                'max_seq_len': 1024,
                'shuffle_hard_negatives': False,
            },
            'drop_last': False,
            'num_workers': 1,
            'max_hard_negatives': 2,
        }

        dl = build_dataloader(cfg, tokenizer, 1)

        for i, batch in enumerate(dl.dataloader):
            # query + positive + max 2 hard negatives
            assert batch['input_ids'].shape[1] <= 4
            if ds_format == 'one_query_one_response':
                # 0th item is the query, 1st item is the positive, 2nd item is (optionally) the negative
                expected = tokenizer([
                    f'hello {i}',
                    f'world {i}',
                ],
                                     padding='max_length',
                                     max_length=1024,
                                     return_tensors='pt')['input_ids']
            else:
                # 0th item is the query, 1st item is the positive, 2nd and 3rd items are the negatives
                expected = tokenizer([
                    f'query {i}',
                    f'positive passage {i}',
                    f'negative passage {i}',
                    f'negative passage {i + 1}',
                ],
                                     padding='max_length',
                                     max_length=1024,
                                     return_tensors='pt')['input_ids']

            assert torch.allclose(batch['input_ids'][0], expected)
