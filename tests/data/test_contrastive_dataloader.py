# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from typing import cast

import pytest
import torch

from llmfoundry.utils.builders import build_dataloader
from tests.data_utils import temporary_contrastive_streaming_dataset
from tests.test_utils import MockTokenizer


@pytest.fixture
def mock_tokenizer() -> MockTokenizer:
    return MockTokenizer()


@pytest.mark.parametrize(
    'ds_format',
    ['one_query_one_response', 'one_query_multiple_responses'],
)
def test_pairs_dataloader(
    ds_format: str,
    mock_tokenizer: MockTokenizer,
) -> None:
    with temporary_contrastive_streaming_dataset(ds_format) as data_dir:
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

        dl = build_dataloader(cfg, mock_tokenizer, 1)

        for i, batch in enumerate(dl.dataloader):
            batch_dict = cast(dict[str, torch.Tensor], batch)
            batch_input_ids = batch_dict['input_ids']
            # query + positive + max 2 hard negatives
            assert batch_input_ids.shape[1] <= 4
            if ds_format == 'one_query_one_response':
                # 0th item is the query, 1st item is the positive, 2nd item is (optionally) the negative
                tokenizer_output = mock_tokenizer(
                    [f'hello {i}', f'world {i}'],
                    padding='max_length',
                    max_length=1024,
                    return_tensors='pt',
                )
                tokenizer_dict = cast(dict[str, torch.Tensor], tokenizer_output)
                expected_ids = tokenizer_dict['input_ids']
            else:
                # 0th item is the query, 1st item is the positive, 2nd and 3rd items are the negatives
                tokenizer_output = mock_tokenizer(
                    [
                        f'query {i}',
                        f'positive passage {i}',
                        f'negative passage {i}',
                        f'negative passage {i + 1}',
                    ],
                    padding='max_length',
                    max_length=1024,
                    return_tensors='pt',
                )
                tokenizer_dict = cast(dict[str, torch.Tensor], tokenizer_output)
                expected_ids = tokenizer_dict['input_ids']

            assert torch.allclose(batch_input_ids[0], expected_ids)
