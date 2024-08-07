# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
import pathlib

import numpy as np
import pytest
import torch
from streaming import MDSWriter

from llmfoundry.data import SUPPORTED_MDS_ENCODING_TYPES, StreamingTextDataset
from llmfoundry.data.finetuning.tasks import StreamingFinetuningDataset


@pytest.mark.parametrize(
    'token_encoding_type',
    SUPPORTED_MDS_ENCODING_TYPES + ['default'],
)
@pytest.mark.parametrize('use_bytes', [True, False])
@pytest.mark.parametrize('samples', [10])
@pytest.mark.parametrize('max_seq_len', [2048])
def test_encoding_types_text(
    tmp_path: pathlib.Path,
    token_encoding_type: str,
    use_bytes: bool,
    samples: int,
    max_seq_len: int,
):
    dataset_local_path = str(tmp_path)
    if token_encoding_type != 'default':
        encoding_dtype = getattr(np, token_encoding_type)
    else:
        encoding_dtype = None

    if use_bytes:
        columns = {
            'tokens': 'bytes',
        }
    else:
        columns = {
            'tokens':
                'ndarray:' + token_encoding_type
                if token_encoding_type != 'default' else 'ndarray',
        }

    with MDSWriter(out=dataset_local_path, columns=columns) as writer:
        for _ in range(samples):
            if token_encoding_type != 'default':
                tokens = np.random.randint(
                    0,
                    np.iinfo(encoding_dtype).max,
                    max_seq_len,
                    dtype=encoding_dtype,
                )
            else:
                tokens = np.random.randint(
                    0,
                    200,
                    max_seq_len,
                )
            if use_bytes:
                tokens = tokens.tobytes()
            writer.write({'tokens': tokens})

    if use_bytes and token_encoding_type != 'default':
        dataset = StreamingTextDataset(
            tokenizer=None,
            token_encoding_type=token_encoding_type,
            max_seq_len=max_seq_len,
            local=dataset_local_path,
            batch_size=1,
        )
    else:
        # There should be no need to pass in the token encoding type if writing out ndarrays,
        # or if using the default token encoding type.
        dataset = StreamingTextDataset(
            tokenizer=None,
            max_seq_len=max_seq_len,
            local=dataset_local_path,
            batch_size=1,
        )

    for _, sample in enumerate(dataset):
        # StreamingTextDataset should return an int64 torch Tensor
        assert sample.dtype == torch.int64
        assert sample.shape == (max_seq_len,)


@pytest.mark.parametrize(
    'token_encoding_type',
    SUPPORTED_MDS_ENCODING_TYPES + ['default'],
)
@pytest.mark.parametrize('use_bytes', [True, False])
@pytest.mark.parametrize('samples', [10])
@pytest.mark.parametrize('max_seq_len', [2048])
def test_encoding_types_finetuning(
    tmp_path: pathlib.Path,
    token_encoding_type: str,
    use_bytes: bool,
    samples: int,
    max_seq_len: int,
):
    dataset_local_path = str(tmp_path)
    if token_encoding_type != 'default':
        encoding_dtype = getattr(np, token_encoding_type)
    else:
        encoding_dtype = None

    if use_bytes:
        columns = {
            'input_ids': 'bytes',
            'labels': 'bytes',
        }
    else:
        columns = {
            'input_ids':
                'ndarray:' + token_encoding_type
                if token_encoding_type != 'default' else 'ndarray',
            'labels':
                'ndarray:' + token_encoding_type
                if token_encoding_type != 'default' else 'ndarray',
        }

    with MDSWriter(out=dataset_local_path, columns=columns) as writer:
        for _ in range(samples):
            if token_encoding_type != 'default':
                input_ids = np.random.randint(
                    0,
                    np.iinfo(encoding_dtype).max,
                    max_seq_len,
                    dtype=encoding_dtype,
                )
                labels = np.random.randint(
                    0,
                    np.iinfo(encoding_dtype).max,
                    max_seq_len,
                    dtype=encoding_dtype,
                )
            else:
                input_ids = np.random.randint(
                    0,
                    200,
                    max_seq_len,
                )
                labels = np.random.randint(
                    0,
                    200,
                    max_seq_len,
                )
            if use_bytes:
                input_ids = input_ids.tobytes()
                labels = labels.tobytes()
            writer.write({'input_ids': input_ids, 'labels': labels})

    if use_bytes and token_encoding_type != 'default':
        dataset = StreamingFinetuningDataset(
            tokenizer=None,
            token_encoding_type=token_encoding_type,
            local=dataset_local_path,
            max_seq_len=max_seq_len,
            batch_size=1,
        )
    else:
        # There should be no need to pass in the token encoding type if writing out ndarrays,
        # or if using the default token encoding type.
        dataset = StreamingFinetuningDataset(
            tokenizer=None,
            local=dataset_local_path,
            max_seq_len=max_seq_len,
            batch_size=1,
        )

    for _, sample in enumerate(dataset):
        # StreamingFinetuningDataset puts samples in a list, and converts arrays to lists too.
        assert isinstance(sample['turns'][0]['input_ids'][0], int)
        assert len(sample['turns'][0]['input_ids']) == max_seq_len
        assert isinstance(sample['turns'][0]['labels'][0], int)
        assert len(sample['turns'][0]['labels']) == max_seq_len


@pytest.mark.parametrize(
    'token_encoding_type',
    ['int17', 'float32', 'complex', 'int4'],
)
@pytest.mark.parametrize('use_finetuning', [True, False])
def test_unsupported_encoding_type(
    token_encoding_type: str,
    use_finetuning: bool,
):
    with pytest.raises(ValueError, match='The token_encoding_type*'):
        if use_finetuning:
            StreamingFinetuningDataset(
                tokenizer=None,
                token_encoding_type=token_encoding_type,
                local='dataset/path',
                max_seq_len=2048,
                batch_size=1,
            )
        else:
            StreamingTextDataset(
                tokenizer=None,
                token_encoding_type=token_encoding_type,
                max_seq_len=2048,
                local='dataset/path',
                batch_size=1,
            )
