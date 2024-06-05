# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
import os
import pathlib
from streaming import MDSWriter
from llmfoundry.data import StreamingTextDataset
import pytest
import numpy as np
import torch

@pytest.mark.parametrize('token_encoding_type', ['int16', 'int32', 'int64'])
@pytest.mark.parametrize('samples', [10])
@pytest.mark.parametrize('max_seq_len', [2048])
@pytest.mark.parametrize('vocab_size', [10000])
def test_encoding_types(tmp_path: pathlib.Path,
                        token_encoding_type: str,
                        samples: int,
                        max_seq_len: int,
                        vocab_size: int):
    dataset_local_path = str(tmp_path)
    encoding_dtype = getattr(np, token_encoding_type)

    columns = {
        'tokens': 'ndarray:'+token_encoding_type,
    }
    
    with MDSWriter(out=dataset_local_path, columns=columns) as writer:
        for _ in range(samples):
            tokens = np.random.randint(0, vocab_size, max_seq_len, dtype=encoding_dtype)
            writer.write({'tokens': tokens})
    
    print('Dataset local path:', dataset_local_path)
    print(os.listdir(dataset_local_path))

    dataset = StreamingTextDataset(
        tokenizer=None,
        token_encoding_type=token_encoding_type,
        max_seq_len=max_seq_len,
        local=dataset_local_path,
        batch_size=1,
    )

    for _, sample in enumerate(dataset):
        # StreamingTextDataset returns a torch Tensor, not numpy array
        assert sample.dtype == getattr(torch, token_encoding_type)
        assert sample.shape == (max_seq_len,)

@pytest.mark.parametrize('token_encoding_type', ['int17', 'float32', 'complex', 'int8'])
def test_unsupported_encoding_type(token_encoding_type: str):
    with pytest.raises(ValueError, match='The token_encoding_type*'):
        StreamingTextDataset(
            tokenizer=None,
            token_encoding_type=token_encoding_type,
            max_seq_len=2048,
            local='dataset/path',
            batch_size=1,
        )