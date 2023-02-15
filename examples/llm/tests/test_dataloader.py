# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import os
import shutil

import pytest
import torch
from omegaconf import OmegaConf as om

from examples.common.text_data import build_text_dataloader


def get_config(conf_path='yamls/mosaic_gpt/125m.yaml'):
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    with open(conf_path) as f:
        test_cfg = om.load(f)
    return test_cfg


@pytest.mark.parametrize('tokenizer_name', ['gpt2', 'facebook/opt-125m'])
@pytest.mark.parametrize('pretokenize', [False, True])
def test_correct_padding(tokenizer_name, pretokenize, batch_size=4):
    if tokenizer_name == 'gpt2' and not pretokenize:
        pytest.xfail('Must pretokenize data if using "gpt2" tokenizer')

    data_local = f'my-copy-c4-{tokenizer_name}-pretokenize-{pretokenize}'
    split = 'val_small'
    tokenizer_args = {
        'gpt2': '--eos_text "<|endoftext|>"',
        'facebook/opt-125m': '--bos_text "</s>"'
    }[tokenizer_name]

    path = os.path.join(os.getcwd(), data_local)
    shutil.rmtree(path, ignore_errors=True)
    if pretokenize:
        os.system(
            f'python ../common/convert_dataset.py --dataset c4 --data_subset en --out_root {path} --splits val_small --concat_tokens 2048 --tokenizer {tokenizer_name} {tokenizer_args}'
        )
    else:
        os.system(
            f'python ../common/convert_dataset.py --dataset c4 --data_subset en --out_root {path} --splits val_small'
        )
    if not os.path.isdir(path):
        raise RuntimeError(f'c4 dataset at {path} not set up as expected')

    test_cfg = get_config(conf_path='yamls/mosaic_gpt/125m.yaml')
    test_cfg.tokenizer_name = tokenizer_name
    test_cfg.data_local = data_local
    test_cfg.eval_loader.dataset.split = split

    # Dataloaders
    eval_loader = build_text_dataloader(test_cfg.eval_loader, batch_size)
    batch = next(iter(eval_loader))

    assert batch['input_ids'].shape == torch.Size([batch_size, 2048])
    assert batch['input_ids'].type() == 'torch.LongTensor'

    # we follow the convention (from huggingface) that non-attended tokens are 0 in the attn mask and -100 in the labels
    attention_mask = batch.get(
        'attention_mask', torch.ones_like(batch['input_ids'], dtype=torch.bool))
    a = attention_mask == 0
    b = batch['labels'] == -100
    assert torch.equal(a, b)
