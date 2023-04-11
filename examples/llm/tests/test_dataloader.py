# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import os
import shutil

import pytest
import torch
from omegaconf import OmegaConf as om

from examples.common.text_data import (ConcatenatedSequenceCollatorWrapper,
                                       build_text_dataloader)
from examples.llm.src import build_text_denoising_dataloader


def get_config(conf_path='yamls/mosaic_gpt/125m.yaml'):
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    with open(conf_path) as f:
        test_cfg = om.load(f)
    return test_cfg


def get_data_local(tokenizer_name, pretokenize):
    return f'my-copy-c4-{tokenizer_name}-pretokenize-{pretokenize}'


def get_abs_data_path(data_local):
    return os.path.join(os.getcwd(), data_local)


@pytest.mark.parametrize('tokenizer_name', ['gpt2', 'facebook/opt-125m'])
@pytest.mark.parametrize('pretokenize', [False, True])
def test_correct_padding(tokenizer_name, pretokenize, batch_size=4):
    if tokenizer_name == 'gpt2' and not pretokenize:
        pytest.xfail('Must pretokenize data if using "gpt2" tokenizer')

    data_local = get_data_local(tokenizer_name, pretokenize)
    split = 'val_small'
    tokenizer_args = {
        'gpt2': '--eos_text "<|endoftext|>"',
        'facebook/opt-125m': '--bos_text "</s>"'
    }[tokenizer_name]

    path = get_abs_data_path(data_local)
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


@pytest.mark.parametrize(('eos_token_id', 'bos_token_id'),
                         [(5, None), (None, 5),
                          pytest.param(5, 5, marks=pytest.mark.xfail)])
def test_sequence_id_wrapper(eos_token_id, bos_token_id):
    wrapper = ConcatenatedSequenceCollatorWrapper(
        lambda x: x,  # placeholder
        eos_token_id=eos_token_id,
        bos_token_id=bos_token_id,
    )

    batch = {'input_ids': torch.Tensor([[0, 1, 2, 5, 0, 1, 5, 0, 6]])}
    sequence_id = wrapper.get_sequence_id_from_batch(batch)

    if eos_token_id is not None:
        assert torch.equal(sequence_id,
                           torch.Tensor([[0, 0, 0, 0, 1, 1, 1, 2, 2]]))
    elif bos_token_id is not None:
        assert torch.equal(sequence_id,
                           torch.Tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2]]))
    else:
        raise NotImplementedError()


@pytest.mark.parametrize('decoder_only_format', [True, False])
@pytest.mark.parametrize('pretokenize', [True, False])
@pytest.mark.parametrize('packing_ratio', [None, 5.5])
def test_denoising_dataloader(decoder_only_format, pretokenize, packing_ratio):
    # Use the datasets just built in the last test
    tokenizer_name = 'facebook/opt-125m'
    data_local = get_data_local(tokenizer_name, pretokenize)
    path = get_abs_data_path(data_local)
    max_seq_len = 256 if decoder_only_format else 128

    if (decoder_only_format is False) and (packing_ratio is not None):
        pytest.xfail('packing_ratio only supported for decoder-only format.')

    cfg = {
        'name': 'text_denoising',
        'dataset': {
            'local': path,
            'remote': path,
            'split': 'val_small',
            'shuffle': False,
            'tokenizer_name': tokenizer_name,
            'max_seq_len': max_seq_len,
            'packing_ratio': packing_ratio,
            'predownload': 1000,
            'keep_zip': False,  # in case we need compressed files after testing
        },
        'mixture_of_denoisers': {
            'decoder_only_format': decoder_only_format,
            'span_mean_lengths_and_ratios': [[3, .15], [8, .5]],
            'sequence_mask_ratios': 0.25,
        },
        'drop_last': False,
        'num_workers': 0,
    }
    cfg = om.create(cfg)
    device_batch_size = 2

    expected_keys = ['input_ids', 'attention_mask', 'labels']
    if decoder_only_format:
        expected_keys += ['bidirectional_mask']
    else:
        expected_keys += ['decoder_attention_mask', 'decoder_input_ids']

    if packing_ratio is not None:
        expected_keys += ['sequence_id']

    loader = build_text_denoising_dataloader(cfg, device_batch_size)
    batch_ix = 0
    for batch in loader:
        for k in expected_keys:
            assert k in batch
            t = batch[k]
            assert t.shape[0] == device_batch_size
            assert t.shape[1] <= max_seq_len
        batch_ix += 1
        if batch_ix >= 5:
            break
