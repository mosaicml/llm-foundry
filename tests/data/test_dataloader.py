# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
import contextlib
import os
import pathlib
import random
import shutil
import tempfile
from argparse import Namespace
from typing import Literal, Optional, Union
from unittest.mock import MagicMock

import pytest
import torch
import transformers
from composer.utils import dist
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from streaming import MDSWriter

from llmfoundry import (build_finetuning_dataloader,
                        build_text_denoising_dataloader)
from llmfoundry.data import build_dataloader
from llmfoundry.data.finetuning.tasks import (_ALLOWED_PROMPT_KEYS,
                                              _ALLOWED_RESPONSE_KEYS,
                                              _tokenize_formatted_example)
from llmfoundry.data.text_data import (ConcatenatedSequenceCollatorWrapper,
                                       build_text_dataloader,
                                       get_tokens_per_batch_func)
from llmfoundry.utils.builders import build_tokenizer
from scripts.data_prep.convert_dataset_hf import main as main_hf
from tests.data_utils import make_tiny_ft_dataset


def get_config(conf_path: str = 'yamls/mpt/125m.yaml'):
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    with open(conf_path) as f:
        test_cfg = om.load(f)
    return test_cfg


def get_data_local(tokenizer_name: str, pretokenize: bool):
    return f'my-copy-c4-{tokenizer_name}-pretokenize-{pretokenize}'


def get_abs_data_path(data_local: str):
    return os.path.join(os.getcwd(), data_local)


def build_mock_ft_streaming_dataset(data_path: str, split: str):
    columns = {'prompt': 'str', 'response': 'str'}

    dataset = [{
        'prompt': 'This is just a test1',
        'response': 'Hello World1'
    }, {
        'prompt': 'This is just a test2',
        'response': 'Hello world2'
    }]

    output_path = os.path.join(data_path, split)

    with MDSWriter(columns=columns, out=output_path,
                   compression=None) as output_writer:
        for sample in dataset:
            output_writer.write(sample)


@pytest.mark.parametrize('tokenizer_name', ['gpt2', 'facebook/opt-125m'])
@pytest.mark.parametrize('pretokenize', [False, True])
def test_correct_padding(tokenizer_name: str,
                         pretokenize: bool,
                         batch_size: int = 4):
    if tokenizer_name == 'gpt2' and not pretokenize:
        pytest.xfail('Must pretokenize data if using "gpt2" tokenizer')

    data_local = get_data_local(tokenizer_name, pretokenize)
    split = 'val_xsmall'
    eos_text = ''
    bos_text = ''
    if tokenizer_name == 'gpt2':
        eos_text = '<|endoftext|>'
    elif tokenizer_name == 'facebook/opt-125m':
        bos_text = '</s>'

    path = get_abs_data_path(data_local)
    shutil.rmtree(path, ignore_errors=True)
    if pretokenize:
        main_hf(
            Namespace(
                **{
                    'dataset': 'c4',
                    'data_subset': 'en',
                    'splits': [split],
                    'out_root': path,
                    'compression': None,
                    'concat_tokens': 2048,
                    'tokenizer': tokenizer_name,
                    'tokenizer_kwargs': {},
                    'bos_text': bos_text,
                    'eos_text': eos_text,
                    'no_wrap': False,
                    'num_workers': None
                }))
    else:
        main_hf(
            Namespace(
                **{
                    'dataset': 'c4',
                    'data_subset': 'en',
                    'splits': [split],
                    'out_root': path,
                    'compression': None,
                    'concat_tokens': None,
                    'tokenizer': tokenizer_name,
                    'tokenizer_kwargs': {},
                    'bos_text': bos_text,
                    'eos_text': eos_text,
                    'no_wrap': False,
                    'num_workers': None,
                }))
    if not os.path.isdir(path):
        raise RuntimeError(f'c4 dataset at {path} not set up as expected')

    test_cfg = get_config(
        conf_path='scripts/train/yamls/pretrain/mpt-125m.yaml')
    test_cfg.data_local = data_local
    test_cfg.eval_loader.dataset.split = split
    test_cfg.dataset = om.create({
        'num_canonical_nodes': 1,
        'predownload': 3000,
    })

    tokenizer = build_tokenizer(
        tokenizer_name=tokenizer_name,
        tokenizer_kwargs={},
    )

    # Dataloaders
    eval_loader = build_text_dataloader(
        test_cfg.eval_loader,
        tokenizer,
        batch_size,
    ).dataloader
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
def test_sequence_id_wrapper(eos_token_id: Optional[int],
                             bos_token_id: Optional[int]):
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
def test_denoising_dataloader(decoder_only_format: bool, pretokenize: bool,
                              packing_ratio: Optional[float]):
    # Use the datasets just built in the last test
    tokenizer_name = 'facebook/opt-125m'
    data_local = get_data_local(tokenizer_name, pretokenize)
    path = get_abs_data_path(data_local)
    max_seq_len = 256 if decoder_only_format else 128

    if (decoder_only_format is False) and (packing_ratio is not None):
        pytest.xfail('packing_ratio only supported for decoder-only format.')

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = {
            'name': 'text_denoising',
            'dataset': {
                'local': tmpdir,
                'remote': path,
                'split': 'val_xsmall',
                'shuffle': False,
                'max_seq_len': max_seq_len,
                'packing_ratio': packing_ratio,
                'predownload': 1000,
                'keep_zip': False,
                'num_workers': None
            },
            'mixture_of_denoisers': {
                'decoder_only_format': decoder_only_format,
                'span_mean_lengths_and_ratios': [[3, .15], [8, .5]],
                'sequence_mask_ratios': 0.25,
            },
            'drop_last': False,
            'num_workers': 4,
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

        tokenizer = build_tokenizer(
            tokenizer_name=tokenizer_name,
            tokenizer_kwargs={'model_max_length': max_seq_len})

        loader = build_text_denoising_dataloader(cfg, tokenizer,
                                                 device_batch_size).dataloader
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


@pytest.mark.parametrize('decoder_only_format', [True, False])
@pytest.mark.parametrize('allow_pad_trimming', [True, False])
@pytest.mark.parametrize('packing_ratio', [10.0, None, 'auto'])
def test_finetuning_dataloader(decoder_only_format: bool,
                               allow_pad_trimming: bool,
                               packing_ratio: Optional[Union[float,
                                                             Literal['auto']]]):
    if (decoder_only_format is False) and (packing_ratio is not None):
        pytest.xfail('packing_ratio only supported for decoder-only format.')

    tokenizer_name = 'gpt2' if decoder_only_format else 't5-base'
    max_seq_len = 2048 if decoder_only_format else 1024

    cfg = {
        'name': 'finetuning',
        'dataset': {
            'hf_name': 'HuggingFaceH4/databricks_dolly_15k',
            'split': 'train',
            'max_seq_len': max_seq_len,
            'decoder_only_format': decoder_only_format,
            'allow_pad_trimming': allow_pad_trimming,
            'packing_ratio': packing_ratio,
            'shuffle': True,
        },
        'drop_last': False,
        'num_workers': 0,
        'pin_memory': False,
        'prefetch_factor': None,
        'persistent_workers': False,
        'timeout': 0
    }

    cfg = om.create(cfg)

    tokenizer = build_tokenizer(
        tokenizer_name=tokenizer_name,
        tokenizer_kwargs={'model_max_length': max_seq_len})

    device_batch_size = 2

    expected_keys = ['input_ids', 'attention_mask', 'labels']
    if decoder_only_format:
        expected_keys += ['bidirectional_mask']
    else:
        expected_keys += ['decoder_attention_mask', 'decoder_input_ids']

    loader = build_finetuning_dataloader(cfg, tokenizer,
                                         device_batch_size).dataloader
    batch_ix = 0
    for batch in loader:
        for k in expected_keys:
            assert k in batch
            t = batch[k]
            assert t.shape[
                0] == device_batch_size, f'{k} has incorrect batch size'
            assert t.shape[1] <= max_seq_len, f'{k} exceeds max_seq_len'
        batch_ix += 1
        if batch_ix >= 3:
            break


@pytest.mark.world_size(2)
@pytest.mark.gpu
@pytest.mark.parametrize('dataset_size', [4, 8])
@pytest.mark.parametrize('device_batch_size', [2, 4])
@pytest.mark.parametrize('drop_last', [True, False])
@pytest.mark.parametrize('invalid_dataset', [True, False])
def test_finetuning_dataloader_small_data(dataset_size: int,
                                          device_batch_size: int,
                                          drop_last: bool,
                                          invalid_dataset: bool):
    tokenizer_name = 'gpt2'
    max_seq_len = 2048
    tiny_dataset_folder_path = os.path.join(os.getcwd(), 'test-ift-data-small')
    tiny_dataset_path = os.path.join(tiny_dataset_folder_path, 'train.jsonl')
    if dist.get_global_rank() == 0:
        make_tiny_ft_dataset(path=tiny_dataset_path,
                             size=dataset_size,
                             add_bad_data_error=invalid_dataset)

    cfg = {
        'name': 'finetuning',
        'dataset': {
            'hf_name': tiny_dataset_folder_path,
            'split': 'train',
            'max_seq_len': max_seq_len,
            'decoder_only_format': True,
            'allow_pad_trimming': False,
            'packing_ratio': None,
            'shuffle': True,
        },
        'drop_last': drop_last,
        'num_workers': 4,
        'pin_memory': False,
        'prefetch_factor': 2,
        'persistent_workers': False,
        'timeout': 0
    }

    cfg = om.create(cfg)

    tokenizer = build_tokenizer(
        tokenizer_name=tokenizer_name,
        tokenizer_kwargs={'model_max_length': max_seq_len},
    )

    expected_keys = ['input_ids', 'attention_mask', 'labels']
    expected_keys += ['bidirectional_mask']

    error_context = contextlib.nullcontext()
    if (dist.get_world_size() * device_batch_size > dataset_size) and drop_last:
        error_context = pytest.raises(ValueError, match='Your dataset')
    if invalid_dataset:
        error_context = pytest.raises(TypeError,
                                      match='Unable to tokenize example')

    with error_context:
        _ = build_finetuning_dataloader(cfg, tokenizer, device_batch_size)

    if dist.get_global_rank() == 0:
        shutil.rmtree(tiny_dataset_folder_path)


def test_tokenize_example_malformed():
    no_keys = {}
    no_prompt_key = {'response': 'response'}
    no_response_key = {'prompt': 'prompt'}
    extra_keys_with_prompt = {'prompt': 'prompt', 'extra': 'extra'}
    extra_keys_with_response = {'response': 'response', 'extra': 'extra'}
    multiple_allowed_response_keys = {
        'prompt': 'prompt',
        'response': 'response',
        'completion': 'completion'
    }

    malformed_examples = [
        no_keys, no_prompt_key, no_response_key, extra_keys_with_prompt,
        extra_keys_with_response, multiple_allowed_response_keys
    ]

    for example in malformed_examples:
        with pytest.raises(KeyError):
            _tokenize_formatted_example(example, MagicMock())


def test_tokenize_example_well_formed():
    tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')

    for prompt_key in _ALLOWED_PROMPT_KEYS:
        for response_key in _ALLOWED_RESPONSE_KEYS:
            example = {prompt_key: 'prompt', response_key: 'response'}
            tokenized_example = _tokenize_formatted_example(example, tokenizer)
            assert 'input_ids' in tokenized_example
            assert 'labels' in tokenized_example


@pytest.mark.parametrize('split', ['train', 'custom', 'data'])
def test_finetuning_dataloader_custom_split(tmp_path: pathlib.Path, split: str):
    tokenizer_name = 'gpt2'
    max_seq_len = 2048
    tiny_dataset_folder_path = str(tmp_path)
    tiny_dataset_path = os.path.join(tiny_dataset_folder_path, 'data',
                                     f'{split}-00000-of-00001.jsonl')
    if dist.get_global_rank() == 0:
        make_tiny_ft_dataset(path=tiny_dataset_path, size=16)

    cfg = {
        'name': 'finetuning',
        'dataset': {
            'hf_name': tiny_dataset_folder_path,
            'split': split,
            'max_seq_len': max_seq_len,
            'decoder_only_format': True,
            'allow_pad_trimming': False,
            'packing_ratio': None,
            'shuffle': True,
        },
        'drop_last': False,
        'num_workers': 4,
        'pin_memory': False,
        'prefetch_factor': 2,
        'persistent_workers': False,
        'timeout': 0
    }

    cfg = om.create(cfg)

    tokenizer = build_tokenizer(
        tokenizer_name=tokenizer_name,
        tokenizer_kwargs={'model_max_length': max_seq_len},
    )

    _ = build_finetuning_dataloader(cfg, tokenizer, 4)


def mock_get_file(path: str, destination: str, overwrite: bool = False):
    make_tiny_ft_dataset(path=destination, size=16)


@pytest.mark.parametrize('split', ['train', 'custom', 'custom-dash', 'data'])
def test_finetuning_dataloader_custom_split_remote(
        tmp_path: pathlib.Path, split: str, monkeypatch: pytest.MonkeyPatch):
    tokenizer_name = 'gpt2'
    max_seq_len = 2048

    cfg = {
        'name': 'finetuning',
        'dataset': {
            'hf_name': 's3://test-bucket/path/to/data',
            'split': split,
            'max_seq_len': max_seq_len,
            'decoder_only_format': True,
            'allow_pad_trimming': False,
            'packing_ratio': None,
            'shuffle': True,
        },
        'drop_last': False,
        'num_workers': 4,
        'pin_memory': False,
        'prefetch_factor': 2,
        'persistent_workers': False,
        'timeout': 0
    }

    cfg = om.create(cfg)

    tokenizer = build_tokenizer(
        tokenizer_name=tokenizer_name,
        tokenizer_kwargs={'model_max_length': max_seq_len},
    )

    with monkeypatch.context() as m:
        m.setattr('llmfoundry.data.finetuning.dataloader.get_file',
                  mock_get_file)
        _ = build_finetuning_dataloader(cfg, tokenizer, 4)


def test_finetuning_dataloader_streaming(tmp_path: pathlib.Path):
    max_seq_len = 2048

    remote_path = os.path.join(tmp_path, 'remote')
    local_path = os.path.join(tmp_path, 'local')

    build_mock_ft_streaming_dataset(remote_path, 'train')

    cfg = {
        'name': 'finetuning',
        'dataset': {
            'remote': remote_path,
            'local': local_path,
            'split': 'train',
            'max_seq_len': 2048,
            'decoder_only_format': True,
            'allow_pad_trimming': False,
            'packing_ratio': None,
            'shuffle': True,
        },
        'drop_last': False,
        'num_workers': 4,
        'pin_memory': False,
        'prefetch_factor': 2,
        'persistent_workers': False,
        'timeout': 0
    }

    cfg = om.create(cfg)

    tokenizer = build_tokenizer(
        tokenizer_name='gpt2',
        tokenizer_kwargs={'model_max_length': max_seq_len},
    )

    _ = build_finetuning_dataloader(cfg, tokenizer, 4)


@pytest.mark.parametrize('add_bad_data_dropped', [True, False])
@pytest.mark.parametrize('add_bad_data_error', [True, False])
def test_malformed_data(
    add_bad_data_dropped: bool,
    add_bad_data_error: bool,
    tmp_path: pathlib.Path,
):
    tokenizer_name = 'mosaicml/mpt-7b'
    max_seq_len = 2048
    dataset_size = 5
    device_batch_size = 5
    tiny_dataset_folder_path = tmp_path
    tiny_dataset_path = str(tiny_dataset_folder_path / 'train.jsonl')

    tokenizer = build_tokenizer(
        tokenizer_name=tokenizer_name,
        tokenizer_kwargs={'model_max_length': max_seq_len},
    )
    tokenizer.add_special_tokens({
        'pad_token': '<pad>',
        'bos_token': '<bos>',
        'eos_token': '<eos>',
    })

    if dist.get_global_rank() == 0:
        make_tiny_ft_dataset(
            path=tiny_dataset_path,
            size=dataset_size,
            add_bad_data_dropped=add_bad_data_dropped,
            add_bad_data_error=add_bad_data_error,
            add_just_bos_eos_pad=True,
            pad_token=tokenizer.pad_token,
            start_token=tokenizer.bos_token,
            end_token=tokenizer.eos_token,
        )

    cfg = {
        'name': 'finetuning',
        'dataset': {
            'hf_name': str(tiny_dataset_folder_path),
            'split': 'train',
            'max_seq_len': max_seq_len,
            'decoder_only_format': True,
            'allow_pad_trimming': False,
            'packing_ratio': None,
            'shuffle': True,
        },
        'drop_last': False,
        'num_workers': 0,
        'prefetch_factor': None,
        'pin_memory': False,
        'persistent_workers': False,
        'timeout': 0
    }

    cfg = om.create(cfg)

    expected_keys = ['input_ids', 'attention_mask', 'labels']
    expected_keys += ['bidirectional_mask']

    error_context = contextlib.nullcontext()
    if add_bad_data_error:
        error_context = pytest.raises(TypeError,
                                      match='Unable to tokenize example')

    with error_context:
        dl = build_finetuning_dataloader(cfg, tokenizer,
                                         device_batch_size).dataloader

    if not add_bad_data_error:
        # +5 because we added samples with just bos/eos in each of prompt/response
        expected_num_batches = (dataset_size + 5) // device_batch_size

        actual_num_batches = 0
        for _ in dl:
            actual_num_batches += 1

        assert actual_num_batches == expected_num_batches


@pytest.mark.parametrize('pad_token_id', [0, 100, 1000])
@pytest.mark.parametrize('batch_size', [1, 8, 16])
@pytest.mark.parametrize('model_max_length', [1024, 2048])
@pytest.mark.parametrize('padding_side', ['left', 'right'])
@pytest.mark.parametrize('add_decoder_input_ids', [True, False])
def test_token_counting_func(pad_token_id: int, batch_size: int,
                             model_max_length: int, padding_side: str,
                             add_decoder_input_ids: bool):
    gptt = transformers.AutoTokenizer.from_pretrained('gpt2')
    gptt.pad_token_id = pad_token_id
    gptt.model_max_length = model_max_length
    gptt.padding_side = padding_side

    batch_strings = []
    expected_token_count = 0
    for _ in range(batch_size):
        sample_length = random.randint(1, model_max_length)
        batch_strings.append(' '.join(['hello'] * sample_length))
        expected_token_count += sample_length

    batch_tokenized = gptt(batch_strings, padding=True, return_tensors='pt')

    if add_decoder_input_ids:
        decoder_batch_strings = []
        decoder_expected_token_count = 0
        for _ in range(batch_size):
            sample_length = random.randint(1, model_max_length)
            decoder_batch_strings.append(' '.join(['hello'] * sample_length))
            decoder_expected_token_count += sample_length
            expected_token_count += sample_length
        batch_tokenized['decoder_attention_mask'] = gptt(
            decoder_batch_strings, padding=True,
            return_tensors='pt')['attention_mask']

    token_counting_func = get_tokens_per_batch_func(
        decoder_only=not add_decoder_input_ids)

    actual_token_count = token_counting_func(batch_tokenized)

    assert actual_token_count == expected_token_count


@pytest.mark.parametrize(
    'dataloader_type',
    ['finetuning-hf', 'finetuning-streaming', 'denoising', 'text'])
@pytest.mark.parametrize('pad_token_id', [100, None])
@pytest.mark.parametrize('batch_size', [1, 8])
@pytest.mark.parametrize('model_max_length', [1024])
@pytest.mark.parametrize('padding_side', ['left'])
def test_token_counting_func_dataloader_setting(
        dataloader_type: str, pad_token_id: Optional[int], batch_size: int,
        model_max_length: int, padding_side: str,
        monkeypatch: pytest.MonkeyPatch):
    gptt = transformers.AutoTokenizer.from_pretrained('gpt2')
    gptt.pad_token_id = pad_token_id if pad_token_id is not None else gptt.eos_token_id
    gptt.model_max_length = model_max_length
    gptt.padding_side = padding_side

    batch_strings = []
    expected_token_count = 0
    for _ in range(batch_size):
        sample_length = random.randint(
            1, model_max_length //
            4) if pad_token_id is not None else model_max_length // 4
        batch_strings.append(' '.join(['hello'] * sample_length))
        expected_token_count += sample_length

    batch_tokenized = [
        gptt(b, padding=True if pad_token_id is not None else False)
        for b in batch_strings
    ]

    if dataloader_type == 'denoising':
        expected_token_count += 2 * batch_size  # for the two eos tokens
        expected_token_count += 5 * batch_size  # for the corruption prefix tokens

    if dataloader_type in {'finetuning-hf', 'finetuning-streaming'}:
        for b in batch_tokenized:
            b['labels'] = b['input_ids'].copy()
        expected_token_count *= 2
        expected_token_count += 1 * batch_size  # for the eos token

    common_args = {
        'drop_last': False,
        'num_workers': 0,
        'prefetch_factor': None,
        'pin_memory': False,
        'persistent_workers': False,
        'timeout': 0
    }

    if dataloader_type == 'finetuning-hf':
        cfg = DictConfig({
            'name': 'finetuning',
            'dataset': {
                'hf_name': 'dummy-path',
                'split': 'train',
                'max_seq_len': model_max_length,
                'decoder_only_format': True,
                'allow_pad_trimming': False,
                'packing_ratio': None,
                'shuffle': True,
            },
            **common_args
        })
        monkeypatch.setattr(
            'llmfoundry.data.finetuning.tasks.DatasetConstructor.build_from_hf',
            lambda *args, **kwargs: [])
        dl = build_finetuning_dataloader(cfg, gptt, batch_size)
    elif dataloader_type == 'finetuning-streaming':
        cfg = DictConfig({
            'name': 'finetuning',
            'dataset': {
                'remote': 'dummy-path',
                'local': 'dummy-path',
                'split': 'train',
                'max_seq_len': model_max_length,
                'decoder_only_format': True,
                'allow_pad_trimming': False,
                'packing_ratio': None,
                'shuffle': True,
            },
            **common_args
        })
        monkeypatch.setattr(
            'llmfoundry.data.finetuning.tasks.DatasetConstructor.build_from_streaming',
            lambda *args, **kwargs: [])
        dl = build_finetuning_dataloader(cfg, gptt, batch_size)
    elif dataloader_type == 'text':
        cfg = DictConfig({
            'name': 'text',
            'dataset': {
                'local': 'dummy-path',
                'remote': 'dummy-path',
                'split': 'train',
                'max_seq_len': model_max_length,
                'shuffle': True,
                'shuffle_seed': 0,
            },
            **common_args
        })
        ds_mock = MagicMock()
        ds_mock.tokenizer = gptt
        monkeypatch.setattr('llmfoundry.data.text_data.StreamingTextDataset',
                            lambda *args, **kwargs: ds_mock)
        dl = build_text_dataloader(cfg, gptt, batch_size)
    elif dataloader_type == 'denoising':
        cfg = DictConfig({
            'name': 'text_denoising',
            'dataset': {
                'local': 'dummy-path',
                'remote': 'dummy-path',
                'split': 'val_xsmall',
                'shuffle': False,
                'max_seq_len': model_max_length,
                'packing_ratio': None,
                'predownload': 1000,
                'keep_zip': False,
                'num_workers': None
            },
            'mixture_of_denoisers': {
                'decoder_only_format': False,
                'span_mean_lengths_and_ratios': None,
                'sequence_mask_ratios': 0.25,
            },
            **common_args
        })
        monkeypatch.setattr('llmfoundry.data.denoising.StreamingTextDataset',
                            lambda *args, **kwargs: MagicMock())
        dl = build_text_denoising_dataloader(cfg, gptt, batch_size)
    else:
        raise NotImplementedError()

    cfg = om.create(cfg)

    batch_collated = dl.dataloader.collate_fn(batch_tokenized)  # type: ignore
    actual_token_count = dl.get_num_tokens_in_batch(batch_collated)

    assert actual_token_count == expected_token_count


def test_build_unknown_dataloader():
    cfg = DictConfig({
        'name': 'unknown',
    })
    tokenizer = MagicMock()
    with pytest.raises(ValueError,
                       match='Expected dataloader name to be one of'):
        _ = build_dataloader(cfg, tokenizer, 2)
