# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
import contextlib
import os
import pathlib
import random
import shutil
from argparse import Namespace
from contextlib import nullcontext as does_not_raise
from pathlib import Path
from typing import ContextManager, Literal, Optional, Union
from unittest.mock import MagicMock, patch

import catalogue
import numpy as np
import pytest
import torch
import transformers
from composer.utils import dist
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from streaming import MDSWriter

from llmfoundry import build_finetuning_dataloader
from llmfoundry.data import build_dataloader
from llmfoundry.data.finetuning.collator import (_HF_IGNORE_INDEX,
                                                 validate_target_settings)
from llmfoundry.data.finetuning.tasks import (DOWNLOADED_FT_DATASETS_DIRPATH,
                                              SUPPORTED_EXTENSIONS,
                                              dataset_constructor,
                                              is_valid_ift_example,
                                              tokenize_formatted_example)
from llmfoundry.data.text_data import (ConcatenatedSequenceCollatorWrapper,
                                       build_text_dataloader,
                                       get_tokens_per_batch_func)
from llmfoundry.utils.builders import build_tokenizer
# yapf: disable
from llmfoundry.utils.exceptions import (ConsecutiveRepeatedChatRolesError,
                                         IncorrectMessageKeyQuantityError,
                                         InvalidContentTypeError,
                                         InvalidLastChatMessageRoleError,
                                         InvalidPromptTypeError,
                                         InvalidResponseTypeError,
                                         InvalidRoleError,
                                         MisconfiguredHfDatasetError,
                                         NotEnoughDatasetSamplesError,
                                         TooManyKeysInExampleError,
                                         UnknownExampleTypeError)
# yapf: enable
from scripts.data_prep.convert_dataset_hf import main as main_hf
from scripts.data_prep.convert_finetuning_dataset import get_columns_and_format
from tests.data_utils import (make_tiny_conversation_ft_dataset,
                              make_tiny_ft_dataset)
from tests.test_utils import generate_exclusive_test_params


def get_config(conf_path: str = 'yamls/mpt/125m.yaml'):
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    with open(conf_path) as f:
        test_cfg = om.load(f)
    return test_cfg


def get_data_local(tokenizer_name: str, pretokenize: bool):
    return f'my-copy-c4-{tokenizer_name}-pretokenize-{pretokenize}'


def get_abs_data_path(data_local: str):
    return os.path.join(os.getcwd(), data_local)


def build_mock_ft_streaming_dataset(
        data_path: str,
        split: str,
        pretokenize: bool,
        backwards_compatibility_mode: bool,
        use_bytes: bool,
        tokenizer: Optional[transformers.PreTrainedTokenizerBase] = None):

    dataset = [{
        'prompt': 'This is just a test1',
        'response': 'Hello World1'
    }, {
        'prompt': 'This is just a test2',
        'response': 'Hello world2'
    }, {
        'prompt': 'This is just a test3',
        'response': 'Hello world3'
    }]

    output_path = os.path.join(data_path, split)

    if use_bytes and not backwards_compatibility_mode:
        raise ValueError(
            'use_bytes should only be true when using backwards_compatibility_mode'
        )

    # This is the old code-path, which we want to maintain test coverage of
    # for backwards compatibility
    if backwards_compatibility_mode:
        if pretokenize:
            if use_bytes:
                columns = {'input_ids': 'bytes', 'labels': 'bytes'}
            else:
                columns = {
                    'input_ids': 'ndarray:uint32',
                    'labels': 'ndarray:uint32'
                }
        else:
            columns = {'prompt': 'str', 'response': 'str'}

        with MDSWriter(columns=columns, out=output_path,
                       compression=None) as output_writer:
            for sample in dataset:
                if pretokenize:
                    sample = tokenize_formatted_example(sample,
                                                        tokenizer=tokenizer)
                    # Unpack the first turn to account for changes in `tokenize_formatted_example`
                    sample = sample['turns'][0]
                    sample_to_write = {}
                    for key in columns.keys():
                        if use_bytes:
                            sample_to_write[key] = np.asarray(
                                sample[key]).tobytes()
                        else:
                            sample_to_write[key] = np.asarray(sample[key],
                                                              dtype=np.uint32)
                    output_writer.write(sample_to_write)
                else:
                    output_writer.write(sample)
        return

    columns, data_format = get_columns_and_format(dataset, pretokenize,
                                                  lambda x: x)

    with MDSWriter(columns=columns, out=output_path,
                   compression=None) as output_writer:
        for sample in dataset:
            if pretokenize:
                sample = tokenize_formatted_example(sample, tokenizer=tokenizer)
                sample_to_write = {'turns': []}
                for turn in sample['turns']:
                    turn_to_write = {}
                    for key in ['input_ids', 'labels']:
                        turn_to_write[key] = list(turn[key])
                    sample_to_write['turns'].append(turn_to_write)
                output_writer.write(sample_to_write)
            else:
                if data_format == 'prompt_response':
                    encoded_sample = {
                        key: sample[key].encode('utf-8')
                        for key in ['prompt', 'response']
                    }
                else:
                    encoded_sample = sample
                output_writer.write(encoded_sample)


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


def test_invalid_jsonl_data():
    max_seq_len = 2
    decoder_only_format = True
    packing_ratio = 'auto'
    allow_pad_trimming = False
    cfg = {
        'name': 'finetuning',
        'dataset': {
            'hf_name': 'iamroot/chat_formatted_examples',
            'split': 'train_corrupted',
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
        tokenizer_name='gpt2',
        tokenizer_kwargs={'model_max_length': max_seq_len})

    device_batch_size = 2

    expected_keys = ['input_ids', 'attention_mask', 'labels']
    if not decoder_only_format:
        expected_keys += ['decoder_attention_mask', 'decoder_input_ids']

    with pytest.raises(MisconfiguredHfDatasetError):
        build_finetuning_dataloader(cfg, tokenizer,
                                    device_batch_size).dataloader


@pytest.mark.parametrize('use_chat_formatting', [True, False])
@pytest.mark.parametrize('decoder_only_format', [True, False])
@pytest.mark.parametrize('allow_pad_trimming', [True, False])
@pytest.mark.parametrize('packing_ratio', [10.0, None, 'auto'])
def test_finetuning_dataloader(use_chat_formatting: bool,
                               decoder_only_format: bool,
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
            'hf_name':
                'iamroot/chat_formatted_examples' if use_chat_formatting else
                'HuggingFaceH4/databricks_dolly_15k',
            'split':
                'train',
            'max_seq_len':
                max_seq_len,
            'decoder_only_format':
                decoder_only_format,
            'allow_pad_trimming':
                allow_pad_trimming,
            'packing_ratio':
                packing_ratio,
            'shuffle':
                True,
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
    if not decoder_only_format:
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


@pytest.mark.parametrize(
    'hf_name, hf_revision, expectation',
    [('HuggingFaceH4/databricks_dolly_15k', None, does_not_raise()),
     ('squad', '5fe18c', pytest.raises(FileNotFoundError))])
def test_finetuning_dataloader_safe_load(hf_name: str,
                                         hf_revision: Optional[str],
                                         expectation: ContextManager):
    cfg = DictConfig({
        'name': 'finetuning',
        'dataset': {
            'hf_name': hf_name,
            'split': 'train',
            'max_seq_len': 8,
            'decoder_only_format': True,
            'shuffle': True,
            'safe_load': True,
            'hf_kwargs': {
                'revision': hf_revision
            }
        },
        'drop_last': False,
        'num_workers': 0,
        'pin_memory': False,
        'prefetch_factor': None,
        'persistent_workers': False,
        'timeout': 0
    })

    tokenizer = build_tokenizer('gpt2', {})

    with expectation:
        _ = build_finetuning_dataloader(cfg, tokenizer, 1)

    # If no raised errors, we should expect downloaded files with only safe file types.
    if expectation == does_not_raise():
        download_dir = os.path.join(DOWNLOADED_FT_DATASETS_DIRPATH, hf_name)
        downloaded_files = [
            file for _, _, files in os.walk(download_dir) for file in files
        ]
        assert len(downloaded_files) > 0
        assert all(
            Path(file).suffix in SUPPORTED_EXTENSIONS
            for file in downloaded_files)


@pytest.mark.world_size(2)
@pytest.mark.gpu
@pytest.mark.parametrize('dataset_size', [4, 8])
@pytest.mark.parametrize('device_batch_size', [2, 4])
@pytest.mark.parametrize('drop_last', [True, False])
def test_finetuning_dataloader_small_data(dataset_size: int,
                                          device_batch_size: int,
                                          drop_last: bool):
    tokenizer_name = 'gpt2'
    max_seq_len = 2048
    tiny_dataset_folder_path = os.path.join(os.getcwd(), 'test-ift-data-small')
    tiny_dataset_path = os.path.join(tiny_dataset_folder_path, 'train.jsonl')
    if dist.get_global_rank() == 0:
        make_tiny_ft_dataset(
            path=tiny_dataset_path,
            size=dataset_size,
        )

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

    error_context = contextlib.nullcontext()
    if (dist.get_world_size() * device_batch_size > dataset_size) and drop_last:
        error_context = pytest.raises(NotEnoughDatasetSamplesError,
                                      match='Your dataset')

    with error_context:
        _ = build_finetuning_dataloader(cfg, tokenizer, device_batch_size)

    if dist.get_global_rank() == 0:
        shutil.rmtree(tiny_dataset_folder_path)


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
    if Path(destination).suffix == '.jsonl':
        make_tiny_ft_dataset(path=destination, size=16)
    else:
        raise FileNotFoundError(
            f'Test error in mock_get_file. {path} does not exist.')


@pytest.mark.parametrize('split', ['train', 'custom', 'custom-dash', 'data'])
def test_finetuning_dataloader_custom_split_remote(split: str):
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

    # Mock get_file to avoid downloading the file
    with patch('llmfoundry.data.finetuning.dataloader.get_file',
               wraps=mock_get_file) as f:
        _ = build_finetuning_dataloader(cfg, tokenizer, 4)
        for call in f.call_args_list:
            path_arg = call.kwargs['path']
            dest_arg = call.kwargs['destination']
            assert split in path_arg, 'split name should be downloaded verbatim'
            if '-' in split:
                assert split not in dest_arg, 'split name should have dashes replaced with underscores'
            else:
                assert split in dest_arg, 'split destination should match split name'


@pytest.mark.parametrize('pretokenize', [True, False])
@pytest.mark.parametrize('use_multiple_streams', [True, False])
@pytest.mark.parametrize(('backwards_compatibility_mode', 'use_bytes'),
                         [[False, False], [True, False], [True, True]])
def test_finetuning_dataloader_streaming(pretokenize: bool,
                                         use_multiple_streams: bool,
                                         backwards_compatibility_mode: bool,
                                         use_bytes: bool,
                                         tmp_path: pathlib.Path):
    max_seq_len = 2048

    tokenizer = build_tokenizer(
        tokenizer_name='gpt2',
        tokenizer_kwargs={'model_max_length': max_seq_len},
    )

    streams_config = {'streams': {}}
    num_streams = 2
    for i in range(num_streams):
        remote_path = os.path.join(tmp_path, f'remote_{i}')
        local_path = os.path.join(tmp_path, f'local_{i}')
        build_mock_ft_streaming_dataset(
            remote_path,
            'train',
            pretokenize,
            backwards_compatibility_mode=backwards_compatibility_mode,
            use_bytes=use_bytes,
            tokenizer=tokenizer)
        streams_config['streams'][f'stream_{i}'] = {
            'remote': remote_path,
            'local': local_path,
            'split': 'train'
        }

    cfg = {
        'name': 'finetuning',
        'dataset': {
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
    if use_multiple_streams:
        cfg['dataset'].update(streams_config)
    else:
        cfg['dataset'].update(streams_config['streams']['stream_0'])

    cfg = om.create(cfg)

    dataloader = build_finetuning_dataloader(cfg, tokenizer, 2).dataloader

    expected_keys = ['input_ids', 'labels']
    for batch in dataloader:
        for key in expected_keys:
            assert key in batch
            assert batch[key].shape[0] == 2
        break


@pytest.mark.parametrize('target_prompts', ['none', 'all', 'length>=2'])
@pytest.mark.parametrize('target_responses', ['last', 'all'])
@pytest.mark.parametrize('decoder_only_format', [True, False])
def test_finetuning_dataloader_is_valid_ift_example(
    target_prompts: str,
    target_responses: str,
    decoder_only_format: bool,
):
    if not decoder_only_format:
        if (target_prompts != 'none') or (target_responses != 'last'):
            pytest.xfail(
                'Must use "none" and "last" for target prompts and responses if not using decoder_only_format'
            )
    # This should pass
    validate_target_settings(target_prompts, target_responses,
                             decoder_only_format)

    max_seq_len = 4

    valid_example = {'turns': [{'input_ids': [2, 3, 5], 'labels': [8, 9, 7]}]}
    assert is_valid_ift_example(max_seq_len, target_prompts, target_responses,
                                decoder_only_format, valid_example)

    maybe_too_long_example = {
        'turns': [{
            'input_ids': [2, 3, 5],
            'labels': [8, 9, 7]
        }] * 3
    }
    if any([
            target_responses == 'all',
            target_prompts in {'all', 'length>=2'},
            decoder_only_format == False,
    ]):
        assert is_valid_ift_example(max_seq_len, target_prompts,
                                    target_responses, decoder_only_format,
                                    maybe_too_long_example)
    else:
        assert not is_valid_ift_example(max_seq_len, target_prompts,
                                        target_responses, decoder_only_format,
                                        maybe_too_long_example)

    another_maybe_too_long_example = {
        'turns': [{
            'input_ids': [2, 3, 5, 6, 8],
            'labels': [8, 9, 7]
        }]
    }
    if any([
            target_prompts in {'all', 'length>=2'},
            decoder_only_format == False,
    ]):
        assert is_valid_ift_example(max_seq_len, target_prompts,
                                    target_responses, decoder_only_format,
                                    another_maybe_too_long_example)
    else:
        assert not is_valid_ift_example(max_seq_len, target_prompts,
                                        target_responses, decoder_only_format,
                                        another_maybe_too_long_example)

    empty_input_example = {'turns': [{'input_ids': [], 'labels': [8, 9, 7]}]}
    assert not is_valid_ift_example(max_seq_len, target_prompts,
                                    target_responses, decoder_only_format,
                                    empty_input_example)

    empty_labels_example = {'turns': [{'input_ids': [1, 2], 'labels': []}]}
    assert not is_valid_ift_example(max_seq_len, target_prompts,
                                    target_responses, decoder_only_format,
                                    empty_labels_example)


invalid_prompt_response_params = [
    'add_bad_data_dropped', 'add_invalid_prompt_type',
    'add_invalid_response_type', 'add_unknown_example_type',
    'add_too_many_example_keys'
]


@pytest.mark.parametrize(
    ','.join(invalid_prompt_response_params),
    generate_exclusive_test_params(invalid_prompt_response_params))
def test_malformed_data(
    add_bad_data_dropped: bool,
    add_invalid_prompt_type: bool,
    add_invalid_response_type: bool,
    add_too_many_example_keys: bool,
    add_unknown_example_type: bool,
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
            add_invalid_prompt_type=add_invalid_prompt_type,
            add_invalid_response_type=add_invalid_response_type,
            add_unknown_example_type=add_unknown_example_type,
            add_too_many_example_keys=add_too_many_example_keys,
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

    error_context = contextlib.nullcontext()
    if add_invalid_prompt_type:
        error_context = pytest.raises(InvalidPromptTypeError,
                                      match='Expected prompt to be')
    if add_invalid_response_type:
        error_context = pytest.raises(InvalidResponseTypeError,
                                      match='Expected response to be')
    if add_unknown_example_type:
        error_context = pytest.raises(UnknownExampleTypeError,
                                      match='Unknown example type')
    if add_too_many_example_keys:
        error_context = pytest.raises(TooManyKeysInExampleError,
                                      match='Please specify exactly one.')

    with error_context:
        dl = build_finetuning_dataloader(cfg, tokenizer,
                                         device_batch_size).dataloader

    if not any(invalid_prompt_response_params):
        # +5 because we added samples with just bos/eos in each of prompt/response
        expected_num_batches = (dataset_size + 5) // device_batch_size

        actual_num_batches = 0
        for _ in dl:
            actual_num_batches += 1

        assert actual_num_batches == expected_num_batches


invalid_conversation_params = [
    'add_invalid_last_chat_message', 'add_invalid_message_key_quantity',
    'add_invalid_content_type', 'add_invalid_role', 'add_not_alternating_roles'
]


@pytest.mark.parametrize(
    ','.join(invalid_conversation_params),
    generate_exclusive_test_params(invalid_conversation_params))
def test_malformed_conversation_data(tmp_path: pathlib.Path,
                                     add_invalid_last_chat_message: bool,
                                     add_invalid_message_key_quantity: bool,
                                     add_invalid_content_type: bool,
                                     add_invalid_role: bool,
                                     add_not_alternating_roles: bool):
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
        make_tiny_conversation_ft_dataset(
            path=tiny_dataset_path,
            size=dataset_size,
            add_invalid_last_chat_message=add_invalid_last_chat_message,
            add_invalid_message_key_quantity=add_invalid_message_key_quantity,
            add_invalid_content_type=add_invalid_content_type,
            add_invalid_role=add_invalid_role,
            add_not_alternating_roles=add_not_alternating_roles,
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
    if add_invalid_last_chat_message:
        error_context = pytest.raises(InvalidLastChatMessageRoleError,
                                      match='Invalid last message role:')
    if add_invalid_message_key_quantity:
        error_context = pytest.raises(IncorrectMessageKeyQuantityError,
                                      match='Expected 2 keys in message')
    if add_invalid_content_type:
        error_context = pytest.raises(InvalidContentTypeError,
                                      match='Expected content to be')
    if add_invalid_role:
        error_context = pytest.raises(InvalidRoleError,
                                      match='Expected role to be one of')

    if add_not_alternating_roles:
        error_context = pytest.raises(ConsecutiveRepeatedChatRolesError,
                                      match='Conversation roles must alternate')

    with error_context:
        build_finetuning_dataloader(cfg, tokenizer,
                                    device_batch_size).dataloader


def test_finetune_dataloader_pure_pad_responses():
    """Test that dataloader can handle pure-pad responses."""

    @dataset_constructor.register('pad-response')
    def pad_preprocessing_function(  # type: ignore
            inp: dict[str, str]) -> dict[str, str]:
        """Split out prompt/response from text."""
        try:
            prompt, response = inp['text'].split('### Response:')
            prompt += '### Response:'
        except Exception as e:
            raise ValueError(
                f"Unable to extract prompt/response from 'text'={inp['text']}"
            ) from e
        return {'prompt': prompt, 'response': '|PAD|' * len(response.split())}

    cfg = om.create({
        'dataset': {
            'hf_name': 'tatsu-lab/alpaca',
            'preprocessing_fn': 'pad-response',
            'split': 'train',
            'packing_ratio': 18.0,
            'max_seq_len': 2048,
            'decoder_only_format': True,
            'allow_pad_trimming': False,
            'shuffle': True,
            'target_responses': 'last',
            'target_prompts': 'none',
        },
        'drop_last': False,
        'num_workers': 0,
        'pin_memory': False,
        'prefetch_factor': None,
        'persistent_workers': False,
        'timeout': 0
    })

    tokenizer_name = 'EleutherAI/gpt-neox-20b'
    tokenizer_kwargs = {
        'model_max_length': cfg.dataset.max_seq_len,
        'pad_token': '|PAD|'
    }
    tokenizer = build_tokenizer(tokenizer_name, tokenizer_kwargs)

    assert tokenizer('|PAD|').input_ids[0] == tokenizer.pad_token_id

    device_batch_size = 1
    dataloader = build_finetuning_dataloader(cfg, tokenizer,
                                             device_batch_size).dataloader

    # We should be able to iterate through this dataset without crashing
    for i, batch in enumerate(dataloader):
        # This check here just makes sure that the labels sequence is all padding
        # (except for an EOS at the end)
        for subseq in range(int(batch['sequence_id'][0].max()) + 1):
            is_subseq = batch['sequence_id'][0] == subseq
            labels = batch['labels'][
                0,
                torch.
                logical_and(is_subseq, batch['labels'][0] != _HF_IGNORE_INDEX)]
            assert all(labels[:-1] == tokenizer.pad_token_id)
        if i >= 20:
            break
        continue


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


@pytest.mark.parametrize('dataloader_type,tensor_input',
                         [('finetuning-hf', False),
                          ('finetuning-streaming', False), ('text', True),
                          ('text', False)])
@pytest.mark.parametrize('pad_token_id', [100, None])
@pytest.mark.parametrize('batch_size', [1, 8])
@pytest.mark.parametrize('model_max_length', [1024])
@pytest.mark.parametrize('padding_side', ['left'])
def test_token_counting_func_dataloader_setting(
        dataloader_type: str, tensor_input: bool, pad_token_id: Optional[int],
        batch_size: int, model_max_length: int, padding_side: str,
        monkeypatch: pytest.MonkeyPatch):
    gptt = transformers.AutoTokenizer.from_pretrained('gpt2')
    gptt.pad_token_id = pad_token_id if pad_token_id is not None else gptt.eos_token_id
    gptt.model_max_length = model_max_length
    gptt.padding_side = padding_side

    batch_strings = []
    expected_token_count = 0
    for _ in range(batch_size):
        # Get randomly different lengths if we are going to add padding
        sample_length = random.randint(
            1, model_max_length //
            4) if (pad_token_id is not None and
                   not tensor_input) else model_max_length // 4
        batch_strings.append(' '.join(['hello'] * sample_length))
        expected_token_count += sample_length

    batch_tokenized = [
        gptt(b, padding=True if pad_token_id is not None else False)
        for b in batch_strings
    ]

    if tensor_input:
        batch_tokenized = [
            torch.tensor(b['input_ids']) for b in batch_tokenized
        ]

    if dataloader_type in {'finetuning-hf', 'finetuning-streaming'}:
        for b in batch_tokenized:
            b['labels'] = b['input_ids'].copy()  # type: ignore
        batch_tokenized = [{'turns': [b]} for b in batch_tokenized]
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
    with pytest.raises(catalogue.RegistryError):
        _ = build_dataloader(cfg, tokenizer, 2)


invalid_conversation_params_sharegpt = [
    'add_invalid_last_chat_message', 'add_invalid_content_type',
    'add_invalid_role', 'add_not_alternating_roles'
]


@pytest.mark.parametrize(
    ','.join(invalid_conversation_params_sharegpt),
    generate_exclusive_test_params(invalid_conversation_params_sharegpt))
def test_sharegpt_format(tmp_path: pathlib.Path,
                         add_invalid_last_chat_message: bool,
                         add_invalid_content_type: bool, add_invalid_role: bool,
                         add_not_alternating_roles: bool):
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
        make_tiny_conversation_ft_dataset(
            path=tiny_dataset_path,
            size=dataset_size,
            add_invalid_last_chat_message=add_invalid_last_chat_message,
            add_invalid_message_key_quantity=False,
            add_invalid_content_type=add_invalid_content_type,
            add_invalid_role=add_invalid_role,
            add_not_alternating_roles=add_not_alternating_roles,
            use_messages_format=False,
        )

    cfg = {
        'name': 'finetuning',
        'dataset': {
            'hf_name': str(tiny_dataset_folder_path),
            'preprocessing_fn': 'teknium/OpenHermes-2.5',
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

    error_context = contextlib.nullcontext()
    if add_invalid_last_chat_message:
        error_context = pytest.raises(InvalidLastChatMessageRoleError,
                                      match='Invalid last message role:')
    if add_invalid_content_type:
        error_context = pytest.raises(InvalidContentTypeError,
                                      match='Expected content to be')
    if add_invalid_role:
        error_context = pytest.raises(InvalidRoleError,
                                      match='Expected role to be one of')

    if add_not_alternating_roles:
        error_context = pytest.raises(ConsecutiveRepeatedChatRolesError,
                                      match='Conversation roles must alternate')

    with error_context:
        build_finetuning_dataloader(cfg, tokenizer,
                                    device_batch_size).dataloader
