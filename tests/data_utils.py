# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import pathlib
import shutil
from argparse import Namespace
from typing import Optional

from omegaconf import DictConfig
from omegaconf import OmegaConf as om

from scripts.data_prep.convert_dataset_hf import main as main_hf  # noqa: E402
from scripts.data_prep.convert_dataset_json import \
    main as main_json  # noqa: E402


def make_tiny_ft_dataset(
    path: str,
    size: int = 4,
    add_bad_data_dropped: bool = False,
    add_bad_data_error: bool = False,
    add_just_bos_eos_pad: bool = False,
    pad_token: Optional[str] = None,
    start_token: Optional[str] = None,
    end_token: Optional[str] = None,
):
    good_sample = {'prompt': 'hello', 'response': 'goodbye'}
    samples = [good_sample] * size
    if add_bad_data_dropped:
        if pad_token is None:
            raise ValueError(
                'pad_token, start_token, and end_token must be specified if add_bad_data is True'
            )
        # empty prompt
        samples.append({'prompt': '', 'response': 'goodbye'})
        # empty response
        samples.append({'prompt': 'hello', 'response': ''})
        # response just pad
        samples.append({'prompt': 'hello', 'response': pad_token})
        # response just pad multiple times
        samples.append({'prompt': 'hello', 'response': pad_token * 3})

    if add_bad_data_error:
        # prompt just None
        samples.append({
            'prompt': None,
            'response': 'goodbye'
        })  # type: ignore (intentional test)
        # response just None
        samples.append({
            'prompt': 'hello',
            'response': None
        })  # type: ignore (intentional test)

    if add_just_bos_eos_pad:
        if pad_token is None or start_token is None or end_token is None:
            raise ValueError(
                'pad_token, start_token, and end_token must be specified if add_just_bos_eos is True'
            )
        # prompt just start
        samples.append({'prompt': start_token, 'response': 'goodbye'})
        # response just start
        samples.append({'prompt': 'hello', 'response': start_token})
        # prompt just end
        samples.append({'prompt': end_token, 'response': 'goodbye'})
        # response just end
        samples.append({'prompt': 'hello', 'response': end_token})
        # prompt just pad
        samples.append({'prompt': pad_token, 'response': 'goodbye'})

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as _f:
        for sample in samples:
            _f.write(json.dumps(sample))
            _f.write('\n')


def create_c4_dataset_xxsmall(path: pathlib.Path) -> str:
    """Creates a small mocked version of the C4 dataset."""
    c4_dir = os.path.join(path, f'my-copy-c4')
    downloaded_split = 'val_xxsmall'  # very fast to convert

    # Hyperparameters from https://github.com/mosaicml/llm-foundry/blob/340a56658560ebceb2a3aa69d6e37813e415acd0/README.md#L188
    main_hf(
        Namespace(
            **{
                'dataset': 'c4',
                'data_subset': 'en',
                'splits': [downloaded_split],
                'out_root': c4_dir,
                'compression': None,
                'concat_tokens': 2048,
                'tokenizer': 'EleutherAI/gpt-neox-20b',
                'tokenizer_kwargs': {},
                'bos_text': '',
                'eos_text': '<|endoftext|>',
                'no_wrap': False,
                'num_workers': 8
            }))

    # copy the small downloaded_split to other c4 splits for mocking purposes
    mocked_splits = ['train', 'val']
    for mocked_split in mocked_splits:
        shutil.copytree(os.path.join(c4_dir, 'val_xxsmall'),
                        os.path.join(c4_dir, mocked_split))
    assert os.path.exists(c4_dir)
    return c4_dir


def create_arxiv_dataset(path: pathlib.Path) -> str:
    """Creates an arxiv dataset."""
    arxiv_dir = os.path.join(path, f'my-copy-arxiv')
    downloaded_split = 'train'

    arxiv_path = 'data_prep/example_data/arxiv.jsonl'
    if not os.getcwd().endswith('scripts'):
        arxiv_path = os.path.join('scripts', arxiv_path)

    main_json(
        Namespace(
            **{
                'path': arxiv_path,
                'out_root': arxiv_dir,
                'compression': None,
                'split': downloaded_split,
                'concat_tokens': None,
                'bos_text': None,
                'eos_text': None,
                'no_wrap': False,
                'num_workers': None
            }))

    return arxiv_dir


def gpt_tiny_cfg(dataset_name: str, device: str):
    """Create gpt tiny cfg."""
    from tests.fixtures.autouse import REPO_DIR
    conf_path: str = os.path.join(
        REPO_DIR,
        'scripts/train/yamls/pretrain/testing.yaml',
    )
    with open(conf_path) as f:
        test_cfg = om.load(f)
    assert isinstance(test_cfg, DictConfig)

    test_cfg.data_local = dataset_name
    test_cfg.global_train_batch_size = 8
    test_cfg.device_eval_batch_size = 4
    test_cfg.device_train_microbatch_size = 4
    test_cfg.max_duration = '4ba'
    test_cfg.eval_interval = '4ba'
    test_cfg.run_name = 'gpt-mini-integration-test'

    if device == 'cpu':
        test_cfg.model.init_device = 'cpu'
        test_cfg.fsdp_config = None
        test_cfg.model.attn_config.attn_impl = 'torch'
        test_cfg.model.loss_fn = 'torch_crossentropy'
        test_cfg.precision = 'fp32'

    return test_cfg
