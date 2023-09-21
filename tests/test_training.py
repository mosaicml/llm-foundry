# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
import os
import shutil
import sys
from argparse import Namespace
from typing import Any

import pytest
from composer.loggers import InMemoryLogger
from omegaconf import DictConfig, ListConfig
from omegaconf import OmegaConf as om

# Add repo root to path so we can import scripts and test it
repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(repo_dir)

from scripts.data_prep.convert_dataset_hf import main as main_hf  # noqa: E402
from scripts.train.train import main  # noqa: E402


def create_c4_dataset_xsmall(prefix: str) -> str:
    """Creates a small mocked version of the C4 dataset."""
    c4_dir = os.path.join(os.getcwd(), f'my-copy-c4-{prefix}')
    shutil.rmtree(c4_dir, ignore_errors=True)
    downloaded_split = 'val_xsmall'  # very fast to convert

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
        shutil.copytree(os.path.join(c4_dir, 'val_xsmall'),
                        os.path.join(c4_dir, mocked_split))
    assert os.path.exists(c4_dir)
    return c4_dir


def gpt_tiny_cfg(dataset_name: str, device: str):
    """Create gpt tiny cfg."""
    conf_path: str = os.path.join(repo_dir,
                                  'scripts/train/yamls/pretrain/testing.yaml')
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


@pytest.fixture(autouse=False)
def set_correct_cwd():
    if not os.getcwd().endswith('llm-foundry/scripts'):
        os.chdir('scripts')

    yield

    if os.getcwd().endswith('llm-foundry/scripts'):
        os.chdir('..')


def test_train_gauntlet(set_correct_cwd: Any):
    """Test training run with a small dataset."""
    dataset_name = create_c4_dataset_xsmall('cpu-gauntlet')
    test_cfg = gpt_tiny_cfg(dataset_name, 'cpu')
    test_cfg.icl_tasks = ListConfig([
        DictConfig({
            'label':
                'lambada_openai',
            'dataset_uri':
                'eval/local_data/language_understanding/lambada_openai_small.jsonl',
            'num_fewshot': [0],
            'icl_task_type':
                'language_modeling'
        })
    ])
    test_cfg.icl_subset_num_batches = 1  # -1 to evaluate on all batches

    test_cfg.eval_gauntlet = DictConfig({
        'weighting':
            'EQUAL',
        'subtract_random_baseline':
            True,
        'rescale_accuracy':
            True,
        'categories':
            ListConfig([
                DictConfig({
                    'name':
                        'language_understanding_lite',
                    'benchmarks':
                        ListConfig([
                            DictConfig({
                                'name': 'lambada_openai',
                                'num_fewshot': 0,
                                'random_baseline': 0.0
                            })
                        ])
                })
            ])
    })

    test_cfg.icl_seq_len = 128
    test_cfg.max_duration = '1ba'
    test_cfg.eval_interval = '1ba'
    test_cfg.loggers = DictConfig({'inmemory': DictConfig({})})
    trainer = main(test_cfg)

    assert isinstance(trainer.logger.destinations, tuple)

    assert len(trainer.logger.destinations) > 0
    inmemorylogger = trainer.logger.destinations[
        0]  # pyright: ignore [reportGeneralTypeIssues]
    assert isinstance(inmemorylogger, InMemoryLogger)
    assert 'icl/metrics/eval_gauntlet/average' in inmemorylogger.data.keys()
    assert isinstance(inmemorylogger.data['icl/metrics/eval_gauntlet/average'],
                      list)
    assert len(inmemorylogger.data['icl/metrics/eval_gauntlet/average'][-1]) > 0
    assert isinstance(
        inmemorylogger.data['icl/metrics/eval_gauntlet/average'][-1], tuple)

    assert inmemorylogger.data['icl/metrics/eval_gauntlet/average'][-1][-1] == 0
