# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
import copy
import os
import pathlib
import shutil
import sys
from argparse import Namespace
from typing import Any, Optional

import pytest
from composer.loggers import InMemoryLogger
from composer.utils import using_torch_2
from omegaconf import DictConfig, ListConfig
from omegaconf import OmegaConf as om

# Add repo root to path so we can import scripts and test it
repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(repo_dir)

from scripts.data_prep.convert_dataset_hf import main as main_hf  # noqa: E402
from scripts.data_prep.convert_dataset_json import \
    main as main_json  # noqa: E402
from scripts.train.train import main  # noqa: E402


def create_c4_dataset_xsmall(path: pathlib.Path) -> str:
    """Creates a small mocked version of the C4 dataset."""
    c4_dir = os.path.join(path, f'my-copy-c4')
    downloaded_split = 'val_xxsmall'
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

    main_json(
        Namespace(
            **{
                'path': 'data_prep/example_data/arxiv.jsonl',
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
    conf_path: str = os.path.join(repo_dir,
                                  'scripts/train/yamls/pretrain/testing.yaml')
    with open(conf_path) as f:
        test_cfg = om.load(f)
    assert isinstance(test_cfg, DictConfig)

    test_cfg.data_local = dataset_name
    test_cfg.global_train_batch_size = 1
    test_cfg.device_eval_batch_size = 2
    test_cfg.device_train_microbatch_size = 1
    test_cfg.max_duration = '4ba'
    test_cfg.eval_interval = '4ba'
    test_cfg.run_name = 'gpt-mini-integration-test'

    test_cfg.model.n_layer = 2
    test_cfg.model.n_embd = 64

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


@pytest.mark.parametrize('averages', [{
    'core_average': ['language_understanding_lite']
}, None])
def test_train_gauntlet(averages: Optional[dict], set_correct_cwd: Any,
                        tmp_path: pathlib.Path):
    """Test training run with a small dataset."""
    dataset_name = create_c4_dataset_xsmall(tmp_path)
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
    test_cfg.icl_subset_num_batches = 1
    test_cfg.eval_subset_num_batches = 2
    test_cfg.train_loader.num_workers = 0
    test_cfg.train_loader.prefetch_factor = None if using_torch_2() else 2
    test_cfg.train_loader.persistent_workers = False
    test_cfg.eval_loader.num_workers = 0
    test_cfg.eval_loader.prefetch_factor = None if using_torch_2() else 2
    test_cfg.eval_loader.persistent_workers = False

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

    if averages is not None:
        test_cfg.eval_gauntlet['averages'] = averages

    test_cfg.icl_seq_len = 16
    test_cfg.max_duration = '1ba'
    test_cfg.eval_interval = '1ba'
    test_cfg.loggers = DictConfig({'inmemory': DictConfig({})})
    trainer = main(test_cfg)

    assert isinstance(trainer.logger.destinations, tuple)

    assert len(trainer.logger.destinations) > 0
    inmemorylogger = trainer.logger.destinations[
        0]  # pyright: ignore [reportGeneralTypeIssues]
    assert isinstance(inmemorylogger, InMemoryLogger)

    category_name = 'default_average' if averages is None else 'core_average'
    assert f'icl/metrics/eval_gauntlet/{category_name}' in inmemorylogger.data.keys(
    )
    assert isinstance(
        inmemorylogger.data[f'icl/metrics/eval_gauntlet/{category_name}'], list)
    assert len(inmemorylogger.data[f'icl/metrics/eval_gauntlet/{category_name}']
               [-1]) > 0
    assert isinstance(
        inmemorylogger.data[f'icl/metrics/eval_gauntlet/{category_name}'][-1],
        tuple)

    assert inmemorylogger.data[f'icl/metrics/eval_gauntlet/{category_name}'][
        -1][-1] == 0


def test_train_multi_eval(set_correct_cwd: Any, tmp_path: pathlib.Path):
    """Test training run with multiple eval datasets."""
    c4_dataset_name = create_c4_dataset_xsmall(tmp_path)
    test_cfg = gpt_tiny_cfg(c4_dataset_name, 'cpu')
    # Set up multiple eval dataloaders
    first_eval_loader = test_cfg.eval_loader
    first_eval_loader.label = 'c4'
    # Create second eval dataloader using the arxiv dataset.
    second_eval_loader = copy.deepcopy(first_eval_loader)
    arxiv_dataset_name = create_arxiv_dataset(tmp_path)
    second_eval_loader.data_local = arxiv_dataset_name
    second_eval_loader.label = 'arxiv'
    test_cfg.eval_loader = om.create([first_eval_loader, second_eval_loader])
    test_cfg.eval_subset_num_batches = 1  # -1 to evaluate on all batches

    test_cfg.max_duration = '1ba'
    test_cfg.eval_interval = '1ba'
    test_cfg.loggers = DictConfig({'inmemory': DictConfig({})})
    trainer = main(test_cfg)

    assert isinstance(trainer.logger.destinations, tuple)

    assert len(trainer.logger.destinations) > 0
    inmemorylogger = trainer.logger.destinations[
        0]  # pyright: ignore [reportGeneralTypeIssues]
    assert isinstance(inmemorylogger, InMemoryLogger)
    print(inmemorylogger.data.keys())

    # Checks for first eval dataloader
    assert 'metrics/eval/c4/LanguageCrossEntropy' in inmemorylogger.data.keys()
    assert isinstance(
        inmemorylogger.data['metrics/eval/c4/LanguageCrossEntropy'], list)
    assert len(
        inmemorylogger.data['metrics/eval/c4/LanguageCrossEntropy'][-1]) > 0
    assert isinstance(
        inmemorylogger.data['metrics/eval/c4/LanguageCrossEntropy'][-1], tuple)

    # Checks for second eval dataloader
    assert 'metrics/eval/arxiv/LanguageCrossEntropy' in inmemorylogger.data.keys(
    )
    assert isinstance(
        inmemorylogger.data['metrics/eval/arxiv/LanguageCrossEntropy'], list)
    assert len(
        inmemorylogger.data['metrics/eval/arxiv/LanguageCrossEntropy'][-1]) > 0
    assert isinstance(
        inmemorylogger.data['metrics/eval/arxiv/LanguageCrossEntropy'][-1],
        tuple)
