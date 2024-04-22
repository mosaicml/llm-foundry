# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import copy
import os
import pathlib
from typing import Optional

import pytest
from composer.loggers import InMemoryLogger
from omegaconf import DictConfig, ListConfig
from omegaconf import OmegaConf as om

from llmfoundry.utils.config_utils import (make_dataclass_and_log_config,
                                           to_dict_recursive,
                                           update_batch_size_info)
from scripts.train.train import TrainConfig  # noqa: E402
from scripts.train.train import TRAIN_CONFIG_KEYS, main, validate_config
from tests.data_utils import create_c4_dataset_xxsmall, gpt_tiny_cfg
from tests.fixtures.autouse import REPO_DIR


@pytest.mark.parametrize('averages', [{
    'core_average': ['language_understanding_lite']
}, None])
def test_train_gauntlet(averages: Optional[dict], tmp_path: pathlib.Path):
    """Test training run with a small dataset."""
    dataset_name = create_c4_dataset_xxsmall(tmp_path)
    test_cfg = gpt_tiny_cfg(dataset_name, 'cpu')
    test_cfg.icl_tasks = ListConfig([
        DictConfig({
            'label':
                'lambada_openai',
            'dataset_uri':
                'scripts/eval/local_data/language_understanding/lambada_openai_small.jsonl',
            'num_fewshot': [0],
            'icl_task_type':
                'language_modeling'
        })
    ])
    test_cfg.icl_subset_num_batches = 1
    test_cfg.eval_subset_num_batches = 2
    test_cfg.train_loader.num_workers = 0
    test_cfg.train_loader.prefetch_factor = None
    test_cfg.train_loader.persistent_workers = False
    test_cfg.eval_loader.num_workers = 0
    test_cfg.eval_loader.prefetch_factor = None
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


def test_train_multi_eval(tmp_path: pathlib.Path):
    """Test training run with multiple eval datasets."""
    c4_dataset_name = create_c4_dataset_xxsmall(tmp_path)
    test_cfg = gpt_tiny_cfg(c4_dataset_name, 'cpu')
    # Set up multiple eval dataloaders
    first_eval_loader = test_cfg.eval_loader
    first_eval_loader.label = 'c4'
    # Create second eval dataloader using the arxiv dataset.
    second_eval_loader = copy.deepcopy(first_eval_loader)
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


@pytest.mark.gpu
def test_validate_config():
    conf_path: str = os.path.join(
        REPO_DIR,
        'scripts/train/yamls/pretrain/testing-moe.yaml',
    )
    with open(conf_path) as f:
        test_cfg: DictConfig = om.load(f)  # type: ignore
    test_cfg.model.ffn_config.moe_world_size = 4
    test_cfg.fsdp_config.use_orig_params = False
    test_cfg_dict = to_dict_recursive(test_cfg)
    test_cfg_dict = update_batch_size_info(test_cfg_dict)
    with pytest.raises(
            ValueError,
            match=
            'MoEs with expert parallelism (.*) require `use_orig_params=True`.'
    ):
        _, cfg_obj = make_dataclass_and_log_config(test_cfg_dict, TrainConfig,
                                                   TRAIN_CONFIG_KEYS)
        validate_config(cfg_obj)


def test_eval_metrics_with_no_train_metrics(tmp_path: pathlib.Path):
    """Test using use_train_metrics=False does not disable eval metrics."""
    c4_dataset_name = create_c4_dataset_xxsmall(tmp_path)
    test_cfg = gpt_tiny_cfg(c4_dataset_name, 'cpu')
    first_eval_loader = test_cfg.eval_loader
    first_eval_loader.label = 'c4'
    test_cfg.eval_loader = om.create([first_eval_loader])
    test_cfg.eval_subset_num_batches = 1  # -1 to evaluate on all batches
    test_cfg.max_duration = '1ba'
    test_cfg.eval_interval = '1ba'
    test_cfg.loggers = DictConfig({'inmemory': DictConfig({})})
    test_cfg.model['use_train_metrics'] = False
    trainer = main(test_cfg)

    # Check eval metrics exist
    inmemorylogger = trainer.logger.destinations[
        0]  # pyright: ignore [reportGeneralTypeIssues]
    assert isinstance(inmemorylogger, InMemoryLogger)

    assert 'metrics/eval/c4/LanguageCrossEntropy' in inmemorylogger.data.keys()
    assert isinstance(
        inmemorylogger.data['metrics/eval/c4/LanguageCrossEntropy'], list)
    assert len(
        inmemorylogger.data['metrics/eval/c4/LanguageCrossEntropy'][-1]) > 0
    assert isinstance(
        inmemorylogger.data['metrics/eval/c4/LanguageCrossEntropy'][-1], tuple)
