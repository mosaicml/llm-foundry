# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from typing import Any, Callable, Optional
from unittest.mock import MagicMock

import pytest
from composer.core import State
from composer.core.time import Time, TimeUnit
from composer.devices import DeviceCPU
from composer.loggers import Logger
from omegaconf import OmegaConf as om
from torch.utils.data import DataLoader

from llmfoundry.data.text_data import StreamingTextDataset
from llmfoundry.utils.builders import build_callback


@pytest.mark.parametrize(
    'datamix,duration',
    [
        (None, '1ep'),
        ({
            'dataset': 'some_dataset',
        }, '1ep'),
        (None, '10tok'),
        (None, ''),
        ({}, '1ep'),
    ],
)
def test_curriculum_learning_callback_init(
    datamix: Optional[dict[str, Any]],
    duration: str,
    tiny_ft_dataloader_cfg: dict[str, Any],
):
    test_cfg = _get_test_cfg()
    test_cfg['train_loader'] = tiny_ft_dataloader_cfg
    train_loader = test_cfg['train_loader'] if datamix is None else datamix
    kwargs = {
        'schedule': [{
            'duration': duration,
            'train_loader': train_loader,
        }, {
            'duration': '2ep',
            'train_loader': {},
        }],
    }
    if duration == '':
        del kwargs['schedule'][0]['duration']
    if datamix is not None and len(datamix) == 0:
        del kwargs['schedule'][0]['train_loader']

    context = nullcontext()
    if datamix is not None or duration == '':
        context = pytest.raises(ValueError)
    with context:
        callback = build_callback(
            'curriculum_learning',
            kwargs=kwargs,
            train_config=test_cfg,
        )
        assert callback is not None


@pytest.mark.parametrize('duration', ['1ep', '10tok', '2ep'])
def test_curriculum_learning_callback_before_load(
    duration: str,
    build_tiny_mpt: Callable,
):
    model = build_tiny_mpt(loss_fn='torch_crossentropy')
    state = State(
        model=model,
        rank_zero_seed=0,
        run_name='test_state',
        device=DeviceCPU(),
    )
    state.max_duration = '3ep'
    dl_mock = MagicMock(spec=DataLoader)
    dl_mock.dataset = MagicMock(spec=StreamingTextDataset)
    state.train_dataloader = dl_mock
    logger = Logger(state)

    test_cfg = _get_test_cfg()
    kwargs = {
        'schedule': [{
            'duration': duration,
            'train_loader': test_cfg['train_loader'],
        }, {
            'duration': '2ep',
            'train_loader': test_cfg['train_loader'],
        }],
    }

    callback = build_callback(
        'curriculum_learning',
        kwargs=kwargs,
        train_config=test_cfg,
    )
    context = nullcontext()
    if duration != '1ep':
        context = pytest.raises(ValueError)
    with context:
        callback.before_load(state, logger)


def test_curriculum_learning_callback_after_load(build_tiny_mpt: Callable,):
    model = build_tiny_mpt(loss_fn='torch_crossentropy')
    state = State(
        model=model,
        rank_zero_seed=0,
        run_name='test_state',
        device=DeviceCPU(),
    )
    state.max_duration = '3ep'
    dl_mock = MagicMock(spec=DataLoader)
    dl_mock.dataset = MagicMock(spec=StreamingTextDataset)
    state.train_dataloader = dl_mock
    state.timestamp.epoch_in_iteration = 2
    logger = Logger(state)

    test_cfg = _get_test_cfg()
    kwargs = {
        'schedule': [{
            'duration': '1ep',
            'train_loader': test_cfg['train_loader'],
        }, {
            'duration': '2ep',
            'train_loader': test_cfg['train_loader'],
        }],
    }

    callback = build_callback(
        'curriculum_learning',
        kwargs=kwargs,
        train_config=test_cfg,
    )
    assert state.timestamp.iteration == 0
    callback.after_load(state, logger)
    assert state.timestamp.iteration == 1


def test_curriculum_learning_callback_iteration(
    build_tiny_mpt: Callable,
    monkeypatch: pytest.MonkeyPatch,
):
    model = build_tiny_mpt(loss_fn='torch_crossentropy')
    state = State(
        model=model,
        rank_zero_seed=0,
        run_name='test_state',
        device=DeviceCPU(),
    )
    state.max_duration = '3ep'
    dl_mock = MagicMock(spec=DataLoader)
    ds_mock = MagicMock(spec=StreamingTextDataset)
    monkeypatch.setattr(
        'llmfoundry.data.text_data.StreamingTextDataset',
        lambda *args,
        **kwargs: ds_mock,
    )
    dl_mock.dataset = ds_mock
    state.train_dataloader = dl_mock
    state.timestamp.epoch_in_iteration = 2
    logger = Logger(state)

    test_cfg = _get_test_cfg()
    kwargs = {
        'schedule': [{
            'duration': '1ep',
            'train_loader': test_cfg['train_loader'],
        }, {
            'duration': '2ep',
            'train_loader': test_cfg['train_loader'],
        }],
    }

    callback = build_callback(
        'curriculum_learning',
        kwargs=kwargs,
        train_config=test_cfg,
    )

    callback.init(state, logger)
    callback.iteration_start(state, logger)
    assert state._iteration_length == Time(1, TimeUnit.EPOCH)
    callback.iteration_end(state, logger)
    callback.iteration_start(state, logger)
    assert state._iteration_length == Time(2, TimeUnit.EPOCH)


def test_curriculum_learning_callback_state_dict(build_tiny_mpt: Callable,):
    model = build_tiny_mpt(loss_fn='torch_crossentropy')
    state = State(
        model=model,
        rank_zero_seed=0,
        run_name='test_state',
        device=DeviceCPU(),
    )
    state.max_duration = '3ep'
    dl_mock = MagicMock(spec=DataLoader)
    dl_mock.dataset = MagicMock(spec=StreamingTextDataset)
    state.train_dataloader = dl_mock
    state.timestamp.epoch_in_iteration = 2
    logger = Logger(state)

    test_cfg = _get_test_cfg()
    kwargs = {
        'schedule': [{
            'duration': '1ep',
            'train_loader': test_cfg['train_loader'],
        }, {
            'duration': '2ep',
            'train_loader': test_cfg['train_loader'],
        }],
    }

    callback = build_callback(
        'curriculum_learning',
        kwargs=kwargs,
        train_config=test_cfg,
    )
    callback.iteration_start(state, logger)
    callback.iteration_end(state, logger)
    assert callback.state_dict() == {
        'schedule': kwargs['schedule'],
        'schedule_index': 1,
    }


def test_curriculum_learning_callback_load_state_dict(
    build_tiny_mpt: Callable,
):
    model = build_tiny_mpt(loss_fn='torch_crossentropy')
    state = State(
        model=model,
        rank_zero_seed=0,
        run_name='test_state',
        device=DeviceCPU(),
    )
    state.max_duration = '3ep'
    dl_mock = MagicMock(spec=DataLoader)
    dl_mock.dataset = MagicMock(spec=StreamingTextDataset)
    state.train_dataloader = dl_mock
    state.timestamp.epoch_in_iteration = 2
    logger = Logger(state)

    test_cfg = _get_test_cfg()
    kwargs = {
        'schedule': [{
            'duration': '1ep',
            'train_loader': test_cfg['train_loader'],
        }, {
            'duration': '2ep',
            'train_loader': test_cfg['train_loader'],
        }],
    }

    callback = build_callback(
        'curriculum_learning',
        kwargs=kwargs,
        train_config=test_cfg,
    )
    callback.iteration_start(state, logger)
    callback.iteration_end(state, logger)
    assert callback.state_dict() == {
        'schedule': kwargs['schedule'],
        'schedule_index': 1,
    }


def _get_test_cfg() -> dict[str, Any]:
    conf_path = 'scripts/train/yamls/pretrain/testing.yaml'
    with open(conf_path) as f:
        test_cfg = om.load(f)
    batch_size = test_cfg['device_train_microbatch_size'
                         ]  # pyright: ignore [reportGeneralTypeIssues]
    test_cfg['device_train_batch_size'
            ] = batch_size  # pyright: ignore [reportGeneralTypeIssues]
    return om.to_container(
        test_cfg,
        resolve=True,
    )  # pyright: ignore [reportGeneralTypeIssues]
