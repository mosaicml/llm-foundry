# Copyright 2022-2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from typing import List

import pytest
import torch
from composer.core import State, Time, TimeUnit
from composer.devices import DeviceCPU, DeviceGPU
from composer.optim.scheduler import ComposerScheduler

from llmfoundry.optim.scheduler import InverseSquareRootWithWarmupScheduler

_MAX_DURATION = '100ba'
_STEPS_PER_EPOCH = 100


@pytest.fixture
def dummy_schedulers_state(request: pytest.FixtureRequest):
    device = None
    for item in request.session.items:
        device = DeviceCPU(
        ) if item.get_closest_marker('gpu') is None else DeviceGPU()
        break
    assert device != None
    state = State(
        model=torch.nn.Linear(5, 5),
        run_name='run_name',
        device=device,
        rank_zero_seed=17,
        max_duration=_MAX_DURATION,
    )
    state.set_dataloader([None] * _STEPS_PER_EPOCH, 'train')
    return state


@pytest.mark.parametrize('scheduler,ssr,test_times,expected_lrs', [
    pytest.param(
        InverseSquareRootWithWarmupScheduler(t_warmup='10ba',
                                             t_scale='10ba',
                                             t_cooldown='0ba',
                                             alpha_f_decay=0,
                                             alpha_f_cooldown=0), 1.0,
        ['0ba', '5ba', '10ba', '40ba', '90ba', '100ba'],
        [0.0, 0.5, 1.0, 0.5, 0.33333, 0.31623]),
    pytest.param(
        InverseSquareRootWithWarmupScheduler(t_warmup='20ba',
                                             t_scale='2ba',
                                             t_cooldown='10ba',
                                             alpha_f_decay=0.4,
                                             alpha_f_cooldown=0.1), 1.0,
        ['0ba', '10ba', '20ba', '36ba', '90ba', '95ba', '100ba'],
        [0.0, 0.5, 1.0, 0.6, 0.5, 0.3, 0.1]),
])
def test_scheduler_init(scheduler: ComposerScheduler, ssr: float,
                        test_times: List[str], expected_lrs: List[float],
                        dummy_schedulers_state: State):

    state = dummy_schedulers_state
    assert state.dataloader_len is not None
    assert state.max_duration is not None
    state.max_duration = Time(value=int(state.max_duration.value * ssr),
                              unit=state.max_duration.unit)
    for test_time, expected_lr in zip(test_times, expected_lrs):
        parsed_time = Time.from_timestring(test_time)
        assert parsed_time.unit in [TimeUnit.EPOCH, TimeUnit.BATCH]
        state.timestamp = state.timestamp.copy(
            batch=parsed_time,
            epoch=Time(
                int(parsed_time) // int(state.dataloader_len), TimeUnit.EPOCH),
        )
        lr = scheduler(state, ssr)
        assert lr == pytest.approx(expected_lr, abs=1e-3)


@pytest.mark.parametrize('state_unit,warmup_unit,scale_unit,cooldown_unit', [
    ['ep', 'ba', 'ba', 'ba'],
    ['ba', 'ep', 'ep', 'ep'],
    ['ep', 'ep', 'ba', 'ep'],
])
def test_scheduler_units_match_error(state_unit: str, warmup_unit: str,
                                     scale_unit: str, cooldown_unit: str,
                                     dummy_schedulers_state: State):

    state = dummy_schedulers_state
    state.max_duration = f'1{state_unit}'
    scheduler = InverseSquareRootWithWarmupScheduler(
        t_warmup=f'10{warmup_unit}',
        t_scale=f'10{scale_unit}',
        t_cooldown=f'10{cooldown_unit}')
    with pytest.raises(ValueError, match='does not match'):
        _ = scheduler(state, 1.0)


@pytest.mark.parametrize('warmup_unit,scale_unit,cooldown_unit', [
    ['dur', 'ba', 'ba'],
    ['ba', 'dur', 'ba'],
    ['ba', 'ba', 'dur'],
])
def test_unit_dur_error(warmup_unit: str, scale_unit: str, cooldown_unit: str):
    with pytest.raises(ValueError, match='cannot be in units of "dur".'):
        _ = InverseSquareRootWithWarmupScheduler(t_warmup=f'1{warmup_unit}',
                                                 t_scale=f'1{scale_unit}',
                                                 t_cooldown=f'1{cooldown_unit}')


def test_alpha_f_error():
    with pytest.raises(ValueError, match='alpha_f_decay >= alpha_f_cooldown.'):
        _ = InverseSquareRootWithWarmupScheduler(t_warmup='10ba',
                                                 t_scale='10ba',
                                                 t_cooldown='10ba',
                                                 alpha_f_decay=0.0,
                                                 alpha_f_cooldown=0.1)
