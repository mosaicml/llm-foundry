# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from functools import partial
from typing import List

import pytest
import torch
from composer.core import State, Time, TimeUnit
from composer.devices import DeviceCPU, DeviceGPU
from composer.models import ComposerClassifier
from composer.optim.scheduler import ComposerScheduler

from llmfoundry.optim.scheduler import InverseSquareRootWithWarmupScheduler

MAX_DURATION = '100ba'
STEPS_PER_EPOCH = 100


class SimpleModel(ComposerClassifier):
    """Small classification model.

    Args:
        num_features (int): number of input features (default: 1)
        num_classes (int): number of classes (default: 2)
    """

    def __init__(
        self,
        num_features: int = 1,
        num_classes: int = 2,
        num_hidden: int = 8,
        device: str = 'cpu',
        bias: bool = True,
    ) -> None:

        self.num_features = num_features
        self.num_classes = num_classes

        fc1 = torch.nn.Linear(num_features,
                              num_hidden,
                              device=device,
                              bias=bias)
        fc2 = torch.nn.Linear(num_hidden, num_classes, device=device, bias=bias)

        net = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            fc1,
            torch.nn.ReLU(),
            fc2,
            torch.nn.Softmax(dim=-1),
        )
        net.param_init_fn = self.param_init_fn
        super().__init__(module=net, num_classes=num_classes)

        # Important: It is crucial that the FC layers are bound to `self`
        # for the optimizer surgery tests.
        # These tests attempt to perform surgery on `fc1` layer, and we want
        # to make sure that post-surgery, self.fc1 refers to the same parameters
        # as self.net[1]
        self.fc1 = fc1
        self.fc2 = fc2

    def param_init_fn(self, module: torch.nn.Module):
        init_fn = partial(torch.nn.init.normal_, mean=0.0, std=0.1)

        if isinstance(module, torch.nn.Linear):
            init_fn(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)


@pytest.fixture
def dummy_schedulers_state(request: pytest.FixtureRequest):
    device = None
    for item in request.session.items:
        device = DeviceCPU(
        ) if item.get_closest_marker('gpu') is None else DeviceGPU()
        break
    assert device != None
    state = State(
        model=SimpleModel(),
        run_name='run_name',
        device=device,
        rank_zero_seed=17,
        max_duration=MAX_DURATION,
    )
    state.set_dataloader([None] * STEPS_PER_EPOCH, 'train')
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
        if parsed_time.unit == TimeUnit.EPOCH:
            state.timestamp = state.timestamp.copy(
                epoch=parsed_time,
                batch=Time(
                    int(state.dataloader_len) * int(parsed_time),
                    TimeUnit.BATCH),
            )
        else:
            state.timestamp = state.timestamp.copy(
                batch=parsed_time,
                epoch=Time(
                    int(parsed_time) // int(state.dataloader_len),
                    TimeUnit.EPOCH),
            )

        lr = scheduler(state, ssr)
        assert lr == pytest.approx(expected_lr, abs=1e-3)
