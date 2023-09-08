# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import textwrap
import warnings
from typing import Union

from composer.core import State, Time, TimeUnit
from composer.optim import ComposerScheduler, LinearScheduler
from composer.optim.scheduler import _convert_time


def _raise_if_unit_not_ba(time: Union[str, Time]) -> None:
    if isinstance(time, str):
        time = Time.from_timestring(time)
    if time.unit != TimeUnit('ba'):
        raise ValueError


class InverseSquareRootWithWarmupScheduler(ComposerScheduler):

    def __init__(self,
                 t_warmup: Union[str, Time],
                 t_scale: Union[str, Time],
                 t_cooldown: Union[str, Time],
                 t_max: Union[str, Time] = '1dur',
                 alpha_f_decay: float = 0.0,
                 alpha_f_cooldown: float = 0.0,
                 scale_warmup: bool = False):
        _raise_if_unit_not_ba(t_warmup)
        _raise_if_unit_not_ba(t_scale)
        _raise_if_unit_not_ba(t_cooldown)
        if alpha_f_cooldown > alpha_f_decay:
            raise ValueError
        self.t_warmup = t_warmup
        self.t_scale = t_scale
        self.t_cooldown = t_cooldown
        self.t_max = t_max
        self.alpha_f_decay = alpha_f_decay
        self.alpha_f_cooldown = alpha_f_cooldown
        self.scale_warmup = scale_warmup
        self.warmup_scheduler = LinearScheduler(alpha_i=0.0,
                                                alpha_f=1.0,
                                                t_max=t_warmup)

    def __call__(self, state: State, ssr: float = 1.0) -> float:
        assert state.max_duration is not None, 'max_duration should be set whenever schedulers are invoked'
        _raise_if_unit_not_ba(state.max_duration)

        t_warmup = _convert_time(self.t_warmup, state)
        if t_warmup.value == 0:
            warnings.warn(
                textwrap.dedent("""\
                The warmup duration is 0. If you specified warmup as a fraction of total
                training duration, take note that the warmup duration is calculated in the
                same unit as the trainer's max_duration parameter."""))

        if state.timestamp < t_warmup:
            if self.scale_warmup:
                return self.warmup_scheduler(state, ssr)
            return self.warmup_scheduler(state)

        t_scale = _convert_time(self.t_scale, state, ssr=ssr)
        t_cooldown = _convert_time(self.t_cooldown, state, ssr=ssr)
        t_max = _convert_time(self.t_max, state, ssr=ssr)
        current_time = state.timestamp.get(t_scale.unit)

        t_shift = t_scale - t_warmup
        t_cooldown_start = t_max - t_cooldown
        if t_cooldown_start < t_warmup:
            t_cooldown_start = t_warmup

        if state.timestamp < t_cooldown_start:
            coeff = 1 / ((current_time + t_shift) / t_scale).value**0.5
            current_factor = (self.alpha_f_decay + coeff *
                              (1.0 - self.alpha_f_decay))
            return current_factor

        else:
            coeff = 1 / ((t_cooldown_start + t_shift) / t_scale).value**0.5
            alpha_i = self.alpha_f_decay + coeff * (1.0 - self.alpha_f_decay)

            if t_cooldown.value == 0:
                return alpha_i

            frac_of_cooldown = ((current_time - t_cooldown_start) /
                                t_cooldown).value
            frac_of_cooldown = min(1.0, frac_of_cooldown)
            current_factor = (alpha_i + frac_of_cooldown *
                              (self.alpha_f_cooldown - alpha_i))
            return current_factor
