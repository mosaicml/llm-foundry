# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Experimental learning rate schedulers used for training LLMs."""

import textwrap
import warnings
from typing import Union

from composer.core import State, Time, TimeUnit
from composer.optim import ComposerScheduler, LinearScheduler
from composer.optim.scheduler import _convert_time

__all__ = ['InverseSquareRootWithWarmupScheduler']


def _raise_if_units_dont_match(time: Union[str, Time], t_max: Union[str, Time],
                               name: str) -> None:
    if isinstance(time, str):
        time = Time.from_timestring(time)
    if isinstance(t_max, str):
        t_max = Time.from_timestring(t_max)

    assert not isinstance(time, str) and not isinstance(t_max, str)

    if time.unit != t_max.unit:
        raise ValueError(f'{time.unit=} does not match {t_max.unit=}.')


def _raise_if_units_dur(time: Union[str, Time], name: str) -> None:
    if isinstance(time, str):
        time = Time.from_timestring(time)

    assert not isinstance(time, str)

    if time.unit == TimeUnit('dur'):
        raise ValueError(f'{name} cannot be in units of "dur".')


class InverseSquareRootWithWarmupScheduler(ComposerScheduler):
    r"""Inverse square root LR decay with warmup and optional linear cooldown.

    Specifically, the learning rate multiplier :math:`\alpha(t)` can be expressed as:

    .. math::
        \alpha(t) = \begin{cases}
            t / t_{warmup}, & \text{if } t < t_{warmup} \\
            \alpha_{f,decay} + \frac{1 - \alpha_{f,decay}}{\sqrt{\tau_d}}, & \text{if } t_{warmup} <= t < t_{max} - t_{cooldown} \\
            \alpha_i + (alpha_{f,cooldown} - \alpha_i) \times \tau_c, & \text{otherwise}
        \end{cases}

    Given :math:`\tau_d`, the time elapsed during the inverse square root decay (normalized by :math:`t_scale`), as:

    .. math::
        \tau_d = (t - t_{warmup} + t_{scale}) / {t_scale}

    :math:`\alpha_i` as the value of the learning rate multiplier when :math:`\tau_d` is evaluated at :math:`t = t_{max} - t_{cooldown}`,
    and :math:`\tau_c`, the fraction of linear cooldown time elapsed (clipped to the interval :math:`[0, 1]`), as:

    .. math::
        \tau_c = (t - t_{max} + t_{cooldown}) / t_{cooldown}

    Where :math:`t_{warmup}` represents the warmup time, :math:`t_{scale}` represents the time scale,
    :math:`t_{cooldown}` represents the cooldown time, :math:`t_{max}` represents the duration of this scheduler,
    :math:`\alpha_{f,decay}` represents the learning rate multiplier that the inverse square root decays to at infinite time,
    and :math:`\alpha_{f,cooldown}` represents the learning rate multiplier that the linear cooldown decays to.

    Note, :math:`\alpha_{f,decay} >= \alpha_{f,cooldown}` to ensure that the learning rate is monotonically decreasing after warmup.

    Also note, ``t_warmup``, ``t_scale``, and ``t_cooldown`` cannot be specified in units of duration; since this schedule is designed for continual learning,
    ``max_duration`` is expected to change. Instead, these parameters need to be specified in the same units as ``max_duration`` passed to the trainer.

    Args:
        t_warmup (str | Time): The warmup time.
        t_scale (str | Time): The time scale.
        t_cooldown (str | Time): The cooldown time.
        t_max (str | Time): The duration of this scheduler. Default = ``"1dur"``.
        alpha_f_decay (float): The learning rate multiplier to decay inverse square root decay to. Default = ``0.0``.
        alpha_f_cooldown (float): The learning rate multiplier to decay linear cooldown to. Default = ``0.0``.
    """

    def __init__(self,
                 t_warmup: Union[str, Time],
                 t_scale: Union[str, Time],
                 t_cooldown: Union[str, Time],
                 t_max: Union[str, Time] = '1dur',
                 alpha_f_decay: float = 0.0,
                 alpha_f_cooldown: float = 0.0) -> None:
        if alpha_f_decay < alpha_f_cooldown:
            raise ValueError(('Required: alpha_f_decay >= alpha_f_cooldown. '
                              f'Current: alpha_f_decay={alpha_f_decay}, '
                              f'alpha_f_cooldown={alpha_f_cooldown}.'))
        _raise_if_units_dur(t_warmup, 't_warmup')
        _raise_if_units_dur(t_scale, 't_scale')
        _raise_if_units_dur(t_cooldown, 't_cooldown')
        self.t_warmup = t_warmup
        self.t_scale = t_scale
        self.t_cooldown = t_cooldown
        self.t_max = t_max
        self.alpha_f_decay = alpha_f_decay
        self.alpha_f_cooldown = alpha_f_cooldown
        self.warmup_scheduler = LinearScheduler(alpha_i=0.0,
                                                alpha_f=1.0,
                                                t_max=t_warmup)

    def __call__(self, state: State, ssr: float = 1.0) -> float:
        assert state.max_duration is not None, 'max_duration should be set whenever schedulers are invoked'
        _raise_if_units_dont_match(self.t_warmup, state.max_duration,
                                   't_warmup')
        _raise_if_units_dont_match(self.t_scale, state.max_duration, 't_scale')
        _raise_if_units_dont_match(self.t_cooldown, state.max_duration,
                                   't_cooldown')

        t_warmup = _convert_time(self.t_warmup, state)
        if t_warmup.value == 0:
            warnings.warn(
                textwrap.dedent("""\
                The warmup duration is 0. If warmup was specified as a fraction of the total
                training duration, the warmup duration is calculated in the
                same unit as the trainer's max_duration parameter."""))

        if state.timestamp < t_warmup:
            return self.warmup_scheduler(state)

        t_scale = _convert_time(self.t_scale, state, ssr=ssr)
        t_cooldown = _convert_time(self.t_cooldown, state, ssr=ssr)
        t_max = _convert_time(self.t_max, state, ssr=ssr)
        current_time = state.timestamp.get(t_scale.unit)

        t_shift = t_scale - t_warmup
        # t_cooldown_start is max of t_warmup, t_max - t_cooldown
        t_cooldown_start = t_max - t_cooldown
        if t_cooldown_start < t_warmup:
            t_cooldown_start = t_warmup

        if state.timestamp < t_cooldown_start:
            # Rescale LR by a coefficient equal to the inverse square root of the time
            # elapsed after warmup, rescaled by the time scale, such that, at
            # infinite time, the LR decays to alpha_f_decay.
            coeff = 1 / ((current_time + t_shift) / t_scale).value**0.5
            current_factor = (self.alpha_f_decay + coeff *
                              (1.0 - self.alpha_f_decay))
            return current_factor

        else:
            coeff = 1 / ((t_cooldown_start + t_shift) / t_scale).value**0.5
            alpha_i = self.alpha_f_decay + coeff * (1.0 - self.alpha_f_decay)

            if t_cooldown.value == 0:
                return alpha_i

            # Linearly decay the LR from its value at the step at which cooldown
            # started to alpha_f_cooldown over t_cooldown time.
            frac_of_cooldown = ((current_time - t_cooldown_start) /
                                t_cooldown).value
            frac_of_cooldown = min(1.0, frac_of_cooldown)
            current_factor = (alpha_i + frac_of_cooldown *
                              (self.alpha_f_cooldown - alpha_i))
            return current_factor
