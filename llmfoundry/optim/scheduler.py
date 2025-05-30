# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
"""Experimental learning rate schedulers used for training LLMs."""

import math
import textwrap
import warnings
from typing import Union

from composer.core import State, Time, TimeUnit
from composer.optim import ComposerScheduler, LinearScheduler
from composer.optim.scheduler import _convert_time

from llmfoundry.utils.warnings import experimental_class

__all__ = ['InverseSquareRootWithWarmupScheduler']


def _raise_if_units_dont_match(
    time: Union[str, Time],
    t_max: Union[str, Time],
    name: str,
) -> None:
    new_time = Time.from_input(time)
    new_t_max = Time.from_input(t_max)

    if new_time.unit != new_t_max.unit:
        raise ValueError(
            f'{name} (unit {new_time.unit=}) must match max_duration unit ({new_t_max.unit=}).',
        )


def _raise_if_units_dur(time: Union[str, Time], name: str) -> None:
    new_time = Time.from_input(time)

    if new_time.unit == TimeUnit('dur'):
        raise ValueError(f'{name} cannot be in units of "dur".')


@experimental_class('InverseSquareRootWithWarmupScheduler')
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

    def __init__(
        self,
        t_warmup: Union[str, Time],
        t_scale: Union[str, Time],
        t_cooldown: Union[str, Time],
        t_max: Union[str, Time] = '1dur',
        alpha_f_decay: float = 0.0,
        alpha_f_cooldown: float = 0.0,
    ) -> None:
        if alpha_f_decay < alpha_f_cooldown:
            raise ValueError((
                'Required: alpha_f_decay >= alpha_f_cooldown. '
                f'Current: alpha_f_decay={alpha_f_decay}, '
                f'alpha_f_cooldown={alpha_f_cooldown}.'
            ))
        _raise_if_units_dur(t_warmup, 't_warmup')
        _raise_if_units_dur(t_scale, 't_scale')
        _raise_if_units_dur(t_cooldown, 't_cooldown')
        self.t_warmup = t_warmup
        self.t_scale = t_scale
        self.t_cooldown = t_cooldown
        self.t_max = t_max
        self.alpha_f_decay = alpha_f_decay
        self.alpha_f_cooldown = alpha_f_cooldown
        self.warmup_scheduler = LinearScheduler(
            alpha_i=0.0,
            alpha_f=1.0,
            t_max=t_warmup,
        )

    def __call__(self, state: State, ssr: float = 1.0) -> float:
        assert state.max_duration is not None, 'max_duration should be set whenever schedulers are invoked'
        _raise_if_units_dont_match(
            self.t_warmup,
            state.max_duration,
            't_warmup',
        )
        _raise_if_units_dont_match(self.t_scale, state.max_duration, 't_scale')
        _raise_if_units_dont_match(
            self.t_cooldown,
            state.max_duration,
            't_cooldown',
        )

        t_warmup = _convert_time(self.t_warmup, state)
        if t_warmup.value == 0:
            warnings.warn(
                textwrap.dedent(
                    """\
                The warmup duration is 0. If warmup was specified as a fraction of the total
                training duration, the warmup duration is calculated in the
                same unit as the trainer's max_duration parameter.""",
                ),
            )

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
            current_factor = (
                self.alpha_f_decay + coeff * (1.0 - self.alpha_f_decay)
            )
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
            current_factor = (
                alpha_i + frac_of_cooldown * (self.alpha_f_cooldown - alpha_i)
            )
            return current_factor


def _cosine_anneal(x: float, min_y: float = 0.0, max_y: float = 1.0) -> float:
    """Implements a cosine decay curve.

    Curve is cos(x) on domain [0, pi], stretched to the domain [0, 1] and range
    [min_y, max_y]. Additionally, param x is clipped to the interval [0, 1]
    """
    x = min(max(x, 0.0), 1.0)
    return min_y + (max_y - min_y) * (1 + math.cos(x * math.pi)) / 2


@experimental_class('CosineAnnealingWarmRestartsWithWarmupScheduler')
class CosineAnnealingWarmRestartsWithWarmupScheduler(ComposerScheduler):
    r"""Cosine annealing warm restarts scheduler with an initial warmup.

    .. seealso::
        This scheduler combines :class:`~.CosineAnnealingWarmRestartsScheduler` with an initial warmup phase.

    During the warmup phase, the learning rate multiplier increases linearly from 0 to 1. After the warmup,
    the scheduler behaves like CosineAnnealingWarmRestartsScheduler, with cycles starting from the end of the warmup period.

    Specifically, the learning rate multiplier :math:`\alpha` can be expressed as:

    .. math::
        \alpha(t) = \begin{cases}
            t / t_{warmup}, & \text{if } t < t_{warmup} \\
            \alpha_f + (1 - \alpha_f) \times \frac{1}{2}(1 + \cos(\pi \times \tau_i)) & \text{otherwise}
        \end{cases}

    Given :math:`\tau_i`, the fraction of time elapsed through the :math:`i^\text{th}` cycle (after warmup), as:

    .. math::
        \tau_i = (t - t_{warmup} - \sum_{j=0}^{i-1} t_0 t_{mult}^j) / (t_0 t_{mult}^i)

    Where :math:`t_{warmup}` represents the warmup time, :math:`t_0` represents the period of the first cycle
    (after warmup), :math:`t_{mult}` represents the multiplier for the duration of successive cycles, and
    :math:`\alpha_f` represents the learning rate multiplier to decay to.

    Note, ``t_warmup`` and ``t_0`` cannot be specified in units of duration; since cycles may continue beyond
    the initial ``max_duration``, these parameters need to be specified in the same units as ``max_duration``
    passed to the trainer.

    .. warning::
            By default, initial warmup time is **not** scaled according to any provided scale schedule ratio.
            To change this behavior, set ``scale_warmup=True``.

    Args:
        t_warmup (str | Time): The warmup time.
        t_0 (str | Time): The period of the first cycle (after warmup).
        t_mult (float): The multiplier for the duration of successive cycles. Default = ``1.0``.
        alpha_f (float): Learning rate multiplier to decay to. Default = ``0.0``.
        scale_warmup (bool): SSR also scales the warmup period. Default = ``False``.
    """

    def __init__(
        self,
        t_warmup: Union[str, Time],
        t_0: Union[str, Time],
        t_mult: float = 1.0,
        alpha_f: float = 0.0,
        scale_warmup: bool = False,
    ) -> None:
        _raise_if_units_dur(t_warmup, 't_warmup')
        _raise_if_units_dur(t_0, 't_0')

        self.t_warmup = t_warmup
        self.t_0 = t_0
        self.t_mult = t_mult
        self.alpha_f = alpha_f
        self.scale_warmup = scale_warmup
        self.warmup_scheduler = LinearScheduler(
            alpha_i=0.0,
            alpha_f=1.0,
            t_max=t_warmup,
        )

    def __call__(self, state: State, ssr: float = 1.0) -> float:
        assert state.max_duration is not None, 'max_duration should be set whenever schedulers are invoked'

        # Validate that t_warmup and t_0 match max_duration units
        _raise_if_units_dont_match(
            self.t_warmup,
            state.max_duration,
            't_warmup',
        )
        _raise_if_units_dont_match(
            self.t_0,
            state.max_duration,
            't_0',
        )

        # Convert warmup time (potentially without SSR scaling)
        t_warmup = _convert_time(
            self.t_warmup,
            state,
            ssr=ssr if self.scale_warmup else 1.0,
        )

        # Check for zero warmup duration
        if t_warmup.value == 0:
            warnings.warn(
                textwrap.dedent(
                    """\
                The warmup duration is 0. If warmup was specified as a fraction of the total
                training duration, the warmup duration is calculated in the
                same unit as the trainer's max_duration parameter.""",
                ),
            )

        # During warmup phase
        if state.timestamp < t_warmup:
            if self.scale_warmup:
                return self.warmup_scheduler(state, ssr)
            return self.warmup_scheduler(state)

        # Convert t_0 with SSR scaling
        t_0 = _convert_time(self.t_0, state, ssr=ssr)

        # After warmup: apply warm restarts logic
        # Calculate time elapsed since warmup ended
        current_time = state.timestamp.get(t_warmup.unit)
        time_since_warmup = current_time - t_warmup

        # Find which cycle we're in and the position within that cycle
        current_interval_len = t_0
        current_interval_end = t_0
        cycle_start_time = Time(0, t_0.unit)

        while current_interval_end <= time_since_warmup:
            if current_interval_len.value == 0:
                raise ValueError(
                    'Interval between restarts for cosine annealing warm restarts scheduler has decayed to 0.',
                )

            cycle_start_time = current_interval_end
            current_interval_len = Time(
                value=int(self.t_mult * current_interval_len.value),
                unit=current_interval_len.unit,
            )
            current_interval_end = cycle_start_time + current_interval_len

        # Calculate fraction of current cycle completed
        time_in_current_cycle = time_since_warmup - cycle_start_time
        frac_of_current_interval = (
            time_in_current_cycle / current_interval_len
        ).value

        return _cosine_anneal(x=frac_of_current_interval, min_y=self.alpha_f)
