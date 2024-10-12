# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from click import Option
from composer import Callback, Logger, State, Time
from git import Optional

from llmfoundry.utils.warnings import experimental_class

__all__ = [
    'SlidingWindowSizeWarmerUpper',
]


@experimental_class('SlidingWindowSizeWarmerUpper')
class SlidingWindowSizeWarmerUpper(Callback):
    """Warms up the sliding window size for the model based on a schedule.

    Args:
        t_warmup (str|None): Warmup duration for sliding window size, defaults to scheduler.t_warmup.
    """

    def __init__(
        self,
        t_warmup: Optional[str | Time] = None,
    ):
        if isinstance(t_warmup, str):
            t_warmup = Time.from_timestring(t_warmup)
        self.t_warmup = t_warmup

    def before_train_batch(self, state: State, logger: Logger):
        del logger
        t_warmup = self.t_warmup or state.model.scheduler.t_warmup
