# Copyright 2022-2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import gc

import torch
from composer.core import Callback, State
from composer.loggers import Logger


def gc_cuda():
    """Garbage collect Torch (CUDA) memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class ScheduledGarbageCollector(Callback):
    """Disable automatic garbage collection and collect garbage at interval.

    Args:
        batch_interval (int): Number of batches between checkpoints call to gc.collect()
        eval_keep_disabled (bool): keep gc disabled during eval (default: False)
    """

    def __init__(
        self,
        batch_interval: int,
        eval_keep_disabled: bool = False,
    ):
        self.batch_interval = batch_interval
        self.eval_keep_disabled = eval_keep_disabled
        self.gc_init_state = None

    def fit_start(self, state: State, logger: Logger) -> None:
        del state, logger  # unused

        # cache if automatic garbage collection is enabled; reset at fit_end
        self.gc_init_state = gc.isenabled()

        # disable automatic garbage collection
        gc.disable()
        gc_cuda()

    def fit_end(self, state: State, logger: Logger) -> None:
        del state, logger  # unused

        gc_cuda()

        # reset automatic garbage collection at fit_end
        if self.gc_init_state:
            gc.enable()
        else:
            gc.disable()

    def before_dataloader(self, state: State, logger: Logger) -> None:
        del logger  # unused

        if state.timestamp.batch.value % self.batch_interval == 0:
            gc_cuda()

    def eval_start(self, state: State, logger: Logger) -> None:
        del state, logger  # unused

        gc_cuda()
        if not self.eval_keep_disabled:
            gc.enable()

    def eval_end(self, state: State, logger: Logger) -> None:
        del state, logger  # unused

        if not self.eval_keep_disabled:
            gc.disable()

        gc_cuda()
