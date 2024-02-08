# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import gc
from typing import Optional

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
        batch_interval (int): Number of batches between calls to gc.collect()
        gen_1_batch_interval(int, optional): Number of batches between calls to gc.collect(1)
        eval_keep_disabled (bool): keep gc disabled during eval (default: False)
    """

    def __init__(
        self,
        batch_interval: int,
        gen_1_batch_interval: Optional[int] = None,
        eval_keep_disabled: bool = False,
    ):
        self.batch_interval = batch_interval
        self.gen_1_batch_interval = gen_1_batch_interval
        self.eval_keep_disabled = eval_keep_disabled
        self.gc_init_state = None

    def fit_start(self, state: State, logger: Logger) -> None:
        #del state, logger  # unused
        del logger

        # cache if automatic garbage collection is enabled; reset at fit_end
        self.gc_init_state = gc.isenabled()

        # disable automatic garbage collection
        gc.disable()
        gc_cuda()

        print("\n")
        for n, p in state.model.named_parameters():
            print("name: ", n)
            print("param mean: ", p.mean().item())
            print("param std: ", p.std().item())

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

        if self.gen_1_batch_interval is not None and state.timestamp.batch.value % self.gen_1_batch_interval == 0:
            gc.collect(1)

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
