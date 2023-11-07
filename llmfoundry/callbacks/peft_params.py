# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import gc

import torch
from composer.core import Callback, State
from composer.loggers import Logger

from llmfoundry.models.hf.hf_causal_lm import print_trainable_parameters


class LogPeftParams(Callback):
    """Disable automatic garbage collection and collect garbage at interval.

    Args:
        batch_interval (int): Number of batches between checkpoints call to gc.collect()
        eval_keep_disabled (bool): keep gc disabled during eval (default: False)
    """

    def fit_start(self, state: State, logger: Logger) -> None:
        params = print_trainable_parameters(state.model)
        logger.log_metrics(params)
