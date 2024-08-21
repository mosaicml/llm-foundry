# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Track training runs for loss spikes or persistently high training loss."""
from __future__ import annotations

import torch
import logging
import numpy as np
from collections import deque
from composer.core import Callback, State
from composer.loggers import Logger, MosaicMLLogger
from llmfoundry.utils.exceptions import LossSpikeError
log = logging.getLogger(__name__)

__all__ = ['KillLossSpike']

class KillLossSpike(Callback):
    """
    This callback detects and handles loss spikes or persistently high training losses during model training.

    Monitors the training loss at the end of each batch and maintains a rolling window of recent losses.
    If recent training losses exceed a specified cap or if a significant spike in loss is detected, the callback can either
    log a warning (displayed as a message on the run event) or raise a LossSpikeError to stop the run without retry.

    Args:
        log_only (bool): If True, the callback will only log warnings without interrupting training. If False, a
                         LossSpikeError will be raised to stop training upon detecting a loss spike or persistently
                         high loss.
        patience (int): The number of consecutive outlier losses tolerated before considering the training loss to be
                        persistently high. Default is 4 (so 5 consecutive outlier losses will trigger an error).
        outlier_multiplier (int): The multiplier used to determine if a loss is an outlier. A loss is considered an
                                  outlier if it is outlier_multiplier times greater than the mean of losses in
                                  the current window. Default is 2.
        window_size (int): The size of the rolling window used to track recent losses. Default is 100.
        loss_cap (int): The maximum allowable loss. If the training loss consistently exceeds this value, 
                        it is considered a diverging or unstable run. Default is 10.

    Raises:
        LossSpikeError: If log_only is False and a loss spike or persistently high loss is detected, this error is
                        raised to stop the run with an error message.
    """
	
    def __init__(self, log_only:bool, patience:int=4, outlier_multiplier:int=2, window_size:int=100, loss_cap:int=10):
        self.log_only = log_only
        self.patience = patience
        self.outlier_multiplier = outlier_multiplier
        self.window_size = window_size
        self.loss_cap = loss_cap
        self.outlier_counter = 0
        self.loss_window = deque(maxlen=self.window_size)

    def batch_end(self, state: State, logger: Logger) -> None:

        if not isinstance(state.loss, torch.Tensor):
            raise NotImplementedError('Multiple losses not supported yet')
        train_loss = state.loss.item()

        # Only start early stopping once a full window of loss data
        if len(self.loss_window) == self.window_size:
            running_loss_avg = np.mean(self.loss_window)
            log.info(f'Running loss average: {running_loss_avg}')

            # If train loss is an outlier
            if train_loss > running_loss_avg * self.outlier_multiplier:
                self.outlier_counter += 1
                log.info(f'Potential loss spike detected. Iteration: {self.outlier_counter}')
                if self.outlier_counter > self.patience:
                    log.info(f'Loss spike detected for {self.outlier_counter} steps. Try lowering the learning rate.')
                    if self.log_only:
                        for destination in logger.destinations:
                            if isinstance(destination, MosaicMLLogger):
                                destination.log_metadata({'loss_spike': f'Training loss spike detected for {self.outlier_counter} consecutive steps.'})
                    else:
                        raise LossSpikeError(self.outlier_multiplier, round(running_loss_avg), self.outlier_counter)

            # Previous step loss was an outlier, current step loss is not. Reset outlier counter.
            elif self.outlier_counter > 0:
                log.info(f'Not a persistent loss spike. Resetting outlier counter.')
                self.outlier_counter = 0

            # Half of the running losses are greater than our "high loss" threshold, after the first window
            elif (state.timestamp.batch >= self.window_size * 2) and (sum(1 for loss in self.loss_window if loss > self.loss_cap) >= self.window_size / 2):
                log.info(f'High losses >{self.loss_cap} detected.')
                if self.log_only:
                    for destination in logger.destinations:
                        if isinstance(destination, MosaicMLLogger):
                            destination.log_metadata({'high_loss': f'Persistently high (>{self.loss_cap}) training losses detected.'})
                else:
                    raise LossSpikeError()

        self.loss_window.append(train_loss)
