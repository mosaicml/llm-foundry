# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Track training runs for loss spikes or persistently high training loss."""
from __future__ import annotations

import torch
import logging
import numpy as np
from composer.core import Callback, State
from composer.loggers import Logger, MosaicMLLogger
# from llmfoundry.utils.exceptions import LossSpikeError
log = logging.getLogger(__name__)

__all__ = ['KillLossSpike']

class KillLossSpike(Callback):
	
    def __init__(self, patience:int=4, outlier_multiplier:int=2, window_size:int=100, loss_cap:int=10):
        self.patience = patience
        self.outlier_multiplier = outlier_multiplier
        self.window_size = window_size
        self.loss_cap = loss_cap
        self.outlier_counter = 0
        self.loss_window = []

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
                    # NOTE: Adding this info the TRAIN_UPDATED event is temporary to 1) collect data on spiky runs and 2) give users information about their run. 
                    # This will be replaced with the hard error LossSpikeError.
                    for destination in logger.destinations:
                        if isinstance(destination, MosaicMLLogger):
                            destination.log_metadata({'loss_spike': f'Training loss spike detected for {self.outlier_counter} consecutive steps.'})
                    # raise LossSpikeError(self.outlier_multiplier, round(running_loss_avg), self.outlier_counter)

            # Previous step loss was an outlier, current step loss is not. Reset outlier counter.
            elif self.outlier_counter > 0:
                log.info(f'Not a persistent loss spike. Resetting outlier counter.')
                self.outlier_counter = 0

            # Half of the running losses are greater than our "high loss" threshold, after the first window
            elif (state.timestamp.batch >= self.window_size * 2) and (sum(1 for loss in self.loss_window if loss > self.loss_cap) >= self.window_size / 2):
                log.info(f'High losses >{self.loss_cap} detected.')
                for destination in logger.destinations:
                    if isinstance(destination, MosaicMLLogger):
                        destination.log_metadata({'high_loss': f'Persistently high (>{self.loss_cap}) training losses detected.'})
                # raise LossSpikeError()

        else:
            log.info(f'Full loss window size not reached ({len(self.loss_window)} < {self.window_size}). Collecting loss data...')

        self.loss_window.append(train_loss)
        if len(self.loss_window) > self.window_size:
            self.loss_window.pop(0)
