# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Monitor rate of change of loss."""
from __future__ import annotations

import torch
import logging
import numpy as np
from composer.core import Callback, State
from composer.loggers import Logger
from llmfoundry.utils.exceptions import UserError
log = logging.getLogger(__name__)

__all__ = ['KillLossSpike']

class KillLossSpike(Callback):
	
    def __init__(self, patience:int=2, outlier_multiplier:int=1.2, window_size:int=50):
        self.patience = patience
        self.outlier_multiplier = outlier_multiplier
        self.window_size = window_size
        self.outlier_counter = 0
        self.loss_window = []

    def batch_end(self, state: State, logger: Logger) -> None:
        if not isinstance(state.loss, torch.Tensor):
            raise NotImplementedError('Multiple losses not supported yet')
        train_loss = state.loss.item()
        self.loss_window.append(train_loss)
        if len(self.loss_window) > self.window_size:
            self.loss_window.pop(0)

        # Only start early stopping once a full window of loss data
        if len(self.loss_window) == self.window_size:
            running_loss_avg = np.mean(self.loss_window)
            log.info(f'Running loss average: {running_loss_avg}')

            # If train loss is an outlier
            if train_loss > running_loss_avg * self.outlier_multiplier:
                self.outlier_counter += 1
                log.info(f'Potential loss spike detected. Iteration: {self.outlier_counter}')
                if self.outlier_counter > self.patience:
                    # Some kind of user error message
                    raise UserError(f'Training stopped due to a loss spike over {self.outlier_counter} consecutive training steps. \
                                    Please try submitting the run again with a lower learning rate.')

            # Previous step loss was an outlier, current step loss is not. Reset outlier counter.
            elif self.outlier_counter > 0:
                log.info(f'Not a persistent loss spike.')
                self.outlier_counter = 0
            
            else:
                log.info('No loss spike detected.')

        else:
            log.info(f'Full loss window size not reached ({len(self.loss_window)} < {self.window_size}). Collecting loss data...')
