# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Monitor rate of change of loss."""
from __future__ import annotations

import torch
import numpy as np
from composer.core import Callback, State
from composer.loggers import Logger
from llmfoundry.utils.exceptions import UserError

__all__ = ['KillLossSpike']

class KillLossSpike(Callback):
	
    def __init__(self, patience:int=10, outlier_multiplier:int=2, window_size:int=100):
            self.patience = patience
            self.outlier_multiplier = outlier_multiplier
            self.window_size = window_size
            self.iterations = 0
            self.running_loss_avg = 0
            self.early_stop = False
            self.loss_window = []

    def batch_end(self, state: State, _: Logger) -> None:
        if not isinstance(state.loss, torch.Tensor):
            raise NotImplementedError('Multiple losses not supported yet')
        train_loss = state.loss.item()

        self.loss_window.append(train_loss)
        if len(self.loss_window) > self.window_size:
            self.loss_window.pop(0)
        # Only start early stopping once a full window of loss data
        if len(self.loss_window) == self.window_size:
            self.running_loss_avg = np.mean(self.loss_window)

        # If train loss exceeds the running average 
        if train_loss > self.running_loss_avg * self.outlier_multiplier:
            self.iterations += 1
            if self.iterations > self.patience:
                self.early_stop = True
                # Some kind of user error message
                raise UserError('Training stopped due to loss spike. Please try submitting the run again with a lower learning rate.')
            else:
                self.iterations = 0
