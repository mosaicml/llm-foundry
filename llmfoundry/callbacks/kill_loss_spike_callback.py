# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Track training runs for loss spikes or persistently high training loss."""
from __future__ import annotations

import logging
from collections import deque

import numpy as np
import torch
from composer.core import Callback, State, TimeUnit
from composer.loggers import Logger, MosaicMLLogger
from composer.utils import dist

from llmfoundry.utils.exceptions import HighLossError, LossSpikeError
from llmfoundry.utils.warnings import experimental_class

log = logging.getLogger(__name__)

__all__ = ['KillLossSpike']

_MIN_WINDOW_SIZE = 100
_MAX_LOSS_CAP = 10
_WINDOW_FRACTION = 0.05


@experimental_class('KillLossSpike')
class KillLossSpike(Callback):
    """Detects and handles loss spikes or high losses during training.

    Monitors the training loss at the end of each batch and maintains a rolling window of recent losses.
    If recent training losses exceed a specified cap or if a significant spike in loss is detected, the callback can either
    log a warning (displayed as a message on the run event) or raise a LossSpikeError to stop the run without retry.

    Args:
        log_only (bool): If True, the callback will only log warnings without interrupting training. If False, a
                         LossSpikeError will be raised to stop training upon detecting a loss spike or persistently
                         high loss. Default is True.
        patience (int): The number of consecutive outlier losses tolerated before considering the training loss to be
                        persistently high. Default is 4 (so 5 consecutive outlier losses will trigger an error).
        outlier_multiplier (int): The multiplier used to determine if a loss is an outlier. A loss is considered an
                                  outlier if it is outlier_multiplier times greater than the mean of losses in
                                  the current window. Default is 2.
        window_size (int): The size of the rolling window used to track recent losses. This is set to 1/20 of the total training batches, with a minimum of 100 steps.
        loss_cap (int): The maximum allowable loss. If the training loss consistently exceeds this value,
                        it is considered a diverging or unstable run. This is set to the maximum loss from the first window of losses, with a maximum of 10.

    Raises:
        LossSpikeError: If log_only is False and a loss spike or persistently high loss is detected, this error is
                        raised to stop the run with an error message.
    """

    def __init__(
        self,
        log_only: bool = True,
        patience: int = 4,
        outlier_multiplier: float = 2,
        window_size: int = _MIN_WINDOW_SIZE,
        loss_cap: float = _MAX_LOSS_CAP,
    ):
        self._enabled = (dist.get_global_rank() == 0)
        self.log_only = log_only
        self.patience = patience
        self.outlier_multiplier = outlier_multiplier
        self.outlier_counter = 0
        self.window_size = window_size
        self.loss_window = deque()
        self.loss_cap = loss_cap
        self.window_size_set = (window_size != _MIN_WINDOW_SIZE)
        self.loss_cap_set = (loss_cap != _MAX_LOSS_CAP)

    def _detect_loss_spike(
        self,
        train_loss: float,
        running_loss_avg: float,
    ) -> bool:
        # Train loss is an outlier
        if train_loss >= running_loss_avg * self.outlier_multiplier:
            self.outlier_counter += 1
            log.info(
                f'Potential loss spike detected. Iteration: {self.outlier_counter}',
            )
            if self.outlier_counter > self.patience:
                log.info(
                    f'Loss spike detected for {self.outlier_counter} steps. Try lowering the learning rate.',
                )
                return True
        # Previous step loss was an outlier, current step loss is not. Reset outlier counter.
        elif self.outlier_counter > 0:
            log.info(f'Not a persistent loss spike. Resetting outlier counter.')
            self.outlier_counter = 0
        return False

    def _detect_high_losses(self, current_step: int) -> bool:
        if current_step < self.window_size * 2:
            return False

        # Half of the running losses are greater than our "high loss" threshold, after an initial buffer period
        high_loss_count = sum(
            1 for loss in self.loss_window if loss > self.loss_cap
        )
        is_high_loss = (high_loss_count >= self.window_size / 2)

        if is_high_loss:
            log.info(
                f'High losses detected: {high_loss_count}/{self.window_size} losses above {self.loss_cap}.',
            )

        return is_high_loss

    def _log_metadata(self, logger: Logger, key: str, value: dict) -> None:
        for destination in logger.destinations:
            if isinstance(destination, MosaicMLLogger):
                destination.log_metadata({
                    key: value,
                    'loss_window': list(self.loss_window),
                })

    def _handle_loss_spike(
        self,
        logger: Logger,
        running_loss_avg: float,
    ) -> None:
        if self.log_only:
            self._log_metadata(
                logger,
                'loss_spike',
                {
                    'outlier_multiplier': self.outlier_multiplier,
                    'running_loss_avg': running_loss_avg,
                    'outlier_counter': self.outlier_counter,
                },
            )
        else:
            raise LossSpikeError(
                outlier_multiplier=self.outlier_multiplier,
                running_loss_avg=running_loss_avg,
                outlier_counter=self.outlier_counter,
                loss_window=list(self.loss_window),
            )

    def _handle_high_losses(self, logger: Logger) -> None:
        if self.log_only:
            self._log_metadata(
                logger,
                'high_loss',
                {
                    'loss_cap': self.loss_cap,
                    'window_size': self.window_size,
                },
            )
        else:
            raise HighLossError(
                loss_cap=self.loss_cap,
                window_size=self.window_size,
                loss_window=list(self.loss_window),
            )

    def _set_window_size(self, state: State) -> None:
        total_training_steps = 0
        current_step = int(state.timestamp.batch)
        current_token = int(state.timestamp.token)

        if state.max_duration is not None:
            if state.max_duration.unit == TimeUnit.EPOCH and state.dataloader_len is not None:
                total_training_steps = state.dataloader_len * state.max_duration.value
            elif state.max_duration.unit == TimeUnit.BATCH:
                total_training_steps = state.max_duration.value
            elif state.max_duration.unit == TimeUnit.TOKEN:
                # This is an approximation of the total batches from the total tokens, assuming the ratio of tokens:batch is constant.
                total_training_steps = current_step * (
                    state.max_duration.value / current_token
                )
            self.window_size = max(
                self.window_size,
                round(float(total_training_steps * _WINDOW_FRACTION)),
            )
        self.loss_window = deque(self.loss_window, maxlen=self.window_size)
        log.info(f'Window size set to: {self.window_size}')
        self.window_size_set = True

    def batch_end(self, state: State, logger: Logger) -> None:

        if not isinstance(state.loss, torch.Tensor):
            raise NotImplementedError('Multiple losses not supported yet')
        train_loss = state.loss.item()

        # Only start early stopping once a full window of loss data
        if len(self.loss_window) == self.window_size:

            current_step = int(state.timestamp.batch)

            # If window size has not yet been set either by user or during run, set window size to a fraction of the total training duration. Minimum 100 batches.
            if not self.window_size_set:
                self._set_window_size(state)
                # Window size has been expanded -- keep adding losses until we reach the window size.
                if self.window_size > _MIN_WINDOW_SIZE:
                    self.loss_window.append(train_loss)
                    return

            # If user does not provide a loss cap, set loss cap to the maximum loss from the first loss window. Hard cap at loss=10.
            if not self.loss_cap_set and current_step == self.window_size:
                self.loss_cap = min(max(self.loss_window), self.loss_cap)
                self.loss_cap_set = True

            running_loss_avg = float(np.mean(self.loss_window))

            if self._detect_loss_spike(train_loss, running_loss_avg):
                self._handle_loss_spike(logger, running_loss_avg)
            elif self._detect_high_losses(current_step):
                self._handle_high_losses(logger)

        self.loss_window.append(train_loss)
