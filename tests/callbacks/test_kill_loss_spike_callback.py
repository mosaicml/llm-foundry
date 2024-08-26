# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
import unittest
from collections import deque
from unittest.mock import MagicMock, patch

import torch
from composer.core import State, Timestamp
from composer.devices import DeviceCPU
from composer.loggers import Logger, MosaicMLLogger

from llmfoundry.callbacks.kill_loss_spike_callback import KillLossSpike
from llmfoundry.utils.exceptions import LossSpikeError


class TestKillLossSpike(unittest.TestCase):

    def __init__(self, *args: str, **kwargs: dict):
        super(TestKillLossSpike, self).__init__(*args, **kwargs)
        self.callback = KillLossSpike(
            log_only=True,
            patience=4,
            outlier_multiplier=2,
        )
        self.callback.window_size = 10
        self.callback.loss_cap = 10

    @patch('llmfoundry.callbacks.kill_loss_spike_callback.log')
    def test_detect_loss_spike_no_spike(self, _):
        self.callback.outlier_counter = 0
        train_loss = 4
        running_loss_avg = 2
        result = self.callback.detect_loss_spike(train_loss, running_loss_avg)
        self.assertFalse(result)

    @patch('llmfoundry.callbacks.kill_loss_spike_callback.log')
    def test_detect_loss_spike_with_spike(self, _):
        self.callback.outlier_counter = 4  # Simulating previous spikes
        train_loss = 4
        running_loss_avg = 2
        result = self.callback.detect_loss_spike(train_loss, running_loss_avg)
        self.assertTrue(result)

    @patch('llmfoundry.callbacks.kill_loss_spike_callback.log')
    def test_no_error_raised_with_log_only_true(self, _):
        build_tiny_mpt = MagicMock()
        build_tiny_mpt.return_value = MagicMock()
        state = State(
            model=build_tiny_mpt(loss_fn='torch_crossentropy'),
            rank_zero_seed=0,
            run_name='test_state',
            device=DeviceCPU(),
        )
        state.loss = torch.tensor(4)
        state.timestamp = Timestamp(batch=21)
        logger = Logger(state, destinations=[MosaicMLLogger()])

        # Loss spike detection should trigger
        self.callback.outlier_counter = 4
        self.callback.loss_window = deque([2] * 10, maxlen=10)

        result = self.callback.detect_loss_spike(state.loss.item(), 2)
        self.assertTrue(result)

        # batch_end should not raise an error due to log_only=True
        try:
            self.callback.batch_end(state, logger)
        except Exception as e:
            self.fail(f'batch_end raised an exception {e} with log_only=True')

    @patch('llmfoundry.callbacks.kill_loss_spike_callback.log')
    def test_error_raised_with_log_only_false(self, _):
        build_tiny_mpt = MagicMock()
        build_tiny_mpt.return_value = MagicMock()
        state = State(
            model=build_tiny_mpt(loss_fn='torch_crossentropy'),
            rank_zero_seed=0,
            run_name='test_state',
            device=DeviceCPU(),
        )
        state.loss = torch.tensor(4)
        state.timestamp = Timestamp(batch=21)
        logger = Logger(state, destinations=[MosaicMLLogger()])

        # Loss spike detection should trigger
        self.callback.outlier_counter = 4
        self.callback.loss_window = deque([2] * 10, maxlen=10)
        self.callback.log_only = False

        result = self.callback.detect_loss_spike(state.loss.item(), 2)
        self.assertTrue(result)

        # batch_end should raise an error due to log_only=False
        with self.assertRaises(LossSpikeError):
            self.callback.batch_end(state, logger)

    @patch('llmfoundry.callbacks.kill_loss_spike_callback.log')
    def test_detect_high_losses_no_high_losses(self, _):
        self.callback.loss_window = deque([2] * 10, maxlen=10)
        current_step = 21
        result = self.callback.detect_high_losses(current_step)
        self.assertFalse(result)

    @patch('llmfoundry.callbacks.kill_loss_spike_callback.log')
    def test_detect_high_losses_with_high_losses(self, _):
        self.callback.loss_window = deque(
            [9, 8, 7, 6, 5, 11, 12, 13, 14, 15],
            maxlen=10,
        )  # Simulate mix of losses in loss window
        current_step = 21
        result = self.callback.detect_high_losses(current_step)
        self.assertTrue(result)
