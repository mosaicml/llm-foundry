# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
import unittest
from collections import deque
from unittest.mock import MagicMock, patch

from composer.core.time import TimeUnit

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
        result = self.callback._detect_loss_spike(train_loss, running_loss_avg)
        self.assertFalse(result)

    @patch('llmfoundry.callbacks.kill_loss_spike_callback.log')
    def test_detect_loss_spike_with_spike(self, _):
        self.callback.outlier_counter = 4  # Simulating previous spikes
        train_loss = 4
        running_loss_avg = 2
        result = self.callback._detect_loss_spike(train_loss, running_loss_avg)
        self.assertTrue(result)

    @patch('llmfoundry.callbacks.kill_loss_spike_callback.log')
    def test_handle_loss_spike_logs_only_when_log_only_true(self, _):
        logger = MagicMock()
        running_loss_avg = 2
        self.callback.log_only = True
        self.callback.outlier_counter = 5

        try:
            self.callback._handle_loss_spike(logger, running_loss_avg)
        except LossSpikeError:
            self.fail('LossSpikeError was raised unexpectedly')

    @patch('llmfoundry.callbacks.kill_loss_spike_callback.log')
    def test_handle_loss_spike_raises_error_log_only_false(self, _):
        logger = MagicMock()
        running_loss_avg = 2
        self.callback.log_only = False
        self.callback.outlier_counter = 5

        # LossSpikeError is raised
        with self.assertRaises(LossSpikeError):
            self.callback._handle_loss_spike(logger, running_loss_avg)

    @patch('llmfoundry.callbacks.kill_loss_spike_callback.log')
    def test_detect_high_losses_no_high_losses(self, _):
        self.callback.loss_window = deque([2] * 10, maxlen=10)
        current_step = 21
        result = self.callback._detect_high_losses(current_step)
        self.assertFalse(result)

    @patch('llmfoundry.callbacks.kill_loss_spike_callback.log')
    def test_detect_high_losses_with_high_losses(self, _):
        self.callback.loss_window = deque(
            [9, 8, 7, 6, 5, 11, 12, 13, 14, 15],
            maxlen=10,
        )  # Simulate mix of losses in loss window
        current_step = 21
        result = self.callback._detect_high_losses(current_step)
        self.assertTrue(result)

    @patch('llmfoundry.callbacks.kill_loss_spike_callback.log')
    def test_set_window_size_from_token(self, _):
        state = MagicMock()
        state.max_duration.unit = TimeUnit.TOKEN
        state.max_duration.value = 100000
        state.timestamp.batch = 100
        state.timestamp.token = 4000

        self.callback._set_window_size(state)

        self.assertEqual(self.callback.window_size, 125)
        self.assertTrue(self.callback.window_size_set)

    @patch('llmfoundry.callbacks.kill_loss_spike_callback.log')
    def test_set_window_size_from_epoch(self, _):
        state = MagicMock()
        state.max_duration.unit = TimeUnit.EPOCH
        state.dataloader_len = 1000
        state.max_duration.value = 3
        state.timestamp.batch = 100

        self.callback._set_window_size(state)

        self.assertEqual(self.callback.window_size, 150)
        self.assertTrue(self.callback.window_size_set)
