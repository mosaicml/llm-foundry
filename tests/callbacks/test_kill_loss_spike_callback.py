# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
import unittest
from unittest.mock import patch
from llmfoundry.callbacks.kill_loss_spike_callback import KillLossSpike
from collections import deque

class TestKillLossSpike(unittest.TestCase):
    def setUp(self):
        self.callback = KillLossSpike(log_only=True, patience=4, outlier_multiplier=2, window_size=10, loss_cap=10) # type: ignore

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
        train_loss = 5
        running_loss_avg = 2
        result = self.callback.detect_loss_spike(train_loss, running_loss_avg)
        self.assertTrue(result)

    @patch('llmfoundry.callbacks.kill_loss_spike_callback.log')
    def test_detect_high_losses_no_high_losses(self, _):
        self.callback.loss_window = deque([2] * 10, maxlen=10)
        current_step = 21
        result = self.callback.detect_high_losses(current_step)
        self.assertFalse(result)

    @patch('llmfoundry.callbacks.kill_loss_spike_callback.log')
    def test_detect_high_losses_with_high_losses(self, _):
        self.callback.loss_window = deque([11] * 10, maxlen=10)  # Simulate high losses in loss window
        current_step = 21
        result = self.callback.detect_high_losses(current_step)
        self.assertTrue(result)

if __name__ == '__main__':
    unittest.main()