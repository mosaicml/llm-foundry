# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from composer.callbacks import SystemMetricsMonitor

from llmfoundry.utils.builders import build_callback


def test_system_metrics_monitor_callback_builds():
    callback = build_callback(
        'system_metrics_monitor',
        kwargs={},
        train_config={'train_loader': {}},
    )
    assert isinstance(callback, SystemMetricsMonitor)