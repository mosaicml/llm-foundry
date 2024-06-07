# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from composer.callbacks import SystemMetricsMonitor
from llmfoundry.utils.builders import build_callback


def test_system_metrics_monitor_callback_builds():
    kwargs = {'log_all_data': True}
    callback = build_callback(
        'system_metrics_monitor',
        kwargs=kwargs,
        train_config={'train_loader': {}},
    )
    assert isinstance(callback, SystemMetricsMonitor)
    assert getattr(callback, 'log_all_data', None) is True
