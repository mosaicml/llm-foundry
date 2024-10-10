# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from composer.core import Callback

from llmfoundry.registry import callbacks, callbacks_with_config
from llmfoundry.utils.builders import build_callback


def test_build_callbacks():
    entry_points = callbacks.get_entry_points()
    registered_callbacks = entry_points.keys()

    for callback_name in registered_callbacks:
        callback = build_callback(
            callback_name,
            kwargs={},
        )
        assert isinstance(callback, entry_points[callback_name])
        assert isinstance(callback, Callback)


def test_build_callbacks_with_config():
    entry_points = callbacks_with_config.get_entry_points()
    registered_callbacks = entry_points.keys()

    for callback_name in registered_callbacks:
        callback = build_callback(
            callback_name,
            kwargs={},
            train_config={'train_loader': {}},
        )
        assert isinstance(callback, entry_points[callback_name])
        assert isinstance(callback, Callback)
