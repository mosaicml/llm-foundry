# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from llmfoundry.utils.builders import build_callback


def test_curriculum_learning_callback_builds():
    kwargs = {'dataset_index': 0}
    callback = build_callback(
        'curriculum_learning',
        kwargs=kwargs,
        train_config={'train_loader': {}},
    )
    assert callback is not None
