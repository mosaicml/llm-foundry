# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from composer import Callback, Logger, State

from llmfoundry.utils.warnings import experimental_class

__all__ = [
    'SlidingWindowSizeScheduler',
]


@experimental_class('SlidingWindowSizeScheduler')
class SlidingWindowSizeScheduler(Callback):

    def before_train_batch(self, state: State, logger: Logger):
        del logger, state
