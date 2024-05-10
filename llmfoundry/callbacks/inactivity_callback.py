# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import signal
import threading
from typing import Optional

from composer import Callback, Logger, State
from composer.loggers import MosaicMLLogger

from llmfoundry.utils.exceptions import RunTimeoutError

log = logging.getLogger(__name__)


def _timeout(timeout: int, mosaicml_logger: Optional[MosaicMLLogger] = None):
    log.error(
        f'Timeout after no Trainer events were triggered for {timeout} seconds.',
    )
    if mosaicml_logger is not None:
        mosaicml_logger.log_exception(RunTimeoutError(timeout=timeout))
    os.kill(os.getpid(), signal.SIGINT)


class InactivityCallback(Callback):

    def __init__(
        self,
        timeout: int = 1800,
        mosaicml_logger: Optional[MosaicMLLogger] = None,
    ):
        self.timeout = timeout
        self.mosaicml_logger = mosaicml_logger
        self.timer: Optional[threading.Timer] = None

    def _reset(self):
        if self.timer is not None:
            self.timer.cancel()
        self.timer = None

    def _timeout(self):
        self._reset()
        self.timer = threading.Timer(
            self.timeout,
            _timeout,
            [self.timeout, self.mosaicml_logger],
        )
        self.timer.daemon = True
        self.timer.start()

    def fit_end(self, state: State, logger: Logger):
        del state
        del logger
        self._timeout()
