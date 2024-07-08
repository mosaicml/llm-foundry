# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import signal
import threading
from typing import Optional

from composer import Callback, Logger, State

from llmfoundry.utils.exceptions import RunTimeoutError

log = logging.getLogger(__name__)


def _timeout(timeout: int):
    log.error(f'Timeout after {timeout} seconds of inactivity after fit_end.',)
    try:
        raise RunTimeoutError(timeout=timeout)
    finally:
        os.kill(os.getpid(), signal.SIGINT)


class RunTimeoutCallback(Callback):

    def __init__(
        self,
        timeout: int = 1800,
    ):
        self.timeout = timeout
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
            [self.timeout],
        )
        self.timer.daemon = True
        self.timer.start()

    def fit_end(self, state: State, logger: Logger):
        del state
        del logger
        self._timeout()
