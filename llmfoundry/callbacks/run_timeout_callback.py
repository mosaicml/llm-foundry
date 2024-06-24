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
from llmfoundry.utils.mosaicml_logger_utils import no_override_excepthook

log = logging.getLogger(__name__)


def _timeout(timeout: int, mosaicml_logger: Optional[MosaicMLLogger] = None):
    log.error(f'Timeout after {timeout} seconds of inactivity after fit_end.',)
    if mosaicml_logger is not None and no_override_excepthook():
        mosaicml_logger.log_exception(RunTimeoutError(timeout=timeout))
    os.kill(os.getpid(), signal.SIGINT)


class RunTimeoutCallback(Callback):

    def __init__(
        self,
        timeout: int = 1800,
    ):
        self.timeout = timeout
        self.mosaicml_logger: Optional[MosaicMLLogger] = None
        self.timer: Optional[threading.Timer] = None

    def init(self, state: State, logger: Logger):
        for callback in state.callbacks:
            if isinstance(callback, MosaicMLLogger):
                self.mosaicml_logger = callback

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
