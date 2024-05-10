from typing import Optional
from composer import Callback, State, Logger
import logging
import os
import signal
import threading

log = logging.getLogger(__name__)

def _timeout(timeout: int):
    log.error(f"Timeout after no Trainer events were triggered for {timeout} seconds.")
    os.kill(os.getpid(), signal.SIGINT)

class InactivityCallback(Callback):
    def __init__(self, timeout: int = 1800):
        self.timeout = timeout
        self.timer: Optional[threading.Timer] = None

    def _reset(self):
        if self.timer is not None:
            self.timer.cancel()
        self.timer = None

    def _timeout(self):
        self._reset()
        self.timer = threading.Timer(self.timeout, _timeout, [self.timeout])
        self.timer.daemon = True
        self.timer.start()

    def fit_end(self, state: State, logger: Logger):
        del state
        del logger
        self._timeout()