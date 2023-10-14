# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Log model run events and information."""

import logging
import time

from composer.core import Callback
from composer.loggers import Logger

__all__ = ['RunEventsCallback']

log = logging.getLogger(__name__)


class RunEventsCallback(Callback):
    """Historical model run events and information.

    This callback logs run information including:
    1. timestamp when model data is validated
    """

    def data_validated(self, logger: Logger) -> None:
        logger.log_metrics({
            'data_validated': time.time(),
        })
