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
    2. the total number of samples in the training dataset
    """      
    
    def data_validated(self, logger: Logger, total_num_samples: int) -> None:
       logger.log_metrics({
           'data_validated': time.time(),
           'total_num_samples': total_num_samples})
