# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import List

from composer.loggers import MosaicMLLogger
from composer.loggers.logger_destination import LoggerDestination


def find_mosaicml_logger(loggers: List[LoggerDestination]) -> MosaicMLLogger:
    return next(
        (logger for logger in loggers if isinstance(logger, MosaicMLLogger)),
        None)


class SpecificWarningFilter(logging.Filter):

    def __init__(self, message_to_suppress: str):
        """Filter out a specific warning message based on its content.

        This can be useful for filtering out specific warning messages from third party packages.

        Args:
            message_to_suppress (str): The warning message to suppress.
        """
        super().__init__()
        self.message_to_suppress = message_to_suppress

    def filter(self, record: logging.LogRecord) -> bool:
        return self.message_to_suppress not in record.getMessage()
