# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import logging
import os

from composer.loggers import MosaicMLLogger
from composer.loggers.mosaicml_logger import (MOSAICML_ACCESS_TOKEN_ENV_VAR,
                                              MOSAICML_PLATFORM_ENV_VAR)

__all__ = [
    'SpecificWarningFilter',
]


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


def get_mosaicml_logger():
    if os.environ.get(MOSAICML_PLATFORM_ENV_VAR, 'false').lower(
    ) == 'true' and os.environ.get(MOSAICML_ACCESS_TOKEN_ENV_VAR):
        # Adds mosaicml logger to composer if the run was sent from Mosaic platform, access token is set
        return MosaicMLLogger()
    else:
        return None
