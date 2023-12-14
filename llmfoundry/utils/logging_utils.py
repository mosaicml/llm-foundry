# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import logging


class SpecificWarningFilter(logging.Filter):

    def __init__(self, message_to_suppress: str):
        super().__init__()
        self.message_to_suppress = message_to_suppress

    def filter(self, record: logging.LogRecord):
        return self.message_to_suppress not in record.getMessage()
