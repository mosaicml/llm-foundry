# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import datetime

__all__ = [
    'get_date_string',
]


def get_date_string() -> str:
    """Get the current date string."""
    return datetime.datetime.now().strftime('%d %b %Y')
