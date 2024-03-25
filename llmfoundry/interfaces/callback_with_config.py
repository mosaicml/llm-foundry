# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import abc
from typing import Any

from composer.core import Callback

__all__ = ['CallbackWithConfig']


class CallbackWithConfig(Callback, abc.ABC):
    """A callback that takes a config dictionary as an argument, in addition to.

    its other kwargs.
    """

    def __init__(self, config: dict[str, Any], *args: Any,
                 **kwargs: Any) -> None:
        del config, args, kwargs
        pass
