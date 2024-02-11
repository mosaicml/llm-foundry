# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Enable and configure float8 linear layers in the model.

This callback is currently experimental. The API may change without warning in
the future.
"""

import logging
from typing import Any, Dict

from composer.core import Callback, State
from composer.loggers import Logger
from streaming import StreamingDataset
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)

class Float8Linears(Callback):
    """Enable and configure float8 linear layers in the model.

    This callback is currently experimental. The API may change without warning in the future.
    """

    def __init__(self):
        try:
            import float8_experimental
        except:
            raise ModuleNotFoundError(
                f'float8_experimental package is not installed. Please install the ',
                'float8_experimental library to use the Float8Linears callback.'
            )
        
    def init(self, state: State, logger: Logger):
        del state, logger

        from float8_experimental.float8_linear_utils import (
                swap_linear_with_float8_linear,
                sync_float8_amax_and_scale_history,
            )
        from float8_experimental.float8_linear import Float8Linear

        # Configure fp8 training with float8_experimental
        # from float8_experimental import config
        # config.enable_amax_init = False  # only needed for autocast + compile + FSDP +  float8 delayed
        # config.enable_pre_and_post_forward = False  # only needed for autocast + compile + FSDP +  float8 delayed

    def after_train_batch(self, state: State, logger: Logger):
        del logger

        from float8_experimental.float8_linear_utils import sync_float8_amax_and_scale_history

        # Sync fp8 scales and amaxes before optimizer step.
        sync_float8_amax_and_scale_history(state.model)
