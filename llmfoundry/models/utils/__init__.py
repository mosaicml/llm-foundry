# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from llmfoundry.models.utils.meta_init_context import (init_empty_weights,
                                                       init_on_device)
from llmfoundry.models.utils.param_init_fns import generic_param_init_fn_

__all__ = [
    'init_empty_weights',
    'init_on_device',
    'generic_param_init_fn_',
]
