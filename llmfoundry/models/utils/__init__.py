# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from llmfoundry.models.utils.config_moe_args import config_moe_args
from llmfoundry.models.utils.meta_init_context import (init_empty_weights,
                                                       init_on_device)
from llmfoundry.models.utils.mpt_param_count import (mpt_get_active_params,
                                                     mpt_get_total_params)
from llmfoundry.models.utils.param_init_fns import (MODEL_INIT_REGISTRY,
                                                    generic_param_init_fn_)

__all__ = [
    'init_empty_weights',
    'init_on_device',
    'generic_param_init_fn_',
    'MODEL_INIT_REGISTRY',
    'config_moe_args',
    'mpt_get_active_params',
    'mpt_get_total_params',
]
