# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from torch import nn

FC_CLASS_REGISTRY = {
    'torch': nn.Linear,
}

try:
    import transformer_engine.pytorch as te
    FC_CLASS_REGISTRY['te'] = te.Linear
except:
    pass

try:
    import float8_experimental
    FC_CLASS_REGISTRY['fp8'] = float8_experimental.float8_linear.Float8Linear
    FC_CLASS_REGISTRY['fp8dl'] = float8_experimental.float8_dynamic_linear.Float8DynamicLinear
except:
    pass
