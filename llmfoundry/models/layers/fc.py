# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from torch import nn

try:
    import transformer_engine.pytorch as te
    teLinear = te.Linear
except:
    teLinear = None

FC_CLASS_REGISTRY = {
    'torch': nn.Linear,
    'te': teLinear,
}
