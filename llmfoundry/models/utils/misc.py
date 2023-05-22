# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
import torch
from packaging import version


def is_torch_2_or_higher():
    if version.parse(torch.__version__) >= version.parse('2.0.0'):
        return True
    return False
