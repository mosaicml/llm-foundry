# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from typing import List

import pytest


def generate_exclusive_test_params(param_names: List[str]):
    for _, name in enumerate(param_names):
        params = {param_name: False for param_name in param_names}
        params[name] = True
        param_values = list(params.values())
        param_id = f'{name}=True'
        yield pytest.param(*param_values, id=param_id)
