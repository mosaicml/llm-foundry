# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from typing import List

import pytest


def generate_exclusive_test_params(param_names: List[str]):
    """Generates pytest.param objects with one true parameter for testing.

    Creates pytest.param objects for each parameter name given. For each
    param object, one parameter is set to True (indicating a test case for
    malformed data) while the rest are set to False.

    Args:
        param_names (List[str]): The names of parameters to create test cases for.

    Yields:
        pytest.param: Each with one parameter set to True, indicating the specific case being tested.
    """
    for _, name in enumerate(param_names):
        params = {param_name: False for param_name in param_names}
        params[name] = True
        param_values = list(params.values())
        param_id = f'{name}=True'
        yield pytest.param(*param_values, id=param_id)
