# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List

import catalogue
import pytest
from omegaconf import DictConfig

from llmfoundry.registry import config_transforms
from llmfoundry.utils.config_utils import (
    TRAIN_CONFIG_KEYS,
    TrainConfig,
    make_dataclass_and_log_config,
)


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


def test_config_transforms():
    config = DictConfig({
        'global_train_batch_size': 1,
        'device_train_microbatch_size': 1,
        'model': {},
        'scheduler': {},
        'max_seq_len': 128,
        'train_loader': {},
        'max_duration': 1,
        'tokenizer': {},
        'eval_interval': 1,
        'seed': 1,
        'optimizer': {},
        'variables': {},
    },)

    def dummy_transform(config: Dict[str, Any]) -> Dict[str, Any]:
        config['variables']['fake_key'] = 'fake_value'
        return config

    config_transforms.register('dummy_transform', func=dummy_transform)

    _, parsed_config = make_dataclass_and_log_config(
        config,
        TrainConfig,
        TRAIN_CONFIG_KEYS,
        transforms='all',
    )

    assert isinstance(parsed_config.variables, Dict)
    assert parsed_config.variables['fake_key'] == 'fake_value'

    del catalogue.REGISTRY[
        ('llmfoundry', 'config_transforms', 'dummy_transform')]
