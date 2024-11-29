# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import copy
from typing import Any, Union

import catalogue
import pytest
import torch
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase

from llmfoundry.registry import config_transforms
from llmfoundry.utils.config_utils import (
    TRAIN_CONFIG_KEYS,
    TrainConfig,
    make_dataclass_and_log_config,
)


def generate_exclusive_test_params(param_names: list[str]):
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

    def dummy_transform(config: dict[str, Any]) -> dict[str, Any]:
        config['variables']['fake_key'] = 'fake_value'
        return config

    config_transforms.register('dummy_transform', func=dummy_transform)

    _, parsed_config = make_dataclass_and_log_config(
        config,
        TrainConfig,
        TRAIN_CONFIG_KEYS,
        transforms='all',
    )

    assert isinstance(parsed_config.variables, dict)
    assert parsed_config.variables['fake_key'] == 'fake_value'

    del catalogue.REGISTRY[
        ('llmfoundry', 'config_transforms', 'dummy_transform')]


def test_logged_cfg():
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
    logged_config, _ = make_dataclass_and_log_config(
        config,
        TrainConfig,
        TRAIN_CONFIG_KEYS,
        transforms='all',
    )
    expected_config = copy.deepcopy(config)
    expected_config.update({
        'n_gpus': 1,
        'device_train_batch_size': 1,
        'device_train_grad_accum': 1,
        'device_eval_batch_size': 1,
    })
    assert expected_config == logged_config


class MockTokenizer(PreTrainedTokenizerBase):

    def __init__(self) -> None:
        super().__init__()
        self.pad_token: str = '<pad>'
        self.eos_token: str = '</s>'
        self.bos_token: str = '<s>'
        self.unk_token: str = '<unk>'
        self._vocab_size: int = 128

    def __len__(self) -> int:
        return self._vocab_size

    def convert_tokens_to_ids(
        self,
        tokens: Union[str, list[str]],
    ) -> Union[int, list[int]]:
        return 0

    @property
    def pad_token_id(self) -> int:
        return 0

    def _batch_encode_plus(self, *args: Any,
                           **kwargs: Any) -> dict[str, torch.Tensor]:
        batch_texts = args[0] if args else kwargs.get(
            'batch_text_or_text_pairs',
            [],
        )
        max_length = kwargs.get('max_length', 1024)

        if isinstance(batch_texts[0], list):
            texts = [t for pair in batch_texts for t in pair]
        else:
            texts = batch_texts

        token_ids = torch.tensor([
            [hash(text) % 1000 + j for j in range(max_length)] for text in texts
        ])

        return {
            'input_ids': token_ids,
            'attention_mask': torch.ones_like(token_ids),
        }
