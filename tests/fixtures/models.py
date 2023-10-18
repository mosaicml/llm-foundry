# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable

from composer.devices import Device
from omegaconf import DictConfig
from pytest import fixture
from transformers import PreTrainedTokenizerBase

from llmfoundry.models.hf.hf_causal_lm import ComposerHFCausalLM
from llmfoundry.models.model_registry import COMPOSER_MODEL_REGISTRY
from llmfoundry.models.mpt.modeling_mpt import ComposerMPTCausalLM
from llmfoundry.utils.builders import build_tokenizer


def _build_model(config: DictConfig, tokenizer: PreTrainedTokenizerBase,
                 device: Device):
    model = COMPOSER_MODEL_REGISTRY[config.name](config, tokenizer)
    model = device.module_to_device(model)
    return model


@fixture
def mpt_tokenizer():
    return build_tokenizer('EleutherAI/gpt-neox-20b', {})


@fixture
def build_mpt(
    mpt_tokenizer: PreTrainedTokenizerBase
) -> Callable[..., ComposerMPTCausalLM]:

    def build(device: Device, **kwargs: Any) -> ComposerMPTCausalLM:
        config = DictConfig({
            'name': 'mpt_causal_lm',
            'd_model': 128,
            'n_heads': 4,
            'n_layers': 2,
            'expansion_ratio': 2,
        })
        config.update(kwargs)
        model = _build_model(config, mpt_tokenizer, device)
        assert isinstance(model, ComposerMPTCausalLM)
        return model

    return build


@fixture
def build_hf_mpt(
    mpt_tokenizer: PreTrainedTokenizerBase
) -> Callable[..., ComposerHFCausalLM]:

    def build(device: Device, **kwargs: Any) -> ComposerHFCausalLM:
        config_overrides = {
            'd_model': 128,
            'n_heads': 4,
            'n_layers': 2,
            'expansion_ratio': 2,
        }
        config_overrides.update(kwargs)
        config = DictConfig({
            'name': 'hf_causal_lm',
            'pretrained_model_name_or_path': 'mosaicml/mpt-7b',
            'pretrained': False,
            'config_overrides': config_overrides,
        })
        model = _build_model(config, mpt_tokenizer, device)
        assert isinstance(model, ComposerHFCausalLM)
        return model

    return build
