# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import copy
from typing import Any, Callable

import pytest
from omegaconf import DictConfig
from pytest import fixture
from transformers import PreTrainedTokenizerBase

from llmfoundry.models.hf.hf_causal_lm import ComposerHFCausalLM
from llmfoundry.models.mpt.modeling_mpt import ComposerMPTCausalLM
from llmfoundry.utils.builders import build_composer_model, build_tokenizer


def _build_model(config: DictConfig, tokenizer: PreTrainedTokenizerBase):
    model = build_composer_model(
        composer_model_name=config.name,
        tokenizer=tokenizer,
        cfg=config,
    )
    return model


@fixture
def mpt_tokenizer():
    return build_tokenizer('EleutherAI/gpt-neox-20b', {})


@fixture
def build_tiny_mpt(
    mpt_tokenizer: PreTrainedTokenizerBase
) -> Callable[..., ComposerMPTCausalLM]:

    def build(**kwargs: Any) -> ComposerMPTCausalLM:
        config = DictConfig({
            'name': 'mpt_causal_lm',
            'd_model': 128,
            'n_heads': 4,
            'n_layers': 2,
            'expansion_ratio': 2,
        })
        config.update(kwargs)
        model = _build_model(config, mpt_tokenizer)
        assert isinstance(model, ComposerMPTCausalLM)
        return model

    return build


@fixture
def build_tiny_hf_mpt(
    mpt_tokenizer: PreTrainedTokenizerBase
) -> Callable[..., ComposerHFCausalLM]:

    def build(**kwargs: Any) -> ComposerHFCausalLM:
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
        model = _build_model(config, mpt_tokenizer)
        assert isinstance(model, ComposerHFCausalLM)
        return model

    return build


def tiny_gpt2_model_helper(config):  # type: ignore
    transformers = pytest.importorskip('transformers')

    return transformers.AutoModelForCausalLM.from_config(config)


@pytest.fixture(scope='session')
def _session_tiny_gpt2_model(_session_tiny_gpt2_config):  # type: ignore
    return tiny_gpt2_model_helper(_session_tiny_gpt2_config)


def tiny_gpt2_config_helper():
    transformers = pytest.importorskip('transformers')

    tiny_overrides = {
        'n_embd': 2,
        'n_head': 2,
        'n_layer': 2,
        'vocab_size': 50258  # 50257 + 1 for pad token
    }
    return transformers.AutoConfig.from_pretrained('gpt2', **tiny_overrides)


@pytest.fixture(scope='session')
def _session_tiny_gpt2_config():  # type: ignore
    return tiny_gpt2_config_helper()


def tiny_gpt2_tokenizer_helper():
    transformers = pytest.importorskip('transformers')

    hf_tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')
    hf_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return hf_tokenizer


@pytest.fixture
def tiny_gpt2_model(_session_tiny_gpt2_model):  # type: ignore
    return copy.deepcopy(_session_tiny_gpt2_model)


@pytest.fixture(scope='session')
def _session_tiny_gpt2_tokenizer():  # type: ignore
    return tiny_gpt2_tokenizer_helper()


@pytest.fixture
def tiny_gpt2_tokenizer(_session_tiny_gpt2_tokenizer):  # type: ignore
    return copy.deepcopy(_session_tiny_gpt2_tokenizer)


def tiny_llama_tokenizer_helper():
    transformers = pytest.importorskip('transformers')

    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(
        'huggyllama/llama-7b', use_fast=False)
    return hf_tokenizer


@pytest.fixture(scope='session')
def _session_tiny_llama_tokenizer():  # type: ignore
    return tiny_llama_tokenizer_helper()


@pytest.fixture
def tiny_llama_tokenizer(_session_tiny_llama_tokenizer):  # type: ignore
    return copy.deepcopy(_session_tiny_llama_tokenizer)


def tiny_opt_tokenizer_helper():
    transformers = pytest.importorskip('transformers')

    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(
        'facebook/opt-125m')
    hf_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return hf_tokenizer


def tiny_opt_model_helper(config):  # type: ignore
    transformers = pytest.importorskip('transformers')

    return transformers.AutoModelForCausalLM.from_config(config)


@pytest.fixture(scope='session')
def _session_tiny_opt_tokenizer():  # type: ignore
    return tiny_opt_tokenizer_helper()


@pytest.fixture(scope='session')
def _session_tiny_opt_config():  # type: ignore
    return tiny_opt_config_helper()


@pytest.fixture(scope='session')
def _session_tiny_opt_model(_session_tiny_opt_config):  # type: ignore
    return tiny_opt_model_helper(_session_tiny_opt_config)


def tiny_opt_config_helper():
    transformers = pytest.importorskip('transformers')

    tiny_overrides = {
        'n_embd': 2,
        'n_head': 2,
        'n_layer': 2,
        'vocab_size': 50272
    }
    return transformers.AutoConfig.from_pretrained('facebook/opt-125m',
                                                   **tiny_overrides)


@pytest.fixture
def tiny_opt_tokenizer(_session_tiny_opt_tokenizer):  # type: ignore
    return copy.deepcopy(_session_tiny_opt_tokenizer)


@pytest.fixture
def tiny_opt_model(_session_tiny_opt_model):  # type: ignore
    return copy.deepcopy(_session_tiny_opt_model)
