# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import copy
from typing import Any, Callable

import pytest
from pytest import fixture
from transformers import PreTrainedTokenizerBase

from llmfoundry.models.hf.hf_causal_lm import ComposerHFCausalLM
from llmfoundry.models.mpt.modeling_mpt import ComposerMPTCausalLM
from llmfoundry.utils.builders import build_composer_model


def _build_model(config: dict[str, Any], tokenizer: PreTrainedTokenizerBase):
    name = config.pop('name')
    model = build_composer_model(
        name=name,
        cfg=config,
        tokenizer=tokenizer,
    )
    return model


@fixture
def mpt_tokenizer(
    tiny_neox_tokenizer: PreTrainedTokenizerBase,
) -> PreTrainedTokenizerBase:
    return tiny_neox_tokenizer


@fixture
def build_tiny_mpt(
    mpt_tokenizer: PreTrainedTokenizerBase,
) -> Callable[..., ComposerMPTCausalLM]:

    def build(**kwargs: Any) -> ComposerMPTCausalLM:
        config = {
            'name': 'mpt_causal_lm',
            'd_model': 128,
            'n_heads': 4,
            'n_layers': 2,
            'expansion_ratio': 2,
        }
        config.update(kwargs)
        model = _build_model(config, mpt_tokenizer)
        assert isinstance(model, ComposerMPTCausalLM)
        return model

    return build


@fixture
def build_tiny_hf_mpt(
    mpt_tokenizer: PreTrainedTokenizerBase,
) -> Callable[..., ComposerHFCausalLM]:

    def build(**kwargs: Any) -> ComposerHFCausalLM:
        config_overrides = {
            'd_model': 128,
            'n_heads': 4,
            'n_layers': 2,
            'expansion_ratio': 2,
        }
        config_overrides.update(kwargs)
        config = {
            'name': 'hf_causal_lm',
            'pretrained_model_name_or_path': 'mosaicml/mpt-7b',
            'pretrained': False,
            'config_overrides': config_overrides,
        }
        model = _build_model(config, mpt_tokenizer)
        assert isinstance(model, ComposerHFCausalLM)
        return model

    return build


## MODEL HELPERS ##
def causal_lm_model_helper(config):  # type: ignore
    transformers = pytest.importorskip('transformers')

    return transformers.AutoModelForCausalLM.from_config(config)


def masked_lm_model_helper(config):  # type: ignore
    transformers = pytest.importorskip('transformers')

    return transformers.AutoModelForMaskedLM.from_config(
        config,
    )  # type: ignore (thirdparty)


## CONFIG HELPERS ##
def tiny_gpt2_config_helper():
    transformers = pytest.importorskip('transformers')

    tiny_overrides = {
        'n_embd': 2,
        'n_head': 2,
        'n_layer': 2,
        'vocab_size': 50258,  # 50257 + 1 for pad token
    }
    return transformers.AutoConfig.from_pretrained('gpt2', **tiny_overrides)


def tiny_opt_config_helper():
    transformers = pytest.importorskip('transformers')

    tiny_overrides = {
        'n_embd': 2,
        'n_head': 2,
        'n_layer': 2,
        'vocab_size': 50272,
    }
    return transformers.AutoConfig.from_pretrained(
        'facebook/opt-125m',
        **tiny_overrides,
    )


def tiny_codellama_config_helper(tie_word_embeddings: bool = False):
    transformers = pytest.importorskip('transformers')

    tiny_overrides = {
        'num_hidden_layers': 2,
        'hidden_size': 32,
        'intermediate_size': 64,
        'vocab_size': 32016,
        'tie_word_embeddings': tie_word_embeddings,
    }
    return transformers.AutoConfig.from_pretrained(
        'codellama/CodeLlama-7b-hf',
        **tiny_overrides,
    )


def tiny_bert_config_helper():
    transformers = pytest.importorskip('transformers')
    tiny_overrides = {
        'hidden_size': 128,
        'num_attention_heads': 2,
        'num_hidden_layers': 2,
        'intermediate_size': 512,
        'attn_implementation': 'eager',
    }
    return transformers.AutoConfig.from_pretrained(
        'google-bert/bert-base-uncased',
        **tiny_overrides,
    )


## TOKENIZER HELPERS ##
def tiny_gpt2_tokenizer_helper(add_pad: bool = False):
    transformers = pytest.importorskip('transformers')

    hf_tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')

    if add_pad:
        hf_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return hf_tokenizer


def tiny_llama_tokenizer_helper():
    transformers = pytest.importorskip('transformers')

    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(
        'huggyllama/llama-7b',
        use_fast=False,
    )
    return hf_tokenizer


def tiny_codellama_tokenizer_helper():
    transformers = pytest.importorskip('transformers')

    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(
        'codellama/CodeLlama-7b-hf',
    )
    return hf_tokenizer


def tiny_opt_tokenizer_helper():
    transformers = pytest.importorskip('transformers')

    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(
        'facebook/opt-125m',
    )
    hf_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return hf_tokenizer


def tiny_neox_tokenizer_helper():
    transformers = pytest.importorskip('transformers')

    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(
        'EleutherAI/gpt-neox-20b',
        model_max_length=2048,
    )
    return hf_tokenizer


def tiny_neo_tokenizer_helper():
    transformers = pytest.importorskip('transformers')

    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(
        'EleutherAI/gpt-neo-125m',
        model_max_length=2048,
    )
    return hf_tokenizer


def tiny_t5_tokenizer_helper():
    transformers = pytest.importorskip('transformers')

    hf_tokenizer = transformers.AutoTokenizer.from_pretrained('t5-base',)
    return hf_tokenizer


def tiny_bert_tokenizer_helper():
    transformers = pytest.importorskip('transformers')

    return transformers.AutoTokenizer.from_pretrained(
        'google-bert/bert-base-uncased',
    )


def tiny_mpt_tokenizer_helper():
    transformers = pytest.importorskip('transformers')

    return transformers.AutoTokenizer.from_pretrained(
        'mosaicml/mpt-7b',
        model_max_length=2048,
    )


def tiny_mpt_chat_tokenizer_helper():
    transformers = pytest.importorskip('transformers')

    return transformers.AutoTokenizer.from_pretrained(
        'mosaicml/mpt-7b-8k-chat',
        model_max_length=2048,
    )


## SESSION MODELS ##
@pytest.fixture(scope='session')
def _session_tiny_gpt2_model(_session_tiny_gpt2_config):  # type: ignore
    return causal_lm_model_helper(_session_tiny_gpt2_config)


@pytest.fixture(scope='session')
def _session_tiny_opt_model(_session_tiny_opt_config):  # type: ignore
    return causal_lm_model_helper(_session_tiny_opt_config)


@pytest.fixture(scope='session')
def _session_tiny_bert_model(_session_tiny_bert_config):  # type: ignore
    return masked_lm_model_helper(_session_tiny_bert_config)


@pytest.fixture(scope='session')
def _session_tiny_codellama_model(  # type: ignore
    _session_tiny_codellama_config,  # type: ignore
):  # type: ignore
    return causal_lm_model_helper(_session_tiny_codellama_config)


@pytest.fixture(scope='session')
def _session_tiny_codellama_wt_model(  # type: ignore
    _session_tiny_codellama_config,  # type: ignore
):  # type: ignore
    return causal_lm_model_helper(_session_tiny_codellama_wt_config)


## SESSION CONFIGS ##
@pytest.fixture(scope='session')
def _session_tiny_gpt2_config():  # type: ignore
    return tiny_gpt2_config_helper()


@pytest.fixture(scope='session')
def _session_tiny_opt_config():  # type: ignore
    return tiny_opt_config_helper()


@pytest.fixture(scope='session')
def _session_tiny_codellama_config():  # type: ignore
    return tiny_codellama_config_helper()


@pytest.fixture(scope='session')
def _session_tiny_codellama_wt_config():  # type: ignore
    return tiny_codellama_config_helper(tie_word_embeddings=True)


@pytest.fixture(scope='session')
def _session_tiny_bert_config():  # type: ignore
    return tiny_bert_config_helper()


## SESSION TOKENIZERS ##
@pytest.fixture(scope='session')
def _session_tiny_gpt2_tokenizer():  # type: ignore
    return tiny_gpt2_tokenizer_helper()

@pytest.fixture(scope='session')
def _session_tiny_gpt2_with_pad_tokenizer():  # type: ignore
    return tiny_gpt2_tokenizer_helper(add_pad=True)

@pytest.fixture(scope='session')
def _session_tiny_llama_tokenizer():  # type: ignore
    return tiny_llama_tokenizer_helper()


@pytest.fixture(scope='session')
def _session_tiny_codellama_tokenizer():  # type: ignore
    return tiny_codellama_tokenizer_helper()


@pytest.fixture(scope='session')
def _session_tiny_opt_tokenizer():  # type: ignore
    return tiny_opt_tokenizer_helper()


@pytest.fixture(scope='session')
def _session_tiny_neox_tokenizer():  # type: ignore
    return tiny_neox_tokenizer_helper()


@pytest.fixture(scope='session')
def _session_tiny_neo_tokenizer():  # type: ignore
    return tiny_neo_tokenizer_helper()


@pytest.fixture(scope='session')
def _session_tiny_t5_tokenizer():  # type: ignore
    return tiny_t5_tokenizer_helper()


@pytest.fixture(scope='session')
def _session_tiny_bert_tokenizer():  # type: ignore
    return tiny_bert_tokenizer_helper()


@pytest.fixture(scope='session')
def _session_tiny_mpt_tokenizer():  # type: ignore
    return tiny_mpt_tokenizer_helper()


@pytest.fixture(scope='session')
def _session_tiny_mpt_chat_tokenizer():  # type: ignore
    return tiny_mpt_chat_tokenizer_helper()


## MODEL FIXTURES ##
@pytest.fixture
def tiny_gpt2_model(_session_tiny_gpt2_model):  # type: ignore
    return copy.deepcopy(_session_tiny_gpt2_model)


@pytest.fixture
def tiny_opt_model(_session_tiny_opt_model):  # type: ignore
    return copy.deepcopy(_session_tiny_opt_model)


@pytest.fixture
def tiny_codellama_model(_session_tiny_codellama_model):  # type: ignore
    return copy.deepcopy(_session_tiny_codellama_model)


@pytest.fixture
def tiny_codellama_wt_model(_session_tiny_codellama_wt_model):  # type: ignore
    return copy.deepcopy(_session_tiny_codellama_wt_model)


@pytest.fixture
def tiny_bert_model(_session_tiny_bert_model):  # type: ignore
    return copy.deepcopy(_session_tiny_bert_model)


## CONFIG FIXTURES ##
@pytest.fixture
def tiny_bert_config(_session_tiny_bert_config):  # type: ignore
    return copy.deepcopy(_session_tiny_bert_config)


@pytest.fixture
def tiny_codellama_wt_config(_session_tiny_codellama_wt_config):  # type: ignore
    return copy.deepcopy(_session_tiny_codellama_wt_config)


## TOKENIZER FIXTURES ##
@pytest.fixture
def tiny_gpt2_tokenizer(_session_tiny_gpt2_tokenizer):  # type: ignore
    return copy.deepcopy(_session_tiny_gpt2_tokenizer)

@pytest.fixture
def tiny_gpt2_with_pad_tokenizer(_session_tiny_gpt2_with_pad_tokenizer):  # type: ignore
    return copy.deepcopy(_session_tiny_gpt2_with_pad_tokenizer)

@pytest.fixture
def tiny_llama_tokenizer(_session_tiny_llama_tokenizer):  # type: ignore
    return copy.deepcopy(_session_tiny_llama_tokenizer)


@pytest.fixture
def tiny_codellama_tokenizer(_session_tiny_codellama_tokenizer):  # type: ignore
    return copy.deepcopy(_session_tiny_codellama_tokenizer)


@pytest.fixture
def tiny_opt_tokenizer(_session_tiny_opt_tokenizer):  # type: ignore
    return copy.deepcopy(_session_tiny_opt_tokenizer)


@pytest.fixture
def tiny_neox_tokenizer(_session_tiny_neox_tokenizer):  # type: ignore
    return copy.deepcopy(_session_tiny_neox_tokenizer)


@pytest.fixture
def tiny_neo_tokenizer(_session_tiny_neo_tokenizer):  # type: ignore
    return copy.deepcopy(_session_tiny_neo_tokenizer)


@pytest.fixture
def tiny_t5_tokenizer(_session_tiny_t5_tokenizer):  # type: ignore
    return copy.deepcopy(_session_tiny_t5_tokenizer)


@pytest.fixture
def tiny_bert_tokenizer(_session_tiny_bert_tokenizer):  # type: ignore
    return copy.deepcopy(_session_tiny_bert_tokenizer)


@pytest.fixture
def tiny_mpt_tokenizer(_session_tiny_mpt_tokenizer):  # type: ignore
    return copy.deepcopy(_session_tiny_mpt_tokenizer)


@pytest.fixture
def tiny_mpt_chat_tokenizer(_session_tiny_mpt_chat_tokenizer):  # type: ignore
    return copy.deepcopy(_session_tiny_mpt_chat_tokenizer)


## GETTERS ##
def get_tokenizer_fixture_by_name(
    request: pytest.FixtureRequest,
    name: str,
) -> PreTrainedTokenizerBase:
    name_to_fixture_name = {
        'gpt2': 'tiny_gpt2_tokenizer',
        'huggyllama/llama-7b': 'tiny_llama_tokenizer',
        'codellama/CodeLlama-7b-hf': 'tiny_codellama_tokenizer',
        'facebook/opt-125m': 'tiny_opt_tokenizer',
        'EleutherAI/gpt-neox-20b': 'tiny_neox_tokenizer',
        'EleutherAI/gpt-neo-125m': 'tiny_neo_tokenizer',
        't5-base': 'tiny_t5_tokenizer',
        'google-bert/bert-base-uncased': 'tiny_bert_tokenizer',
        'mosaicml/mpt-7b': 'tiny_mpt_tokenizer',
        'mosaicml/mpt-7b-8k-chat': 'tiny_mpt_chat_tokenizer',
    }
    return request.getfixturevalue(name_to_fixture_name[name])
