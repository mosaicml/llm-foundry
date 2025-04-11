# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import copy
from typing import Any, Callable

import pytest
from pytest import fixture
from tenacity import retry, stop_after_attempt, wait_fixed
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
    pytest.importorskip('transformers')
    from transformers.models.gpt2.configuration_gpt2 import GPT2Config
    config_dict = {
        'activation_function': 'gelu_new',
        'architectures': ['GPT2LMHeadModel',],
        'attn_pdrop': 0.1,
        'bos_token_id': 50256,
        'embd_pdrop': 0.1,
        'eos_token_id': 50256,
        'initializer_range': 0.02,
        'layer_norm_epsilon': 1e-05,
        'model_type': 'gpt2',
        'n_ctx': 1024,
        'n_embd': 2,
        'n_head': 2,
        'n_layer': 2,
        'n_positions': 1024,
        'resid_pdrop': 0.1,
        'summary_activation': None,
        'summary_first_dropout': 0.1,
        'summary_proj_to_labels': True,
        'summary_type': 'cls_index',
        'summary_use_proj': True,
        'task_specific_params': {
            'text-generation': {
                'do_sample': True,
                'max_length': 50,
            },
        },
        'vocab_size': 50258,
    }

    config_object = GPT2Config(**config_dict,)
    return config_object


def tiny_codellama_config_helper(tie_word_embeddings: bool = False):
    pytest.importorskip('transformers')
    from transformers.models.llama.configuration_llama import LlamaConfig

    config_dict = {
        '_name_or_path': 'codellama/CodeLlama-7b-hf',
        'architectures': ['LlamaForCausalLM',],
        'bos_token_id': 1,
        'eos_token_id': 2,
        'hidden_act': 'silu',
        'hidden_size': 32,
        'initializer_range': 0.02,
        'intermediate_size': 64,
        'max_position_embeddings': 16384,
        'model_type': 'llama',
        'num_attention_heads': 32,
        'num_hidden_layers': 2,
        'num_key_value_heads': 32,
        'pretraining_tp': 1,
        'rms_norm_eps': 1e-05,
        'rope_scaling': None,
        'rope_theta': 1000000,
        'tie_word_embeddings': tie_word_embeddings,
        'torch_dtype': 'bfloat16',
        'transformers_version': '4.33.0.dev0',
        'use_cache': True,
        'vocab_size': 32016,
    }

    config_object = LlamaConfig(**config_dict,)
    return config_object


def tiny_bert_config_helper():
    pytest.importorskip('transformers')
    from transformers.models.bert.configuration_bert import BertConfig

    config_object = {
        'architectures': ['BertForMaskedLM',],
        'attn_implementation': 'eager',
        'attention_probs_dropout_prob': 0.1,
        'gradient_checkpointing': False,
        'hidden_act': 'gelu',
        'hidden_dropout_prob': 0.1,
        'hidden_size': 128,
        'initializer_range': 0.02,
        'intermediate_size': 512,
        'layer_norm_eps': 1e-12,
        'max_position_embeddings': 512,
        'model_type': 'bert',
        'num_attention_heads': 2,
        'num_hidden_layers': 2,
        'pad_token_id': 0,
        'position_embedding_type': 'absolute',
        'transformers_version': '4.6.0.dev0',
        'type_vocab_size': 2,
        'use_cache': True,
        'vocab_size': 30522,
    }

    config_object = BertConfig(**config_object,)
    return config_object


## TOKENIZER HELPERS ##
@retry(
    wait=wait_fixed(5),
    stop=stop_after_attempt(1),
)
def tiny_gpt2_tokenizer_helper(add_pad: bool = False):
    transformers = pytest.importorskip('transformers')

    hf_tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')

    if add_pad:
        hf_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return hf_tokenizer


@retry(
    wait=wait_fixed(5),
    stop=stop_after_attempt(1),
)
def tiny_llama_tokenizer_helper():
    transformers = pytest.importorskip('transformers')

    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(
        'huggyllama/llama-7b',
        use_fast=False,
    )
    return hf_tokenizer


@retry(
    wait=wait_fixed(5),
    stop=stop_after_attempt(1),
)
def tiny_codellama_tokenizer_helper():
    transformers = pytest.importorskip('transformers')

    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(
        'codellama/CodeLlama-7b-hf',
    )
    return hf_tokenizer


@retry(
    wait=wait_fixed(5),
    stop=stop_after_attempt(1),
)
def tiny_neox_tokenizer_helper():
    transformers = pytest.importorskip('transformers')

    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(
        'EleutherAI/gpt-neox-20b',
        model_max_length=2048,
    )
    return hf_tokenizer


@retry(
    wait=wait_fixed(5),
    stop=stop_after_attempt(1),
)
def tiny_t5_tokenizer_helper():
    transformers = pytest.importorskip('transformers')

    hf_tokenizer = transformers.AutoTokenizer.from_pretrained('t5-base',)
    return hf_tokenizer


@retry(
    wait=wait_fixed(5),
    stop=stop_after_attempt(1),
)
def tiny_bert_tokenizer_helper():
    transformers = pytest.importorskip('transformers')

    return transformers.AutoTokenizer.from_pretrained(
        'google-bert/bert-base-uncased',
    )


@retry(
    wait=wait_fixed(5),
    stop=stop_after_attempt(1),
)
def tiny_mpt_tokenizer_helper():
    transformers = pytest.importorskip('transformers')

    return transformers.AutoTokenizer.from_pretrained(
        'mosaicml/mpt-7b',
        model_max_length=2048,
    )


@retry(
    wait=wait_fixed(5),
    stop=stop_after_attempt(1),
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
def _session_tiny_bert_model(_session_tiny_bert_config):  # type: ignore
    return masked_lm_model_helper(_session_tiny_bert_config)


@pytest.fixture(scope='session')
def _session_tiny_codellama_model(  # type: ignore
    _session_tiny_codellama_config,  # type: ignore
):  # type: ignore
    return causal_lm_model_helper(_session_tiny_codellama_config)


## SESSION CONFIGS ##
@pytest.fixture(scope='session')
def _session_tiny_gpt2_config():  # type: ignore
    return tiny_gpt2_config_helper()


@pytest.fixture(scope='session')
def _session_tiny_codellama_config():  # type: ignore
    return tiny_codellama_config_helper()


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
def _session_tiny_neox_tokenizer():  # type: ignore
    return tiny_neox_tokenizer_helper()


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
def tiny_codellama_model(_session_tiny_codellama_model):  # type: ignore
    return copy.deepcopy(_session_tiny_codellama_model)


## CONFIG FIXTURES ##
@pytest.fixture
def tiny_bert_config(_session_tiny_bert_config):  # type: ignore
    return copy.deepcopy(_session_tiny_bert_config)


## TOKENIZER FIXTURES ##
@pytest.fixture
def tiny_gpt2_tokenizer(_session_tiny_gpt2_tokenizer):  # type: ignore
    return copy.deepcopy(_session_tiny_gpt2_tokenizer)


@pytest.fixture
def tiny_gpt2_with_pad_tokenizer(
    _session_tiny_gpt2_with_pad_tokenizer,  # type: ignore
):
    return copy.deepcopy(_session_tiny_gpt2_with_pad_tokenizer)


@pytest.fixture
def tiny_llama_tokenizer(_session_tiny_llama_tokenizer):  # type: ignore
    return copy.deepcopy(_session_tiny_llama_tokenizer)


@pytest.fixture
def tiny_codellama_tokenizer(_session_tiny_codellama_tokenizer):  # type: ignore
    return copy.deepcopy(_session_tiny_codellama_tokenizer)


@pytest.fixture
def tiny_neox_tokenizer(_session_tiny_neox_tokenizer):  # type: ignore
    return copy.deepcopy(_session_tiny_neox_tokenizer)


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
        'EleutherAI/gpt-neox-20b': 'tiny_neox_tokenizer',
        't5-base': 'tiny_t5_tokenizer',
        'google-bert/bert-base-uncased': 'tiny_bert_tokenizer',
        'mosaicml/mpt-7b': 'tiny_mpt_tokenizer',
        'mosaicml/mpt-7b-8k-chat': 'tiny_mpt_chat_tokenizer',
    }
    return request.getfixturevalue(name_to_fixture_name[name])
