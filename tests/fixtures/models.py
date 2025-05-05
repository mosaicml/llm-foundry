# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import os
import zipfile
from typing import Any, Callable

import pytest
import requests
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

    config_object = GPT2Config(
        **config_dict,
    )
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

    config_object = LlamaConfig(
        **config_dict,
    )
    return config_object


def tiny_llama4_config_helper():
    pytest.importorskip('transformers')
    from transformers.models.llama4.configuration_llama4 import Llama4Config

    config_dict = {
        'architectures': ['Llama4ForConditionalGeneration',],
        'boi_token_index': 200080,
        'eoi_token_index': 200081,
        'image_token_index': 200092,
        'model_type': 'llama4',
        'text_config': {
            '_attn_implementation_autoset': True,
            'attention_bias': False,
            'attention_chunk_size': 8192,
            'attention_dropout': 0.0,
            'bos_token_id': 200000,
            'eos_token_id': [
                200001,
                200007,
                200008,
            ],
            'for_llm_compressor': False,
            'head_dim': 128,
            'hidden_act': 'silu',
            'hidden_size': 5120,
            'initializer_range': 0.02,
            'interleave_moe_layer_step': 1,
            'intermediate_size': 8192,
            'intermediate_size_mlp': 16384,
            'max_position_embeddings': 10485760,
            'model_type': 'llama4_text',
            'no_rope_layers': [],
            'num_attention_heads': 40,
            'num_experts_per_tok': 1,
            'num_hidden_layers': 48,
            'num_key_value_heads': 8,
            'num_local_experts': 16,
            'output_router_logits': False,
            'pad_token_id': 200018,
            'rms_norm_eps': 1e-05,
            'rope_scaling': {
                'factor': 16.0,
                'high_freq_factor': 1.0,
                'low_freq_factor': 1.0,
                'original_max_position_embeddings': 8192,
                'rope_type': 'llama3',
            },
            'rope_theta': 500000.0,
            'router_aux_loss_coef': 0.001,
            'router_jitter_noise': 0.0,
            'torch_dtype': 'bfloat16',
            'use_cache': True,
            'use_qk_norm': True,
            'vocab_size': 202048,
        },
        'torch_dtype': 'bfloat16',
        'transformers_version': '4.51.0.dev0',
        'vision_config': {
            '_attn_implementation_autoset': True,
            'attention_dropout': 0.0,
            'hidden_act': 'gelu',
            'hidden_size': 1408,
            'image_size': 336,
            'initializer_range': 0.02,
            'intermediate_size': 5632,
            'model_type': 'llama4_vision_model',
            'multi_modal_projector_bias': False,
            'norm_eps': 1e-05,
            'num_attention_heads': 16,
            'num_channels': 3,
            'num_hidden_layers': 34,
            'patch_size': 14,
            'pixel_shuffle_ratio': 0.5,
            'projector_dropout': 0.0,
            'projector_input_dim': 4096,
            'projector_output_dim': 4096,
            'rope_theta': 10000,
            'vision_feature_layer': -1,
            'vision_feature_select_strategy': 'default',
            'vision_output_dim': 4096,
        },
    }

    config_object = Llama4Config(**config_dict)
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

    config_object = BertConfig(
        **config_object,
    )
    return config_object


def assets_path():
    rank = os.environ.get('RANK', '0')
    folder_name = 'tokenizers' + (f'_{rank}' if rank != '0' else '')
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'assets',
        folder_name,
    )


@pytest.fixture(scope='session')
def tokenizers_assets():
    download_tokenizers_files()


def download_tokenizers_files():
    """Download the tokenizers assets.

    We download from github, because downloading from HF directly is flaky and gets rate limited easily.

    Raises:
        ValueError: If the checksum of the downloaded file does not match the expected checksum.
    """
    # Define paths
    tokenizers_dir = assets_path()

    if os.path.exists(tokenizers_dir):
        return

    # Create assets directory if it doesn't exist
    os.makedirs(tokenizers_dir, exist_ok=True)

    # URL for the tokenizers.zip file
    url = 'https://github.com/mosaicml/ci-testing/releases/download/tokenizers/tokenizers.zip'
    expected_checksum = '12dc1f254270582f7806588f1f1d47945590c5b42dee28925e5dab95f2d08075'

    # Download the zip file
    response = requests.get(url, stream=True)
    response.raise_for_status()

    zip_path = os.path.join(tokenizers_dir, 'tokenizers.zip')

    # Check the checksum
    checksum = hashlib.sha256(response.content).hexdigest()
    if checksum != expected_checksum:
        raise ValueError(
            f'Checksum mismatch: expected {expected_checksum}, got {checksum}',
        )

    with open(zip_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    # Extract the zip file
    print(f'Extracting tokenizers.zip to {tokenizers_dir}')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(tokenizers_dir)

    # Optionally remove the zip file after extraction
    os.remove(zip_path)


## TOKENIZER HELPERS ##
def assets_tokenizer_helper(name: str, **kwargs: Any):
    """Load a tokenizer from the assets directory."""
    transformers = pytest.importorskip('transformers')

    download_tokenizers_files()

    assets_dir = assets_path()
    tokenizer_path = os.path.join(assets_dir, name)

    # Load the tokenizer
    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_path,
        **kwargs,
    )
    return hf_tokenizer


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


@pytest.fixture(scope='session')
def _session_tiny_llama4_config():  # type: ignore
    return tiny_llama4_config_helper()


## SESSION TOKENIZERS ##
@pytest.fixture(scope='session')
def _session_tiny_gpt2_tokenizer(tokenizers_assets):  # type: ignore
    return assets_tokenizer_helper('gpt2')


@pytest.fixture(scope='session')
def _session_tiny_gpt2_with_pad_tokenizer(tokenizers_assets):  # type: ignore
    tokenizer = assets_tokenizer_helper('gpt2')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer


@pytest.fixture(scope='session')
def _session_tiny_llama_tokenizer(tokenizers_assets):  # type: ignore
    return assets_tokenizer_helper('llama')


@pytest.fixture(scope='session')
def _session_tiny_slow_llama_tokenizer(tokenizers_assets):  # type: ignore
    return assets_tokenizer_helper('llama', use_fast=False)


@pytest.fixture(scope='session')
def _session_tiny_codellama_tokenizer(tokenizers_assets):  # type: ignore
    return assets_tokenizer_helper('codellama')


@pytest.fixture(scope='session')
def _session_tiny_neox_tokenizer(tokenizers_assets):  # type: ignore
    return assets_tokenizer_helper('neox')


@pytest.fixture(scope='session')
def _session_tiny_t5_tokenizer(tokenizers_assets):  # type: ignore
    return assets_tokenizer_helper('t5')


@pytest.fixture(scope='session')
def _session_tiny_bert_tokenizer(tokenizers_assets):  # type: ignore
    return assets_tokenizer_helper('bertt')


@pytest.fixture(scope='session')
def _session_tiny_mpt_tokenizer(tokenizers_assets):  # type: ignore
    return assets_tokenizer_helper('mptt')


@pytest.fixture(scope='session')
def _session_tiny_mpt_chat_tokenizer(tokenizers_assets):  # type: ignore
    return assets_tokenizer_helper('mptct')


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


@pytest.fixture
def tiny_llama4_config(_session_tiny_llama4_config):  # type: ignore
    return copy.deepcopy(_session_tiny_llama4_config)


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
def tiny_slow_llama_tokenizer(
    _session_tiny_slow_llama_tokenizer,  # type: ignore
):
    return copy.deepcopy(_session_tiny_slow_llama_tokenizer)


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
