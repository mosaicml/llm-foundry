# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping
from unittest.mock import Mock, patch

import pytest
import torch
from omegaconf import OmegaConf as om
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.models.llama4.configuration_llama4 import Llama4TextConfig

from llmfoundry.models.hf import BaseHuggingFaceModel
from llmfoundry.models.hf.hf_fsdp import rgetattr
from llmfoundry.models.mpt import MPTConfig, MPTForCausalLM
from llmfoundry.utils.builders import build_composer_model
from llmfoundry.utils.config_utils import (
    to_dict_container,
)


@pytest.fixture
def tiny_llama_save_dir(tmp_path: Path, tiny_codellama_model: PreTrainedModel):
    assert isinstance(tiny_codellama_model, PreTrainedModel)
    assert tiny_codellama_model.generation_config is not None

    save_dir = tmp_path / 'model'

    # Save the model.
    tiny_codellama_model.save_pretrained(save_dir)

    return save_dir


def test_remote_code_false_mpt(
    tiny_mpt_tokenizer: PreTrainedTokenizerBase,
    conf_path: str = 'scripts/train/yamls/finetune/mpt-7b_dolly_sft.yaml',
):
    with open(conf_path) as f:
        test_cfg = om.load(f)

    test_cfg.model.pretrained = False
    test_cfg.model.config_overrides = {'n_layers': 2}
    test_cfg.model.trust_remote_code = False

    # Build Model
    # For fast initialization, use `meta` device
    print('Initializing model...')
    device = 'cpu'
    test_cfg.model.init_device = device
    test_cfg.device = device
    test_cfg.precision = 'fp16'

    tokenizer = tiny_mpt_tokenizer

    with pytest.raises(
        ValueError,
        match=
        'The MPT series of models on the Hugging Face Hub is no longer supported by LLM Foundry',
    ):
        name = test_cfg.model.pop('name')
        _ = build_composer_model(
            name=name,
            cfg=to_dict_container(test_cfg.model),
            tokenizer=tokenizer,
        )


@pytest.mark.parametrize('tie_word_embeddings', [True, False])
def test_tie_weights(tie_word_embeddings: bool):
    # Test that the tie_weights function sets lm_head correctly
    hf_config = MPTConfig(
        init_device='cpu',
        d_model=128,
        n_heads=4,
        n_layers=2,
        expansion_ratio=2,
        max_seq_len=2048,
        attn_config={
            'attn_impl': 'torch',
        },
        no_bias=True,
        tie_word_embeddings=tie_word_embeddings,
    )

    mpt = MPTForCausalLM(hf_config)

    assert mpt.config.tie_word_embeddings == tie_word_embeddings
    mpt.tie_weights()
    if tie_word_embeddings:
        assert mpt.lm_head is None
    else:
        assert mpt.lm_head is not None


@pytest.mark.parametrize(
    'model_cfg_overrides',
    [
        {
            'max_position_embeddings': 2048,
        },
        {
            'attention_dropout': 0.1,
        },
        {
            'initializer_range': 0.02,
        },
        {
            'max_position_embeddings': 2048,
            'attention_dropout': 0.1,
            'initializer_range': 0.02,
        },
        pytest.param({'msl': 1024},
                     marks=pytest.mark.xfail(
                         reason='"msl" is a ValueError',
                         strict=True,
                     )),
    ],
)
@patch(
    'llmfoundry.models.layers.attention.is_flash_v2_installed',
    new=Mock(return_value=True),
)
def test_hf_config_override(
    model_cfg_overrides: dict[str, Any],
    tiny_codellama_tokenizer: PreTrainedTokenizerBase,
    conf_path: str = 'scripts/train/yamls/pretrain/testing.yaml',
):
    with open(conf_path) as f:
        test_cfg = om.load(f)

    tokenizer = tiny_codellama_tokenizer

    tiny_overrides = {
        'num_hidden_layers': 2,
        'hidden_size': 128,
        'intermediate_size': 256,  # Added for CodeLlama
    }

    model_cfg_overrides.update(tiny_overrides)

    # load hf causal lm model with config_overrides
    hf_model_config = deepcopy(test_cfg)
    model_cfg = om.create({
        'name': 'hf_causal_lm',
        'pretrained_model_name_or_path': 'codellama/CodeLlama-7b-hf',
        'pretrained': False,
        'config_overrides': model_cfg_overrides,
    })
    hf_model_config.model = model_cfg

    name = hf_model_config.model.pop('name')
    hf_model = build_composer_model(
        name=name,
        cfg=to_dict_container(hf_model_config.model),
        tokenizer=tokenizer,
    )

    for k, v in hf_model_config.model.config_overrides.items():
        if isinstance(v, Mapping):
            for _k, _v in v.items():
                assert getattr(hf_model.config, k)[_k] == _v
        else:
            assert getattr(hf_model.config, k) == v


@pytest.mark.skipif(
    'HF_TOKEN' not in os.environ,
    reason='CI does not have access to llama2',
)
def test_rope_scaling_override():
    model_cfg = {
        'name': 'hf_causal_lm',
        'pretrained_model_name_or_path': 'meta-llama/Meta-Llama-3-8B',
        'config_overrides': {
            'num_hidden_layers': 2,
            'hidden_size': 32,
            'intermediate_size': 64,
            'rope_scaling': {
                'type': 'dynamic',
                'factor': 0.5,
            },
        },
        'use_auth_token': True,
        'pretrained': False,
        'init_device': 'cpu',
    }

    name = model_cfg.pop('name')
    model = build_composer_model(
        name=name,
        cfg=model_cfg,
        tokenizer=None,  # type: ignore
    )
    # This would error if the config isn't parsed into a proper dictionary
    model.get_metadata()  # type: ignore
    assert model.config.rope_scaling == {  # type: ignore
        'type': 'dynamic',
        'factor': 0.5,
    }


@pytest.mark.skipif(
    'HF_TOKEN' not in os.environ,
    reason='CI does not have access to Dbrx',
)
def test_nested_override():
    model_cfg = {
        'name': 'hf_causal_lm',
        'pretrained_model_name_or_path': 'databricks/dbrx-instruct',
        'config_overrides': {
            'ffn_config': {
                'ffn_hidden_size': 500,
            },
        },
        'use_auth_token': True,
        'pretrained': False,
        'init_device': 'meta',
    }

    name = model_cfg.pop('name')
    model = build_composer_model(
        name=name,
        cfg=model_cfg,
        tokenizer=None,  # type: ignore
    )

    # The value we changed
    assert model.config.ffn_config.ffn_hidden_size == 500  # type: ignore
    # Ensure we still have a config, and haven't replaced it with a dictionary
    assert isinstance(model.config.ffn_config, PretrainedConfig)  # type: ignore
    # Ensure the other values still exist and are not set back to their defaults
    assert model.config.ffn_config.moe_num_experts == 16  # type: ignore


def test_simple_dtype():
    model_cfg = {
        'name': 'hf_causal_lm',
        'pretrained_model_name_or_path': 'codellama/CodeLlama-7b-hf',
        'config_overrides': {
            'num_hidden_layers': 2,
            'hidden_size': 32,
            'intermediate_size': 64,
        },
        'pretrained': False,
        'init_device': 'cpu',
        'attn_implementation': 'eager',
    }

    name = model_cfg.pop('name')
    model = build_composer_model(
        name=name,
        cfg=model_cfg,
        tokenizer=None,  # type: ignore
    )

    # Make sure that HF has not cast the parameters to bf16
    assert next(model.parameters()).dtype == torch.float32


@pytest.mark.gpu
def test_use_flash():
    model_cfg = {
        'name': 'hf_causal_lm',
        'pretrained_model_name_or_path': 'codellama/CodeLlama-7b-hf',
        'config_overrides': {
            'num_hidden_layers': 2,
            'hidden_size': 32,
            'intermediate_size': 64,
            'torch_dtype': 'bfloat16',
        },
        'pretrained': False,
        'init_device': 'cpu',
        'attn_implementation': 'flash_attention_2',
    }

    name = model_cfg.pop('name')
    model = build_composer_model(
        name=name,
        cfg=model_cfg,
        tokenizer=None,  # type: ignore
    )

    from transformers.models.llama.modeling_llama import (
        LlamaAttention,
    )
    flash_attn_class = LlamaAttention
    attention_layers_attr = 'model.model.layers'
    attention_attr = 'self_attn'

    # check that it actually used flash attention 2
    assert model.model.config._attn_implementation == (  # type: ignore
        'flash_attention_2'
    )
    attention_layer = rgetattr(
        rgetattr(model, attention_layers_attr)[0],
        attention_attr,
    )
    assert isinstance(attention_layer, flash_attn_class)
    assert next(model.parameters()).dtype == torch.bfloat16


def test_generation_config(
    tmp_path: Path,
    tiny_codellama_model: PreTrainedModel,
):
    assert isinstance(tiny_codellama_model, PreTrainedModel)
    assert tiny_codellama_model.generation_config is not None

    new_bos_token_id = 100

    # Set the bos_token_id to something else
    tiny_codellama_model.generation_config.bos_token_id = new_bos_token_id

    # Generation config and model config no longer match
    assert tiny_codellama_model.generation_config.bos_token_id != tiny_codellama_model.config.bos_token_id

    save_dir = tmp_path / 'model'

    # Save the model.
    tiny_codellama_model.save_pretrained(save_dir)

    # Now load the model from the save directory and check that the bos_token_id is the same as what we set.
    model_cfg = {
        'name': 'hf_causal_lm',
        'pretrained_model_name_or_path': str(save_dir),
        'use_auth_token': True,
        'pretrained': False,
        'init_device': 'cpu',
    }

    name = model_cfg.pop('name')
    model = build_composer_model(
        name=name,
        cfg=model_cfg,
        tokenizer=None,  # type: ignore
    )

    inner_model = model.model

    assert isinstance(inner_model, PreTrainedModel)
    assert inner_model.generation_config is not None  # type: ignore

    # save_pretrained and reloading with hf_causal_lm should use the bos_token_id we set from earlier.
    assert inner_model.generation_config.bos_token_id == new_bos_token_id  # type: ignore


@pytest.mark.gpu
@pytest.mark.parametrize(
    'attn_implementation',
    ['eager', 'flash_attention_2', 'sdpa'],
)
def test_attn_implementation(
    attn_implementation: str,
    tiny_llama_save_dir: Path,
):
    model_cfg = {
        'name': 'hf_causal_lm',
        'pretrained_model_name_or_path': str(tiny_llama_save_dir),
        'config_overrides': {
            'num_hidden_layers': 2,
            'hidden_size': 32,
            'intermediate_size': 64,
            'torch_dtype': 'bfloat16',
        },
        'pretrained': False,
        'attn_implementation': attn_implementation,
    }

    name = model_cfg.pop('name')
    model = build_composer_model(
        name=name,
        cfg=model_cfg,
        tokenizer=None,  # type: ignore
    )

    # llama config uses _attn_implementation
    assert model.config._attn_implementation == attn_implementation  # type: ignore


@pytest.mark.gpu
def test_attn_implementation_override(tiny_llama_save_dir: Path):
    model_cfg = {
        'name': 'hf_causal_lm',
        'pretrained_model_name_or_path': str(tiny_llama_save_dir),
        'config_overrides': {
            'num_hidden_layers': 2,
            'hidden_size': 32,
            'intermediate_size': 64,
            'torch_dtype': 'bfloat16',
        },
        'pretrained': False,
        'use_flash_attention_2': True,
        'attn_implementation': 'eager',
    }

    name = model_cfg.pop('name')
    model = build_composer_model(
        name=name,
        cfg=model_cfg,
        tokenizer=None,  # type: ignore
    )

    # llama config uses _attn_implementation
    assert model.config._attn_implementation == 'flash_attention_2'  # type: ignore


@pytest.mark.gpu
def test_attn_implementation_none(tiny_llama_save_dir: Path):
    model_cfg = {
        'name': 'hf_causal_lm',
        'pretrained_model_name_or_path': str(tiny_llama_save_dir),
        'config_overrides': {
            'num_hidden_layers': 2,
            'hidden_size': 32,
            'intermediate_size': 64,
            'torch_dtype': 'bfloat16',
        },
        'pretrained': False,
        'use_flash_attention_2': False,
    }

    name = model_cfg.pop('name')
    model = build_composer_model(
        name=name,
        cfg=model_cfg,
        tokenizer=None,  # type: ignore
    )

    # llama config uses _attn_implementation
    assert model.config._attn_implementation == 'eager'  # type: ignore


def test_text_config(tiny_llama4_config: PretrainedConfig, tmp_path: Path):
    save_path = tmp_path / 'model'
    tiny_llama4_config.save_pretrained(save_path)

    class TestModel(BaseHuggingFaceModel):
        subselect_config_attr = 'text_config'

    config = TestModel.build_config(
        pretrained_model_name_or_path=str(save_path),
        trust_remote_code=True,
        use_auth_token=False,
        attn_implementation='eager',
        config_overrides={},
    )

    assert isinstance(config, Llama4TextConfig)
    assert config.attn_implementation == 'eager'
    assert config.use_cache == False
    assert config.torch_dtype == 'float32'
