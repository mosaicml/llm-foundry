# Copyright 2022-2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping

import pytest
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from transformers import AutoModelForCausalLM

from llmfoundry import COMPOSER_MODEL_REGISTRY
from llmfoundry.models.mpt import MPTConfig, MPTForCausalLM
from llmfoundry.utils import build_tokenizer


def test_remote_code_false_mpt(
        conf_path: str = 'scripts/train/yamls/finetune/mpt-7b_dolly_sft.yaml'):
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

    tokenizer_cfg: Dict[str,
                        Any] = om.to_container(test_cfg.tokenizer,
                                               resolve=True)  # type: ignore
    tokenizer_name = tokenizer_cfg['name']
    tokenizer_kwargs = tokenizer_cfg.get('kwargs', {})
    tokenizer = build_tokenizer(tokenizer_name, tokenizer_kwargs)

    with pytest.raises(
            ValueError,
            match='trust_remote_code must be set to True for MPT models.'):
        _ = COMPOSER_MODEL_REGISTRY[test_cfg.model.name](test_cfg.model,
                                                         tokenizer)


@pytest.mark.parametrize('model_cfg_overrides', [
    {
        'max_seq_len': 1024
    },
    {
        'attn_config': {
            'attn_impl': 'triton'
        }
    },
    {
        'init_config': {
            'emb_init_std': 5
        }
    },
    {
        'max_seq_len': 1024,
        'attn_config': {
            'attn_impl': 'triton'
        },
        'init_config': {
            'emb_init_std': 5
        },
    },
    pytest.param({'msl': 1024},
                 marks=pytest.mark.xfail(reason='"msl" is a ValueError',
                                         strict=True)),
    pytest.param({'attn_config': {
        'attn_iml': 'triton'
    }},
                 marks=pytest.mark.xfail(reason='"attn_impl" mispelled',
                                         strict=True)),
])
def test_hf_config_override(
    model_cfg_overrides: Dict[str, Any],
    conf_path: str = 'scripts/train/yamls/pretrain/testing.yaml',
):
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    CONFIG_MAPPING._extra_content['mpt'] = MPTConfig
    AutoModelForCausalLM.register(MPTConfig, MPTForCausalLM)

    with open(conf_path) as f:
        test_cfg = om.load(f)

    # Build Model
    # For fast initialization, use `meta` device
    print('Initializing model...')
    device = 'cpu'
    test_cfg.model.init_device = device
    test_cfg.device = device
    test_cfg.precision = 'fp16'
    test_cfg.model.attn_config = {'attn_impl': 'torch', 'alibi': True}

    tokenizer_cfg: Dict[str,
                        Any] = om.to_container(test_cfg.tokenizer,
                                               resolve=True)  # type: ignore
    tokenizer_name = tokenizer_cfg['name']
    tokenizer_kwargs = tokenizer_cfg.get('kwargs', {})
    tokenizer = build_tokenizer(tokenizer_name, tokenizer_kwargs)
    model = COMPOSER_MODEL_REGISTRY[test_cfg.model.name](test_cfg.model,
                                                         tokenizer)

    # save model
    tmp_dir = tempfile.TemporaryDirectory()
    save_path = tmp_dir.name

    tokenizer.save_pretrained(save_path)
    model.config.save_pretrained(save_path)
    torch.save(model.state_dict(), Path(save_path) / 'pytorch_model.bin')

    # load hf causal lm model with config_overrides
    hf_model_config = deepcopy(test_cfg)
    model_cfg = DictConfig({
        'name': 'hf_causal_lm',
        'pretrained_model_name_or_path': save_path,
        'pretrained': False,
        'config_overrides': model_cfg_overrides,
    })
    hf_model_config.model = model_cfg

    hf_model = COMPOSER_MODEL_REGISTRY[hf_model_config.model.name](
        hf_model_config.model, tokenizer=tokenizer)

    for k, v in hf_model_config.model.config_overrides.items():
        if isinstance(v, Mapping):
            for _k, _v in v.items():
                assert getattr(hf_model.config, k)[_k] == _v
        else:
            assert getattr(hf_model.config, k) == v


@pytest.mark.skipif('HUGGING_FACE_HUB_TOKEN' not in os.environ,
                    reason='CI does not have access to llama2')
def test_rope_scaling_override():
    model_cfg = {
        'name': 'hf_causal_lm',
        'pretrained_model_name_or_path': 'meta-llama/Llama-2-7b-hf',
        'config_overrides': {
            'num_hidden_layers': 2,
            'hidden_size': 32,
            'intermediate_size': 64,
            'rope_scaling': {
                'type': 'dynamic',
                'factor': 0.5
            }
        },
        'use_auth_token': True,
        'pretrained': False,
        'init_device': 'cpu',
    }
    model_cfg = om.create(model_cfg)

    model = COMPOSER_MODEL_REGISTRY[model_cfg.name](model_cfg, tokenizer=None)
    # This would error if the config isn't parsed into a proper dictionary
    model.get_metadata()
    assert model.config.rope_scaling == {'type': 'dynamic', 'factor': 0.5}
