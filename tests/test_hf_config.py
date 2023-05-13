# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Mapping

import pytest
import torch
from composer.utils import reproducibility
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from transformers import AutoConfig, AutoModelForCausalLM

from llmfoundry import COMPOSER_MODEL_REGISTRY
from llmfoundry.models.mpt import MPTConfig, MPTForCausalLM
from llmfoundry.utils import build_tokenizer


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
    model_cfg_overrides,
    conf_path='scripts/train/yamls/pretrain/testing.yaml',
):
    AutoConfig.register('mpt', MPTConfig)
    AutoModelForCausalLM.register(MPTConfig, MPTForCausalLM)

    with open(conf_path) as f:
        test_cfg = om.load(f)

    reproducibility.seed_all(test_cfg.seed)

    # Build Model
    # For fast initialization, use `meta` device
    print('Initializing model...')
    device = 'cpu'
    test_cfg.model.init_device = device
    test_cfg.device = device
    test_cfg.precision = 'fp16'
    test_cfg.model.attn_config = {'attn_impl': 'torch', 'alibi': True}

    tokenizer = build_tokenizer(test_cfg.tokenizer)
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
    model_cfg = {
        'name': 'hf_causal_lm',
        'pretrained_model_name_or_path': save_path,
        'pretrained': False,
        'config_overrides': model_cfg_overrides,
    }
    hf_model_config.model = model_cfg

    hf_model = COMPOSER_MODEL_REGISTRY[hf_model_config.model.name](
        hf_model_config.model, tokenizer=tokenizer)

    for k, v in hf_model_config.model.config_overrides.items():
        if isinstance(v, Mapping):
            for _k, _v in v.items():
                assert getattr(hf_model.config, k)[_k] == _v
        else:
            assert getattr(hf_model.config, k) == v
