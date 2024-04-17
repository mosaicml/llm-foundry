# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import pathlib
from typing import Optional
from unittest.mock import patch

import pytest
import torch
import transformers
from composer import Trainer
from omegaconf import OmegaConf as om
from peft import LoraConfig, get_peft_model

from llmfoundry.models.hf.hf_fsdp import prepare_hf_model_for_fsdp
from llmfoundry.utils.builders import build_composer_model, build_tokenizer


def test_peft_wraps():
    mistral_cfg = transformers.AutoConfig.from_pretrained(
        'mistralai/Mistral-7B-v0.1', num_hidden_layers=2)
    mistral = transformers.AutoModelForCausalLM.from_config(mistral_cfg)
    mistral = get_peft_model(mistral, LoraConfig())
    prepare_hf_model_for_fsdp(mistral, 'cpu')

    for n, m in mistral.named_modules():
        if 'lora' in n and 'default' in n:
            has_parameters = any(True for _ in m.parameters())
            has_buffers = any(True for _ in m.buffers())
            if has_parameters or has_buffers:
                assert m._fsdp_wrap


@pytest.mark.world_size(2)
@pytest.mark.gpu
@pytest.mark.parametrize('peft_config', [{
    'peft_type': 'LORA',
    'task_type': 'CAUSAL_LM',
    'lora_alpha': 32,
    'lora_dropout': 0.05,
    'r': 16,
    'target_modules': [
        'q_proj',
        'k_proj',
        'v_proj',
    ],
}])
@pytest.mark.parametrize('init_device', ['mixed'])
@patch('torch.nn.init.kaiming_uniform_', lambda w, a: torch.nn.init.ones_(w))
def test_lora_mixed_init(peft_config: Optional[dict], tmp_path: pathlib.Path,
                         init_device: str):
    model_cfg = {
        'name': 'hf_causal_lm',
        'pretrained_model_name_or_path': 'mistralai/Mistral-7B-v0.1',
        'config_overrides': {
            'num_hidden_layers': 2,
            'hidden_size': 32,
            'intermediate_size': 64,
        },
        'pretrained': False,
        'init_device': init_device,
    }
    tokenizer_name = 'mistralai/Mistral-7B-v0.1'

    assert model_cfg is not None
    assert tokenizer_name is not None
    model_cfg = om.create(model_cfg)
    model_cfg['peft_config'] = peft_config

    fsdp_config = {
        'sharding_strategy': 'FULL_SHARD',
        'mixed_precision': 'PURE',
        'activation_checkpointing': False,
        'activation_checkpointing_reentrant': False,
        'activation_cpu_offload': False,
        'limit_all_gathers': True,
        'state_dict_type': 'full',
        'sync_module_states': True,
    }

    tokenizer = build_tokenizer(
        tokenizer_name=tokenizer_name,
        tokenizer_kwargs={'model_max_length': 32},
    )

    original_model = build_composer_model(
        composer_model_name=model_cfg['name'],
        tokenizer=tokenizer,
        cfg=model_cfg,
    )

    trainer = Trainer(
        model=original_model,
        device='gpu',
        fsdp_config=fsdp_config,
        train_dataloader=[],
        device_train_microbatch_size=1,
    )

    model = trainer.state.model
    underlying_model = model.model.base_model.model
    lora_A = underlying_model.model.layers[0].self_attn.q_proj.lora_A['default']
    lora_B = underlying_model.model.layers[0].self_attn.q_proj.lora_B['default']

    assert (lora_A.weight == 1).all()
    assert (lora_B.weight == 0).all()
