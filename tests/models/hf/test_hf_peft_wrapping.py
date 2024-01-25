# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import transformers
from peft import LoraConfig, get_peft_model

from llmfoundry.models.hf.hf_fsdp import prepare_hf_model_for_fsdp
from typing import Optional
import torch

from omegaconf import OmegaConf as om
import os
from composer.utils import dist
import pathlib
from unittest.mock import patch

from tests.data_utils import make_tiny_ft_dataset


import pytest
from composer import Trainer
from omegaconf import OmegaConf as om

from llmfoundry import COMPOSER_MODEL_REGISTRY
from llmfoundry.data.finetuning import build_finetuning_dataloader
from llmfoundry.utils.builders import build_optimizer, build_tokenizer


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
@pytest.mark.parametrize('peft_config', [
    {
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
    }
])
@pytest.mark.parametrize('init_device', ['mixed'])
@patch('torch.nn.init.kaiming_uniform_', lambda w, a: torch.nn.init.ones_(w))
def test_lora_mixed_init(peft_config: Optional[dict], tmp_path: pathlib.Path, init_device: str):
    model_cfg = {
        'name': 'hf_causal_lm',
        'pretrained_model_name_or_path': 'mistralai/Mistral-7B-v0.1',
        'config_overrides': {
            'num_hidden_layers': 2,
            'hidden_size': 32,
            'intermediate_size': 64,
        },
        'use_auth_token': True,
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

    tiny_dataset_folder_path = os.path.join(os.getcwd(), 'test-ift-data-small')
    tiny_dataset_path = os.path.join(tiny_dataset_folder_path, 'train.jsonl')
    if dist.get_global_rank() == 0:
        make_tiny_ft_dataset(path=tiny_dataset_path, size=4)

    dataloader_cfg = {
        'name': 'finetuning',
        'dataset': {
            'hf_name': tiny_dataset_folder_path,
            'split': 'train',
            'max_seq_len': 32,
            'decoder_only_format': True,
            'allow_pad_trimming': False,
            'packing_ratio': None,
            'shuffle': True,
        },
        'drop_last': False,
        'num_workers': 0,
        'pin_memory': False,
        'prefetch_factor': None,
        'persistent_workers': False,
        'timeout': 0
    }

    dataloader_cfg = om.create(dataloader_cfg)

    tokenizer = build_tokenizer(
        tokenizer_name=tokenizer_name,
        tokenizer_kwargs={'model_max_length': 32},
    )

    train_dataloader = build_finetuning_dataloader(
        dataloader_cfg,
        tokenizer,
        1,
    )

    original_model = COMPOSER_MODEL_REGISTRY[model_cfg['name']](model_cfg,
                                                                tokenizer)

    optimizer_config = {
        'name': 'decoupled_adamw',
        'lr': 6e-4,
        'betas': [0.9, 0.95],
        'eps': 1e-8,
        'weight_decay': 0.0,
    }
    optimizer_name = optimizer_config.pop('name')
    optimizer = build_optimizer(original_model, optimizer_name,
                                optimizer_config)

    trainer = Trainer(
        model=original_model,
        device='gpu',
        fsdp_config=fsdp_config,
        train_dataloader=train_dataloader,
        save_folder=os.path.join(tmp_path, 'checkpoints'),
        save_interval='1ba',
        max_duration='1ba',
        optimizers=optimizer,
        save_latest_filename=None,
    )

    model = trainer.state.model
    lora_A = model.model.base_model.model.model.layers[0].self_attn.q_proj.lora_A['default']
    lora_B = model.model.base_model.model.model.layers[0].self_attn.q_proj.lora_B['default']

    assert (lora_A.weight == 1).all()
    assert (lora_B.weight == 0).all()