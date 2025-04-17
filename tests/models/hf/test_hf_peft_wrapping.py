# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import pathlib
from typing import Optional
from unittest.mock import patch

import pytest
import torch
from composer import Trainer
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from llmfoundry.models.hf.hf_causal_lm import ComposerHFCausalLM
from llmfoundry.models.hf.hf_fsdp import prepare_hf_model_for_fsdp
from llmfoundry.utils.builders import build_composer_model


def test_peft_wraps(tiny_codellama_model: PreTrainedModel):
    codellama = get_peft_model(tiny_codellama_model, LoraConfig())
    assert isinstance(codellama, PeftModel)
    prepare_hf_model_for_fsdp(codellama, 'cpu')

    for n, m in codellama.named_modules():
        if 'lora' in n and 'default' in n:
            has_parameters = any(True for _ in m.parameters())
            has_buffers = any(True for _ in m.buffers())
            if has_parameters or has_buffers:
                assert m._fsdp_wrap


def test_causal_lm_peft_wraps():
    model = ComposerHFCausalLM(
        tokenizer=None,  # type: ignore
        pretrained_model_name_or_path='codellama/CodeLlama-7b-hf',
        pretrained=False,
        config_overrides={'num_hidden_layers': 2},
        peft_config={
            'peft_type': 'LORA',
            'task_type': 'CAUSAL_LM',
        },
    )

    for n, m in model.named_modules():
        if 'lora' in n and 'default' in n:
            has_parameters = any(True for _ in m.parameters())
            has_buffers = any(True for _ in m.buffers())
            if has_parameters or has_buffers:
                assert m._fsdp_wrap


@pytest.mark.world_size(2)
@pytest.mark.gpu
@pytest.mark.parametrize(
    'peft_config',
    [{
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
    }],
)
@pytest.mark.parametrize('init_device', ['mixed'])
@patch('torch.nn.init.kaiming_uniform_', lambda w, a: torch.nn.init.ones_(w))
def test_lora_mixed_init(
    peft_config: Optional[dict],
    tmp_path: pathlib.Path,
    init_device: str,
    tiny_codellama_tokenizer: PreTrainedTokenizerBase,
):
    model_cfg = {
        'name': 'hf_causal_lm',
        'pretrained_model_name_or_path': 'codellama/CodeLlama-7b-hf',
        'config_overrides': {
            'num_hidden_layers': 2,
            'hidden_size': 32,
            'intermediate_size': 64,
        },
        'pretrained': False,
        'init_device': init_device,
    }
    tokenizer_name = 'codellama/CodeLlama-7b-hf'

    assert model_cfg is not None
    assert tokenizer_name is not None
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

    tokenizer = tiny_codellama_tokenizer

    name = model_cfg.pop('name')
    original_model = build_composer_model(
        name=name,
        cfg=model_cfg,
        tokenizer=tokenizer,
    )

    trainer = Trainer(
        model=original_model,
        device='gpu',
        parallelism_config={'fsdp': fsdp_config},
        train_dataloader=[],
        device_train_microbatch_size=1,
    )

    model = trainer.state.model
    underlying_model = model.model.base_model.model  # type: ignore
    lora_A = underlying_model.model.layers[  # type: ignore
        0].self_attn.q_proj.lora_A[  # type: ignore
            'default']
    lora_B = underlying_model.model.layers[  # type: ignore
        0].self_attn.q_proj.lora_B[  # type: ignore
            'default']

    assert (lora_A.weight == 1).all()
    assert (lora_B.weight == 0).all()
