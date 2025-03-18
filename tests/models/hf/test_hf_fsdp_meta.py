# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from composer import Trainer

from llmfoundry.models.hf.hf_causal_lm import ComposerHFCausalLM
from llmfoundry.utils.builders import build_tokenizer


@pytest.mark.gpu
@pytest.mark.world_size(2)
def test_hf_meta_init_fsdp():
    # Use deterministic mode to ensure uninitialized weights are filled with NaNs
    torch.use_deterministic_algorithms(False)
    torch.utils.deterministic.fill_uninitialized_memory = True

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

    tokenizer_name = 'codellama/CodeLlama-7b-hf'
    tokenizer = build_tokenizer(
        tokenizer_name=tokenizer_name,
        tokenizer_kwargs={'model_max_length': 32},
    )

    model = ComposerHFCausalLM(
        tokenizer=tokenizer,
        pretrained_model_name_or_path='codellama/CodeLlama-7b-hf',
        pretrained_lora_id_or_path=None,
        trust_remote_code=False,
        init_device='meta',
        use_flash_attention_2=False,
        use_auth_token=False,
        config_overrides={
            'num_hidden_layers': 2,
            'hidden_size': 32,
            'intermediate_size': 64,
        },
        load_in_8bit=False,
        pretrained=False,
    )

    trainer = Trainer(
        model=model,
        device='gpu',
        parallelism_config={'fsdp': fsdp_config},
    )
    assert trainer.state.fsdp_enabled

    # Check for NaN weights
    for name, param in trainer.state.model.model.named_parameters():
        assert not torch.isnan(
            param,
        ).any(), f'NaN detected in parameter: {name}'
