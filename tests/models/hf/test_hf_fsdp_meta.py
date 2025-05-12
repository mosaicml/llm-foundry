# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from typing import Optional

import pytest
import torch
import torch.nn as nn
from composer import Trainer
from transformers import PreTrainedTokenizerBase

from llmfoundry.models.hf.hf_causal_lm import ComposerHFCausalLM


@contextmanager
def deterministic_torch_context():
    """Context manager for deterministic torch settings."""
    # Save previous settings
    prev_deterministic = torch.are_deterministic_algorithms_enabled()

    # Check if fill_uninitialized_memory exists in this PyTorch version
    prev_fill_uninitialized: Optional[bool] = None
    has_fill_uninitialized = hasattr(
        torch.utils.deterministic,
        'fill_uninitialized_memory',
    )

    if has_fill_uninitialized:
        prev_fill_uninitialized = torch.utils.deterministic.fill_uninitialized_memory  # type: ignore
        torch.utils.deterministic.fill_uninitialized_memory = True  # type: ignore

    # Enable deterministic settings
    torch.use_deterministic_algorithms(True)

    try:
        yield
    finally:
        # Restore previous settings
        torch.use_deterministic_algorithms(prev_deterministic)
        if has_fill_uninitialized and prev_fill_uninitialized is not None:
            torch.utils.deterministic.fill_uninitialized_memory = prev_fill_uninitialized  # type: ignore


@pytest.mark.gpu
@pytest.mark.world_size(2)
def test_hf_meta_init_fsdp(tiny_codellama_tokenizer: PreTrainedTokenizerBase):
    # Use deterministic mode to ensure uninitialized weights are filled with NaNs
    with deterministic_torch_context():
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
        model_instance = trainer.state.model.model
        if isinstance(model_instance, nn.Module):
            for name, param in model_instance.named_parameters():
                if isinstance(param, torch.Tensor):
                    assert not torch.isnan(param).any(
                    ), f'NaN detected in parameter: {name}'

            # Check that the LlamaRMSNorm layers are initialized with ones
            for name, module in model_instance.named_modules():
                if module.__class__.__name__ == 'LlamaRMSNorm':
                    weight = module.weight
                    if isinstance(weight, torch.Tensor):
                        assert torch.allclose(
                            weight,
                            torch.ones_like(weight),
                        ), f'LlamaRMSNorm layer {name} not initialized with ones: {weight}'
