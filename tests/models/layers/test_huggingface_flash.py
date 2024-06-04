# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import contextlib

import pytest
from composer.core.precision import get_precision_context

from llmfoundry.models.hf.hf_fsdp import rgetattr
from llmfoundry.models.layers.attention import is_flash_v2_installed
from llmfoundry.utils.builders import build_composer_model, build_tokenizer


@pytest.mark.gpu
@pytest.mark.world_size(2)
@pytest.mark.parametrize('model_name', ['codellama'])
@pytest.mark.parametrize('use_flash_attention_2', [True, False])
@pytest.mark.parametrize('init_device', ['cpu', 'mixed', 'meta'])
def test_flash2(model_name: str, use_flash_attention_2: bool, init_device: str):
    if model_name == 'codellama':
        model_cfg = {
            'name': 'hf_causal_lm',
            'pretrained_model_name_or_path': 'codellama/CodeLlama-7b-hf',
            'config_overrides': {
                'num_hidden_layers': 2,
                'intermediate_size': 64,
                'hidden_size': 64,
            },
            'pretrained': False,
            'init_device': init_device,
        }

        tokenizer_name = 'codellama/CodeLlama-7b-hf'
        from transformers.models.llama.modeling_llama import (
            LlamaAttention,
            LlamaFlashAttention2,
        )
        flash_attn_class = LlamaFlashAttention2 if use_flash_attention_2 else LlamaAttention
        attention_layers_attr = 'model.model.layers'
        attention_attr = 'self_attn'
    else:
        raise ValueError(f'Unknown model: {model_name}')

    if use_flash_attention_2:
        model_cfg['use_flash_attention_2'] = True

    tokenizer = build_tokenizer(
        tokenizer_name=tokenizer_name,
        tokenizer_kwargs={'model_max_length': 10},
    )
    tokenizer.pad_token = tokenizer.eos_token

    error_context = pytest.raises(
        ValueError,
        match='use_flash_attention_2 is set to True',
    ) if not is_flash_v2_installed(
    ) and use_flash_attention_2 else contextlib.nullcontext()

    with error_context:
        name = model_cfg.pop('name')
        model = build_composer_model(
            name=name,
            cfg=model_cfg,
            tokenizer=tokenizer,
        )

        # check that it actually used flash attention 2
        assert model.model.config._attn_implementation == (
            'flash_attention_2' if use_flash_attention_2 else 'eager'
        )
        attention_layer = rgetattr(
            rgetattr(model, attention_layers_attr)[0],
            attention_attr,
        )
        assert isinstance(attention_layer, flash_attn_class)

        # Skip attempting to run forward/backward when some devices have meta params
        # because we are not instantiating a full Trainer here, which contains the logic
        # to move params off of meta device.
        if init_device == 'cpu':
            tokenized_input = tokenizer([
                'Hello world blah blah',
                'Goodbye world',
            ],
                                        return_tensors='pt',
                                        padding=True)
            tokenized_input['labels'] = tokenized_input['input_ids'].clone()

            tokenized_input = {k: v.cuda() for k, v in tokenized_input.items()}
            model.to('cuda')

            with get_precision_context('amp_bf16'):
                # We're just testing that flash attention 2 runs okay
                outputs = model(tokenized_input)
                loss = outputs.loss
                loss.backward()
