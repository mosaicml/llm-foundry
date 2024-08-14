# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional

import pytest
from composer.models.huggingface import maybe_get_underlying_model
from peft import PeftConfig, PeftModel
from transformers import LlamaForCausalLM, PreTrainedModel

from llmfoundry.models.hf.hf_causal_lm import ComposerHFCausalLM
from llmfoundry.models.utils import init_empty_weights


@pytest.mark.gpu
@pytest.mark.parametrize(
    'peft_config',
    [
        None,
        {
            'peft_type': 'LORA',
            'task_type': 'CAUSAL_LM',
            'lora_alpha': 32,
            'r': 2,
            'target_modules': [
                'q_proj',
                'k_proj',
                'v_proj',
            ],
        },
    ],
)
def test_hf_transform(peft_config: Optional[dict]):
    model_cfg = {
        'pretrained_model_name_or_path': 'codellama/CodeLlama-7b-hf',
        'config_overrides': {
            'num_hidden_layers': 2,
            'hidden_size': 32,
            'intermediate_size': 64,
        },
        'pretrained': False,
        'peft_config': peft_config,
        'init_device': 'meta',
        'tokenizer': 'codellama/CodeLlama-7b-hf',
    }

    class TransformedHFCausalLM(ComposerHFCausalLM):

        def transform_model(self, model: PreTrainedModel) -> PreTrainedModel:
            assert isinstance(model, LlamaForCausalLM)
            with init_empty_weights():
                model.config.num_hidden_layers = 1
                new_model = type(model)(model.config)
            return new_model

        def get_peft_config(
            self,
            peft_config_dict: dict[str, Any],
        ) -> PeftConfig:
            peft_config_dict['target_modules'] = ['o_proj']
            return super().get_peft_config(peft_config_dict)

    composer_model = TransformedHFCausalLM(**model_cfg)
    model = composer_model.model
    inner_model = maybe_get_underlying_model(model)

    if peft_config:
        peft_model = composer_model.model
        assert isinstance(peft_model, PeftModel)

        target_modules = peft_model.peft_config[peft_model.active_adapter
                                               ].target_modules
        assert list(target_modules) == ['o_proj']

    assert isinstance(inner_model, LlamaForCausalLM)
    assert inner_model.config.num_hidden_layers == 1
