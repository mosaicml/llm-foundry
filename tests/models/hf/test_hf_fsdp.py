# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from composer.models.huggingface import maybe_get_underlying_model
from omegaconf import DictConfig

from llmfoundry.models.hf import ComposerHFCausalLM


def test_olmo_wraps():
    conf: dict = {
        'model': {
            'name': 'hf_causal_lm',
            'pretrained_model_name_or_path': 'allenai/OLMo-7B',
            'pretrained': False,
            'trust_remote_code': False,
            'config_overrides': {
                'num_hidden_layers': 2,
            }
        },
    }

    config = DictConfig(conf)

    model = ComposerHFCausalLM(config.model, None)

    # check that all the modules we except are blocked from FSDP wrapping
    underlying_model = maybe_get_underlying_model(model.model)
    assert not underlying_model.model._fsdp_wrap
    assert not underlying_model.model.transformer._fsdp_wrap
    assert not underlying_model.model.transformer.wte._fsdp_wrap
    assert not underlying_model.model.transformer.ff_out._fsdp_wrap
