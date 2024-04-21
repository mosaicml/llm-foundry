# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict

from composer.models.huggingface import maybe_get_underlying_model

from llmfoundry.models.hf import ComposerHFCausalLM


def test_olmo_wraps():
    conf: Dict[str, Any] = {
        'model': {
            'name': 'hf_causal_lm',
            'pretrained_model_name_or_path': 'allenai/OLMo-7B',
            'pretrained': False,
            'config_overrides': {
                'n_layers': 2,
            }
        },
    }

    conf['model'].pop('name')
    model = ComposerHFCausalLM(tokenizer=None, **conf['model'])  # type: ignore

    # check that all the modules we except are blocked from FSDP wrapping
    underlying_model = maybe_get_underlying_model(model.model)
    assert (not hasattr(underlying_model.model,
                        'fsdp_wrap')) or (not underlying_model.model._fsdp_wrap)
    assert (not hasattr(underlying_model.model.transformer, 'fsdp_wrap')) or (
        not underlying_model.model.transformer._fsdp_wrap)
    assert (not hasattr(underlying_model.model.transformer.wte, 'fsdp_wrap')
           ) or (not underlying_model.model.transformer.wte._fsdp_wrap)
    assert (not hasattr(underlying_model.model.transformer.ff_out, 'fsdp_wrap')
           ) or (not underlying_model.model.transformer.ff_out._fsdp_wrap)
