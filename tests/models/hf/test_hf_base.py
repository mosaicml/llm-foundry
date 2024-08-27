# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from llmfoundry.models.hf.hf_base import BaseHuggingFaceModel


def test_build_inner_model_fsdp():
    model = BaseHuggingFaceModel.build_inner_model(
        pretrained_model_name_or_path='codellama/CodeLlama-7b-hf',
        pretrained_lora_id_or_path=None,
        trust_remote_code=False,
        init_device='cpu',
        use_flash_attention_2=False,
        use_auth_token=False,
        config_overrides={
            'num_hidden_layers': 2,
            'hidden_size': 32,
            'intermediate_size': 64,
        },
        load_in_8bit=False,
        pretrained=False,
        prepare_for_fsdp=True,
    )

    assert model.fsdp_wrap_fn(model.model.layers[0])
