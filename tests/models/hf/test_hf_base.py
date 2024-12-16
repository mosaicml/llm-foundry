# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from llmfoundry.models.hf.hf_base import BaseHuggingFaceModel
from peft import PeftModel


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


@pytest.mark.parametrize('trainable', [True, False])
def test_pretrained_peft_trainable(trainable: bool):
    model = BaseHuggingFaceModel.build_inner_model(
        pretrained_model_name_or_path='facebook/opt-350m',
        pretrained_lora_id_or_path='ybelkada/opt-350m-lora',
        trust_remote_code=False,
        init_device='cpu',
        use_flash_attention_2=False,
        use_auth_token=False,
        config_overrides={},
        load_in_8bit=False,
        pretrained=True,
        prepare_for_fsdp=True,
        peft_is_trainable=trainable,
    )

    assert isinstance(model, PeftModel)

    n_trainable, n_all = model.get_nb_trainable_parameters()
    assert n_all > 0

    if trainable:
        assert n_trainable > 0
    else:
        assert n_trainable == 0