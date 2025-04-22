# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
import contextlib
from typing import Any
from unittest.mock import patch

import pytest
from composer.utils import dist
from peft import PeftModel
from transformers import PreTrainedModel

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

    assert model.fsdp_wrap_fn(model.model.layers[0])  # type: ignore


def test_pretrained_peft_trainable():
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
    )

    assert isinstance(model, PeftModel)

    n_trainable, n_all = model.get_nb_trainable_parameters()  # type: ignore
    assert n_all > 0
    assert n_trainable > 0


@pytest.mark.gpu
@pytest.mark.world_size(2)
@pytest.mark.parametrize('error_rank', [None, 0, 1])
def test_build_inner_model_download_thread(
    tiny_gpt2_model: PreTrainedModel,
    error_rank: int | None,
):

    def mock_build_func(
        *args: Any,
        **kwargs: Any,
    ):
        if dist.get_global_rank() == error_rank:
            raise RuntimeError('Constructor error')
        return tiny_gpt2_model

    error_context = contextlib.nullcontext(
    ) if error_rank is None else pytest.raises(
        RuntimeError,
        match=
        rf'Error initializing model on ranks \[{error_rank}\]. See individual rank logs for more details',
    )

    with patch(
        'llmfoundry.models.hf.hf_base.AutoModelForCausalLM.from_pretrained',
        mock_build_func,
    ):
        with error_context:
            _ = BaseHuggingFaceModel.build_inner_model(
                pretrained_model_name_or_path='gpt2',
                pretrained_lora_id_or_path=None,
                trust_remote_code=False,
                init_device='cpu',
                use_flash_attention_2=False,
                use_auth_token=False,
                config_overrides={},
                load_in_8bit=False,
                pretrained=True,
                prepare_for_fsdp=True,
            )
