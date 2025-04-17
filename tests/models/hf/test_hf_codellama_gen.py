# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from composer.core.precision import get_precision_context
from composer.utils import get_device
from transformers import PreTrainedModel, PreTrainedTokenizerBase


@pytest.mark.gpu
@pytest.mark.parametrize('device', ['cpu', 'gpu'])
@pytest.mark.parametrize('attn_impl', ['flash', 'torch'])
def test_init_hfhub_codellama(
    device: str,
    attn_impl: str,
    tiny_codellama_model: PreTrainedModel,
    tiny_llama_tokenizer: PreTrainedTokenizerBase,
    tmp_path: Path,
):
    if device == 'cpu' and attn_impl == 'flash':
        pytest.skip(f'{attn_impl=} not implemented for {device=}.')
    composer_device = get_device(device)

    save_path = tmp_path / 'hf-save'
    tiny_codellama_model.save_pretrained(save_path)

    # Create a configuration for the HF causal LM
    config = {
        'pretrained_model_name_or_path': str(save_path),
        'pretrained': False,
    }

    from llmfoundry.utils.builders import build_composer_model
    model = build_composer_model(
        name='hf_causal_lm',
        cfg=config,
        tokenizer=tiny_llama_tokenizer,
    )
    model = composer_device.module_to_device(model)
    model.eval()

    with get_precision_context(
        'amp_bf16' if composer_device.name == 'gpu' else 'fp32',
    ):
        _ = model.generate( # type: ignore
            composer_device.tensor_to_device(
                tiny_llama_tokenizer('def hello_world():',  # type: ignore
                                     return_tensors='pt')['input_ids'],
            ),
            max_new_tokens=2,
        )


def test_init_hfhub_codellama_cpu(
    tiny_codellama_model: PreTrainedModel,
    tiny_llama_tokenizer: PreTrainedTokenizerBase,
    tmp_path: Path,
):
    """CPU-only version of the test for convenience."""
    test_init_hfhub_codellama(
        device='cpu',
        attn_impl='torch',
        tiny_llama_tokenizer=tiny_llama_tokenizer,
        tiny_codellama_model=tiny_codellama_model,
        tmp_path=tmp_path,
    )
