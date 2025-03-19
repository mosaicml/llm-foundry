# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from composer.core.precision import get_precision_context
from composer.utils import get_device
from transformers import PreTrainedTokenizerBase


@pytest.mark.gpu
@pytest.mark.parametrize('device', ['cpu', 'gpu'])
@pytest.mark.parametrize('attn_impl', ['flash', 'torch'])
def test_init_hfhub_codellama(
    device: str,
    attn_impl: str,
    tiny_llama_tokenizer: PreTrainedTokenizerBase,
):
    if device == 'cpu' and attn_impl == 'flash':
        pytest.skip(f'{attn_impl=} not implemented for {device=}.')
    composer_device = get_device(device)

    # Create a configuration for the HF causal LM
    config = {
        'pretrained_model_name_or_path': 'codellama/CodeLlama-7b-hf',
        'pretrained': False,
        'config_overrides': {
            'num_hidden_layers': 2,
            'hidden_size': 32,
            'intermediate_size': 64,
        },
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
                tiny_llama_tokenizer('def hello_world():',
                                     return_tensors='pt')['input_ids'],
            ),
            max_new_tokens=2,
        )


def test_init_hfhub_codellama_cpu(
    tiny_llama_tokenizer: PreTrainedTokenizerBase,
):
    """CPU-only version of the test for convenience."""
    test_init_hfhub_codellama(
        device='cpu',
        attn_impl='torch',
        tiny_llama_tokenizer=tiny_llama_tokenizer,
    )
