# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from typing import Callable

import pytest
from composer.core.precision import get_precision_context
from composer.utils import get_device
from transformers import PreTrainedTokenizerBase

from llmfoundry.models.hf.hf_causal_lm import ComposerHFCausalLM


@pytest.mark.gpu
@pytest.mark.parametrize('device', ['cpu', 'gpu'])
@pytest.mark.parametrize('attn_impl', ['triton', 'torch'])
def test_init_hfhub_mpt(
    device: str,
    attn_impl: str,
    build_tiny_hf_mpt: Callable[..., ComposerHFCausalLM],
    mpt_tokenizer: PreTrainedTokenizerBase,
):
    if device == 'cpu' and attn_impl == 'triton':
        pytest.skip(f'{attn_impl=} not implemented for {device=}.')
    composer_device = get_device(device)

    model = build_tiny_hf_mpt(attn_config={
        'attn_impl': attn_impl,
        'attn_uses_sequence_id': False,
    })
    model = composer_device.module_to_device(model)

    model.eval()

    with get_precision_context('amp_bf16' if composer_device.name ==
                               'gpu' else 'fp32'):
        _ = model.generate(
            composer_device.tensor_to_device(
                mpt_tokenizer('hello', return_tensors='pt')['input_ids']),
            max_new_tokens=2,
        )


def test_init_hfhub_mpt_cpu(
    build_tiny_hf_mpt: Callable[..., ComposerHFCausalLM],
    mpt_tokenizer: PreTrainedTokenizerBase,
):
    test_init_hfhub_mpt(device='cpu',
                        attn_impl='torch',
                        build_tiny_hf_mpt=build_tiny_hf_mpt,
                        mpt_tokenizer=mpt_tokenizer)
