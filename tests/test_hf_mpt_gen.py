# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict

import pytest
from composer.core.precision import get_precision_context
from composer.utils import get_device, reproducibility
from omegaconf import DictConfig
from omegaconf import OmegaConf as om

from llmfoundry import COMPOSER_MODEL_REGISTRY
from llmfoundry.utils import build_tokenizer


@pytest.mark.gpu
@pytest.mark.parametrize('device', ['cpu', 'gpu'])
@pytest.mark.parametrize('attn_impl', ['triton', 'torch'])
def test_init_hfhub_mpt(device: str, attn_impl: str):
    if device == 'cpu' and attn_impl == 'triton':
        pytest.skip(f'{attn_impl=} not implemented for {device=}.')
    composer_device = get_device(device)

    with open('scripts/train/yamls/pretrain/testing.yaml') as f:
        test_cfg = om.load(f)

    assert isinstance(test_cfg, DictConfig)
    reproducibility.seed_all(test_cfg.get('seed', 42))

    attn_uses_sequence_id = True if test_cfg.get('eos_token_id',
                                                 None) is not None else False
    test_cfg.model = DictConfig({
        'name': 'hf_causal_lm',
        'pretrained_model_name_or_path': 'mosaicml/mpt-7b',
        'pretrained': False,
        'config_overrides': {
            'd_model': 128,
            'n_heads': 4,
            'n_layers': 2,
            'expansion_ratio': 2,
            'attn_config': {
                'attn_impl': attn_impl,
                'attn_uses_sequence_id': attn_uses_sequence_id,
            },
        },
    })

    # build tokenizer
    tokenizer_cfg: Dict[str,
                        Any] = om.to_container(test_cfg.tokenizer,
                                               resolve=True)  # type: ignore
    tokenizer_name = tokenizer_cfg['name']
    tokenizer_kwargs = tokenizer_cfg.get('kwargs', {})
    tokenizer = build_tokenizer(tokenizer_name, tokenizer_kwargs)

    # build model
    model = COMPOSER_MODEL_REGISTRY[test_cfg.model.name](test_cfg.model,
                                                         tokenizer)
    test_cfg.n_params = sum(p.numel() for p in model.parameters())

    model.eval()
    model = composer_device.module_to_device(model)

    with get_precision_context('amp_bf16' if composer_device.name ==
                               'gpu' else 'fp32'):
        _ = model.generate(
            composer_device.tensor_to_device(
                tokenizer('hello', return_tensors='pt')['input_ids']),
            max_new_tokens=10,
        )


def test_init_hfhub_mpt_cpu():
    test_init_hfhub_mpt(device='cpu', attn_impl='torch')
