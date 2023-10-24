# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any, Dict
from unittest.mock import Mock

import pytest
from composer.callbacks import Generate as ComposerGenerate
from composer.core.precision import get_precision_context
from composer.trainer import Trainer
from composer.utils import get_device, reproducibility
from omegaconf import DictConfig
from omegaconf import OmegaConf as om

from llmfoundry import COMPOSER_MODEL_REGISTRY
from llmfoundry.data.finetuning import build_finetuning_dataloader
from llmfoundry.utils import build_tokenizer
from tests.data_utils import make_tiny_ft_dataset


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


@pytest.mark.gpu
def test_mpt_generate_callback(tmpdir: Path):
    composer_device = get_device('gpu')
    reproducibility.seed_all(42)
    max_seq_len = 128

    # testing dataset and dataloader
    dataset_size = 5

    tiny_dataset_path = tmpdir / 'test-ift-data-small'
    tiny_dataset_path.mkdir()
    tiny_dataset_file = tiny_dataset_path / 'train.jsonl'
    make_tiny_ft_dataset(path=str(tiny_dataset_file), size=dataset_size)

    dataloader_cfg = DictConfig({
        'name': 'finetuning',
        'dataset': {
            'hf_name': str(tiny_dataset_path),
            'split': 'train',
            'max_seq_len': max_seq_len,
            'decoder_only_format': True,
            'allow_pad_trimming': False,
            'packing_ratio': None,
            'shuffle': True,
        },
        'drop_last': False,
        'num_workers': 4,
        'pin_memory': False,
        'prefetch_factor': 2,
        'persistent_workers': False,
        'timeout': 0
    })

    # build tokenizer
    tokenizer = build_tokenizer('EleutherAI/gpt-neox-20b', {})

    # build mpt model
    model_config = DictConfig({
        'name': 'mpt_causal_lm',
        'config_overrides': {
            'd_model': 128,
            'n_heads': 4,
            'n_layers': 2,
            'expansion_ratio': 2,
        },
    })
    model = COMPOSER_MODEL_REGISTRY[model_config.name](model_config, tokenizer)
    model = composer_device.module_to_device(model)

    # generate callback
    prompts = [
        'The best banana bread recipe is',
        '2+2=',
        'how much wood could a woodchuck chuck',
    ]
    gen_interval = 1
    generate = ComposerGenerate(
        prompts,
        interval=f'{gen_interval}ba',
        max_new_tokens=5,
        batch_size=len(prompts),
        use_cache=True,
    )
    generate.generate = Mock(wraps=generate.generate, autospec=True)

    # build trainer
    device_batch_size = 1
    train_dataloader = build_finetuning_dataloader(
        dataloader_cfg,
        tokenizer,
        device_batch_size,
    )

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        device=composer_device,
        max_duration=f'{gen_interval}ba',
        callbacks=[generate],
    )
    trainer.logger.log_table = Mock()
    trainer.fit()

    generate.generate.assert_called_once()
    trainer.logger.log_table.assert_called_once()
