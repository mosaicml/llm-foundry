# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
import copy
from unittest.mock import MagicMock

import pytest
import torch
import transformers
from composer.core import State
from composer.core.precision import get_precision_context
from composer.devices import DeviceGPU
from composer.loggers import Logger
from composer.utils import get_device
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from llmfoundry.utils.registry_utils import construct_from_registry
from llmfoundry import registry

from llmfoundry.data.text_data import (
    StreamingTextDataset,
    build_text_dataloader,
)
from llmfoundry.utils.builders import build_composer_model


@pytest.mark.gpu
@pytest.mark.parametrize('shift_labels', [True, False])
def test_loss_perp_v_len_callback(
    shift_labels: bool,
    monkeypatch: pytest.MonkeyPatch,
):
    try:
        from flash_attn.losses.cross_entropy import CrossEntropyLoss as FusedCrossEntropyLoss  # type: ignore # isort: skip
    except:
        pytest.skip('Fused cross entropy was not installed')

    composer_device = get_device(None)

    model_max_length = 12

    gptt = transformers.AutoTokenizer.from_pretrained('gpt2')
    gptt.pad_token_id = gptt.eos_token_id
    gptt.model_max_length = model_max_length
    gptt.padding_side = 'right'

    cfg = {
        'dataset': {
            'local': 'dummy-path',
            'remote': 'dummy-path',
            'split': 'train',
            'max_seq_len': model_max_length,
            'shuffle': True,
            'shuffle_seed': 0,
            'eos_token_id': gptt.eos_token_id,
        },
        'drop_last': False,
        'num_workers': 0,
        'prefetch_factor': None,
        'pin_memory': False,
        'persistent_workers': False,
        'timeout': 0,
    }

    ds_mock = MagicMock(spec=StreamingTextDataset)
    ds_mock.tokenizer = gptt
    monkeypatch.setattr(
        'llmfoundry.data.text_data.StreamingTextDataset',
        lambda *args,
        **kwargs: ds_mock,
    )
    dl = build_text_dataloader(
        **cfg,
        tokenizer=gptt,
        device_batch_size=1,
    )

    batch_strings = [
        'hello hey' + gptt.eos_token + ' the quick brown fox jumps',
    ]

    batch_tokenized = [gptt(b, padding=False) for b in batch_strings]

    batch_tokenized = [b['input_ids'] for b in batch_tokenized]

    batch = dl.dataloader.collate_fn(batch_tokenized)  # type: ignore

    for k, v in batch.items():  # type: ignore
        if isinstance(v, torch.Tensor):
            batch[k] = composer_device.tensor_to_device(v)  # type: ignore

    attention_impl = 'flash'

    conf_path = 'scripts/train/yamls/pretrain/testing.yaml'
    with open(conf_path) as f:
        test_cfg = om.load(f)

    assert isinstance(test_cfg, DictConfig)

    attn_config = {
        'attn_type': 'grouped_query_attention',
        'attn_impl': attention_impl,
        'attn_uses_sequence_id': True,
        'alibi': False,
        'rope': True,
        'rope_theta': 10000,
        'rope_impl': 'dail',
        'rope_dail_config': {
            'type': 'original',
            'pos_idx_in_fp32': True,
            'xpos_scale_base': 512,
        },
    }
    attn_config['kv_n_heads'] = 4

    test_cfg.model.init_device = 'cpu'
    test_cfg.model.init_config = {
        'name': 'baseline_',
        'init_std': 0.02,
    }
    test_cfg.model.attn_config = attn_config
    test_cfg.model.n_layers = 2
    test_cfg.model.n_heads = 8
    test_cfg.model.d_model = 128

    test_cfg = dict(om.to_container(test_cfg, resolve=True))  # type: ignore

    model = build_composer_model(
        name=test_cfg['model']['name'],
        cfg=test_cfg['model'],
        tokenizer=gptt,
    )
    assert model.shift_labels == True
    model.shift_labels = shift_labels

    model = composer_device.module_to_device(model)

    with get_precision_context('amp_bf16'):
        output = model(batch)
        loss = model.loss(output, batch)

        assert isinstance(loss, torch.Tensor)

        callback = construct_from_registry(
            name='loss_perp_v_len',
            registry=registry.callbacks,
            kwargs={
                'log_batch_interval': 100,
                'compute_batch_interval': 1,
            },
        ) 

        callback.loss_perp_v_len = callback.loss_perp_v_len.to(loss.device)
        state = State(
            model=model,
            rank_zero_seed=0,
            run_name='test_state',
            device=DeviceGPU(),
        )
        logger = Logger(state)
        state.outputs = output
        state.batch = batch

        callback.after_backward(state, logger)
        current_metric_dict = callback.loss_perp_v_len.compute()

        mean_loss_seq_id = torch.sum(
            current_metric_dict['mean_loss_seq_id_v_len'] *
            current_metric_dict['sum_length_seq_id'],
        ) / torch.sum(current_metric_dict['sum_length_seq_id'])
        mean_loss = torch.sum(
            current_metric_dict['mean_loss_v_len'] *
            current_metric_dict['sum_length'],
        ) / torch.sum(current_metric_dict['sum_length'])
        assert torch.allclose(loss, mean_loss_seq_id)
        assert torch.allclose(loss, mean_loss)
