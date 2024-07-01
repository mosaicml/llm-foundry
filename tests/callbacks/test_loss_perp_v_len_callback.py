# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
from typing import Any
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

from llmfoundry import registry
from llmfoundry.callbacks.loss_perp_v_len_callback import LossPerpVLen
from llmfoundry.data.text_data import (
    StreamingTextDataset,
    build_text_dataloader,
)
from llmfoundry.utils.builders import build_composer_model
from llmfoundry.utils.registry_utils import construct_from_registry


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


def test_metric():
    batch_size = 2
    seq_len = 100
    labels = torch.tensor([[
        1,
    ] * seq_len] * batch_size)
    logits = torch.tensor([[
        1,
    ] * seq_len] * batch_size)
    sequence_id = torch.tensor([[
        0,
    ] * 10 + [
        1,
    ] * 90, [
        0,
    ] * 50 + [
        1,
    ] * 50])
    loss = torch.rand([batch_size, seq_len])
    perplexity = torch.exp(loss)

    def mock_loss_fn(input_logits: Any, input_labels: Any):
        del input_logits, input_labels
        return loss

    loss_v_len_metric = LossPerpVLen(ignore_index=-100)
    loss_v_len_metric.update(
        labels=labels,
        logits=logits,
        sequence_id=sequence_id,
        loss_fn=mock_loss_fn,
    )
    metric_dict = loss_v_len_metric.compute()

    assert torch.all(metric_dict['sum_length'] == 2 * torch.ones([100]))
    assert torch.all(
        metric_dict['sum_length_seq_id'] == torch.tensor([
            4,
        ] * 10 + [
            3,
        ] * 40 + [
            1,
        ] * 40 + [
            0,
        ] * 10),
    )
    assert torch.all(metric_dict['mean_loss_v_len'] == torch.mean(loss, dim=0))
    assert torch.all(
        metric_dict['mean_perplexity_v_len'] == torch.mean(perplexity, dim=0),
    )

    expected_mean_loss_seq_id_v_len_0 = (
        loss[0][:10] + loss[0][10:20] + loss[1][0:10] + loss[1][50:60]
    ) / 4
    expected_mean_loss_seq_id_v_len_1 = (
        loss[0][20:60] + loss[1][10:50] + loss[1][60:100]
    ) / 3
    expected_mean_loss_seq_id_v_len_2 = loss[0][60:100]
    expected_mean_loss_seq_id_v_len_3 = -1

    assert torch.all(
        metric_dict['mean_loss_seq_id_v_len'][0:10] ==
        expected_mean_loss_seq_id_v_len_0,
    )
    assert torch.all(
        metric_dict['mean_loss_seq_id_v_len'][10:50] ==
        expected_mean_loss_seq_id_v_len_1,
    )
    assert torch.all(
        metric_dict['mean_loss_seq_id_v_len'][50:90] ==
        expected_mean_loss_seq_id_v_len_2,
    )
    assert torch.all(
        metric_dict['mean_loss_seq_id_v_len'][90:100] ==
        expected_mean_loss_seq_id_v_len_3,
    )

    expected_mean_perplexity_seq_id_v_len_0 = (
        perplexity[0][:10] + perplexity[0][10:20] + perplexity[1][0:10] +
        perplexity[1][50:60]
    ) / 4
    expected_mean_perplexity_seq_id_v_len_1 = (
        perplexity[0][20:60] + perplexity[1][10:50] + perplexity[1][60:100]
    ) / 3
    expected_mean_perplexity_seq_id_v_len_2 = perplexity[0][60:100]
    expected_mean_perplexity_seq_id_v_len_3 = -1

    assert torch.all(
        metric_dict['mean_perplexity_seq_id_v_len'][0:10] ==
        expected_mean_perplexity_seq_id_v_len_0,
    )
    assert torch.all(
        metric_dict['mean_perplexity_seq_id_v_len'][10:50] ==
        expected_mean_perplexity_seq_id_v_len_1,
    )
    assert torch.all(
        metric_dict['mean_perplexity_seq_id_v_len'][50:90] ==
        expected_mean_perplexity_seq_id_v_len_2,
    )
    assert torch.all(
        metric_dict['mean_perplexity_seq_id_v_len'][90:100] ==
        expected_mean_perplexity_seq_id_v_len_3,
    )


def test_valid_labels():
    batch_size = 1
    seq_len = 100
    ignore_labels_len = 10
    labels = torch.tensor([[
        1,
    ] * (seq_len - ignore_labels_len) + [
        -100,
    ] * ignore_labels_len] * batch_size)
    logits = torch.tensor([[
        1,
    ] * seq_len] * batch_size)
    sequence_id = torch.tensor([[
        0,
    ] * seq_len])
    loss = torch.rand([batch_size, seq_len])

    def mock_loss_fn(input_logits: Any, input_labels: Any):
        del input_logits, input_labels
        return loss

    loss_v_len_metric = LossPerpVLen(ignore_index=-100)
    loss_v_len_metric.update(
        labels=labels,
        logits=logits,
        sequence_id=sequence_id,
        loss_fn=mock_loss_fn,
    )
    metric_dict = loss_v_len_metric.compute()
    assert torch.all(metric_dict['sum_length'][-ignore_labels_len:] == 0)
    assert torch.all(metric_dict['sum_length_seq_id'][-ignore_labels_len:] == 0)
    assert torch.all(metric_dict['mean_loss_v_len'][-ignore_labels_len:] == -1)
    assert torch.all(
        metric_dict['mean_perplexity_v_len'][-ignore_labels_len:] == -1,
    )
    assert torch.all(
        metric_dict['mean_loss_seq_id_v_len'][-ignore_labels_len:] == -1,
    )
    assert torch.all(
        metric_dict['mean_perplexity_seq_id_v_len'][-ignore_labels_len:] == -1,
    )


def test_padding():
    batch_size = 2
    seq_len = 100

    labels_no_pad = torch.tensor([[
        1,
    ] * seq_len] * batch_size)
    logits_no_pad = torch.tensor([[
        1,
    ] * seq_len] * batch_size)
    sequence_id_no_pad = torch.tensor([[
        0,
    ] * 10 + [
        1,
    ] * 90, [
        0,
    ] * 50 + [
        1,
    ] * 50])
    loss_no_pad = torch.rand([batch_size, seq_len])

    def mock_loss_fn_no_pad(input_logits: Any, input_labels: Any):
        del input_logits, input_labels
        return loss_no_pad

    loss_v_len_metric_no_pad = LossPerpVLen(ignore_index=-100)
    loss_v_len_metric_no_pad.update(
        labels=labels_no_pad,
        logits=logits_no_pad,
        sequence_id=sequence_id_no_pad,
        loss_fn=mock_loss_fn_no_pad,
    )
    metric_dict_no_pad = loss_v_len_metric_no_pad.compute()

    pad_len = 10
    labels_pad = torch.tensor([[
        1,
    ] * seq_len + [
        -100,
    ] * pad_len] * batch_size)
    logits_pad = torch.tensor([[
        1,
    ] * (seq_len + pad_len)] * batch_size)
    sequence_id_pad = torch.tensor([[
        0,
    ] * 10 + [
        1,
    ] * 90 + [
        -1,
    ] * pad_len, [
        0,
    ] * 50 + [
        1,
    ] * 50 + [
        -1,
    ] * pad_len])
    loss_pad = torch.cat([loss_no_pad,
                          torch.rand([batch_size, pad_len])],
                         dim=-1)

    def mock_loss_fn_pad(input_logits: Any, input_labels: Any):
        del input_logits, input_labels
        return loss_pad

    loss_v_len_metric_pad = LossPerpVLen(ignore_index=-100)
    loss_v_len_metric_pad.update(
        labels=labels_pad,
        logits=logits_pad,
        sequence_id=sequence_id_pad,
        loss_fn=mock_loss_fn_pad,
    )
    metric_dict_pad = loss_v_len_metric_pad.compute()

    assert torch.all(metric_dict_pad['sum_length'][-pad_len:] == 0)
    assert torch.all(metric_dict_pad['sum_length_seq_id'][-pad_len:] == 0)
    assert torch.all(metric_dict_pad['mean_loss_v_len'][-pad_len:] == -1)
    assert torch.all(metric_dict_pad['mean_perplexity_v_len'][-pad_len:] == -1)
    assert torch.all(metric_dict_pad['mean_loss_seq_id_v_len'][-pad_len:] == -1)
    assert torch.all(
        metric_dict_pad['mean_perplexity_seq_id_v_len'][-pad_len:] == -1,
    )

    assert torch.all(
        metric_dict_pad['sum_length'][:-pad_len] ==
        metric_dict_no_pad['sum_length'],
    )
    assert torch.all(
        metric_dict_pad['sum_length_seq_id'][:-pad_len] ==
        metric_dict_no_pad['sum_length_seq_id'],
    )
    assert torch.all(
        metric_dict_pad['mean_loss_v_len'][:-pad_len] ==
        metric_dict_no_pad['mean_loss_v_len'],
    )
    assert torch.all(
        metric_dict_pad['mean_perplexity_v_len'][:-pad_len] ==
        metric_dict_no_pad['mean_perplexity_v_len'],
    )
    assert torch.all(
        metric_dict_pad['mean_loss_seq_id_v_len'][:-pad_len] ==
        metric_dict_no_pad['mean_loss_seq_id_v_len'],
    )
    assert torch.all(
        metric_dict_pad['mean_perplexity_seq_id_v_len'][:-pad_len] ==
        metric_dict_no_pad['mean_perplexity_seq_id_v_len'],
    )
