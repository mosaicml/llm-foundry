# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib
from pathlib import Path
from typing import Optional

import pytest
from composer import Trainer
from composer.utils import dist
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
)

from llmfoundry.command_utils.train import train
from llmfoundry.models.mpt.modeling_mpt import ComposerMPTCausalLM
from llmfoundry.utils.builders import build_tp_strategies
from llmfoundry.utils.config_utils import process_init_device
from tests.data_utils import gpt_tiny_cfg


@pytest.mark.gpu
@pytest.mark.filterwarnings(
    'ignore:tp_strategies is experimental and may change with future versions.',
)
def test_ffn_tp_strategy():
    """Test the FFN tensor parallelism strategy is correct."""
    # Create layer plan from fnn tp_strategy
    tp_config = {
        'strategy': 'ffn',
    }

    model_cfg = {
        'name': 'mpt_causal_lm',
        'd_model': 128,
        'n_heads': 4,
        'n_layers': 3,
        'expansion_ratio': 1,
        'max_seq_len': 16,
        'vocab_size': 50368,
    }
    model = ComposerMPTCausalLM(**model_cfg)
    layer_plan = build_tp_strategies(tp_config['strategy'], model)

    # Expected layer plan
    _expected_layer_plan = {
        'ffn':
            PrepareModuleInput(
                input_layouts=Shard(0),
                desired_input_layouts=Replicate(),
                use_local_output=True,
            ),
        'ffn.down_proj':
            RowwiseParallel(
                input_layouts=Shard(-1),
                output_layouts=Shard(0),
            ),
        'ffn.up_proj':
            ColwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(-1),
            ),
    }
    expected_layer_plan = {
        f'model.transformer.blocks.{layer_idx}.{name}': layer_plan
        for name, layer_plan in _expected_layer_plan.items()
        for layer_idx in range(model_cfg['n_layers'])
    }

    # Compare expected and actual layer plans
    for (n1, lp1), (n2, lp2) in zip(
        sorted(expected_layer_plan.items()),
        sorted(layer_plan.items()),
    ):
        assert n1 == n2
        assert type(lp1) == type(lp2)
        if isinstance(
            lp1,
            PrepareModuleInput,
        ) and isinstance(lp2, PrepareModuleInput):
            assert lp1.input_layouts == lp2.input_layouts
            assert lp1.desired_input_layouts == lp2.desired_input_layouts
            assert lp1.use_local_output == lp2.use_local_output
        elif (
            isinstance(lp1, ColwiseParallel) and
            isinstance(lp2, ColwiseParallel)
        ) or (
            isinstance(lp1, RowwiseParallel) and
            isinstance(lp2, RowwiseParallel)
        ):
            assert lp1.input_layouts == lp2.input_layouts
            assert lp1.output_layouts == lp2.output_layouts
            assert lp1.use_local_output == lp2.use_local_output
        else:
            raise ValueError(f'Layer plan of wrong type: {type(layer_plan)}')


def get_cfg(
    dataset_name: pathlib.Path,
    tp_strategy: Optional[str] = None,
    tp_degree: Optional[int] = None,
    yaml_path: str = 'scripts/train/yamls/pretrain/testing.yaml',
):
    # Read cfg from `testing.yaml`
    from tests.fixtures.autouse import REPO_DIR
    cfg_path: str = os.path.join(REPO_DIR, yaml_path)
    with open(cfg_path, 'r', encoding='utf-8') as f:
        train_cfg = om.load(f)
    assert isinstance(train_cfg, DictConfig)

    # Set the name, dataset, loggers
    train_cfg.variables.run_name = 'fsdp-test'
    train_cfg.variables.data_local = dataset_name
    train_cfg.loggers = DictConfig({'inmemory': DictConfig({})})

    # Set batch size, duration
    train_cfg.global_train_batch_size = 16
    train_cfg.device_eval_batch_size = 2
    train_cfg.device_train_microbatch_size = 2
    train_cfg.max_duration = '1ep'
    train_cfg.eval_interval = '1ep'

    # TP needs unfused qkv (even without TP, we unfuse qkv for a fair comparison)
    train_cfg.model.attn_cfg = {'fused_qkv': False}

    if tp_strategy and tp_degree:
        train_cfg.variables.run_name = 'tp-test'
        train_cfg.tp_config = {
            'strategy': tp_strategy,
            'tensor_parallel_degree': tp_degree,
        }

    return train_cfg


def get_loss_array(trainer: Trainer):
    logger = trainer.logger.destinations[0]
    loss_array = logger.get_timeseries( # type: ignore
        'loss/train/total',
    )['loss/train/total']
    return loss_array


@pytest.mark.gpu
@pytest.mark.world_size(4)
@pytest.mark.parametrize('tp_degree', [2])
@pytest.mark.parametrize('tp_strategy', ['ffn'])
def test_tp_train(
    tp_degree: int,
    tp_strategy: str,
    tmp_path: Path,
    tiny_text_dataset_path: Path,
):
    """Test that we can train with FSDP-TP."""
    tp_dataset_name = dist.all_gather_object(tiny_text_dataset_path)[0]

    # Train model with TP and get loss
    tp_cfg = get_cfg(pathlib.Path(tp_dataset_name), tp_strategy, tp_degree)
    tp_trainer = train(tp_cfg)
    tp_trainer.close()
    tp_loss = get_loss_array(tp_trainer)

    # Compare loss and expected loss for TP
    import numpy as np
    expected_tp_loss = np.array([
        11.846275,
        11.854343,
        11.881471,
        11.873639,
        11.814651,
        11.809362,
    ])
    np.testing.assert_allclose(tp_loss, expected_tp_loss)


@pytest.mark.gpu
def test_tp_train_with_one_gpu(tmp_path: Path, tiny_text_dataset_path: Path):
    """Test that when we have one GPU, we train DDP and not FSDP-TP."""
    # Make `train_cfg`` with a tensor parallelism strategy
    train_cfg = gpt_tiny_cfg(str(tiny_text_dataset_path), 'gpu')
    train_cfg.tp_config = {'strategy': 'ffn'}

    # Expect a warning
    with pytest.warns(
        UserWarning,
        match=
        r'FSDP\+TP is not applicable for single-GPU training. Reverting to DDP.',
    ):
        train(train_cfg)


@pytest.mark.gpu  # use gpu because `megablocks` only installed with `gpu` dependencies
@pytest.mark.parametrize('tp_degree', [2])
@pytest.mark.parametrize('tp_strategy', ['ffn'])
def test_tp_train_with_moes(tp_degree: int, tp_strategy: str):
    """Test that tensor parallelism is not compatible with MoEs."""
    # Make `cfg` for MoE model, fsdp, and tp
    moe_yaml_path: str = 'scripts/train/yamls/pretrain/testing-moe.yaml'
    dataset_name = Path('')  # dummy dataset path
    train_cfg = get_cfg(dataset_name, tp_strategy, tp_degree, moe_yaml_path)

    # Expect an error
    with pytest.raises(
        ValueError,
        match='Tensor Parallelism is not currently supported for MoE models.',
    ):
        process_init_device(
            train_cfg.model,
            train_cfg.fsdp_config,
            train_cfg.tp_config,
        )
