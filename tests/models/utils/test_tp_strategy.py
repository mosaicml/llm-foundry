# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from omegaconf import OmegaConf as om
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
)

from llmfoundry.command_utils.train import train
from llmfoundry.models.mpt.modeling_mpt import ComposerMPTCausalLM
from llmfoundry.utils.builders import build_tp_strategy
from llmfoundry.utils.config_utils import process_init_device
from tests.data_utils import create_c4_dataset_xxsmall, gpt_tiny_cfg


@pytest.mark.gpu
def test_ffn_tp_strategy_layer_plan():
    # Actual layer plan from tp_strategy=fnn
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
    layer_plan = build_tp_strategy(tp_config['strategy'], model)

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


@pytest.mark.gpu
def test_no_tp_with_one_gpu():
    with TemporaryDirectory() as tmp_path:
        # Make `train_cfg`` with a tensor parallelism strategy
        train_cfg_path: str = 'scripts/train/yamls/pretrain/mpt-125m.yaml'
        with open(train_cfg_path, 'r', encoding='utf-8') as f:
            train_cfg = om.load(f)
        dataset_name = create_c4_dataset_xxsmall(Path(tmp_path))
        train_cfg = gpt_tiny_cfg(dataset_name, 'gpu')
        train_cfg.tp_config = {'strategy': 'ffn'}

        # Expect a warning to use DDP and not FSDP-TP when we have one GPU.
        with pytest.warns(
            UserWarning,
            match=
            r'FSDP\+TP is not applicable for single-GPU training. Reverting to DDP.',
        ):
            train(train_cfg)


@pytest.mark.gpu  # use gpu because `megablocks` only installed with `gpu` dependencies
def test_no_tp_with_moes():
    # Make `cfg` for MoE model, fsdp, and tp (tensor parallelism)
    train_cfg_path: str = 'scripts/train/yamls/pretrain/testing-moe.yaml'
    with open(train_cfg_path, 'r', encoding='utf-8') as f:
        train_cfg = om.load(f)
    model_cfg = train_cfg.model
    fsdp_cfg = train_cfg.fsdp_config
    tp_cfg = {'strategy': 'ffn'}

    # Expect an error for using tensor parallelism with MoEs
    with pytest.raises(
        ValueError,
        match='Tensor Parallelism is not currently supported for MoE models.',
    ):
        process_init_device(model_cfg, fsdp_cfg, tp_cfg)
