# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

import pytest
from composer.callbacks import MemoryMonitor
from icecream import ic, install
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
from tests.data_utils import create_c4_dataset_xxsmall, gpt_tiny_cfg

install()


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
):
    # Read cfg from `testing.yaml`
    from tests.fixtures.autouse import REPO_DIR
    cfg_path: str = os.path.join(
        REPO_DIR, 'scripts/train/yamls/pretrain/testing.yaml'
    )
    with open(cfg_path, 'r', encoding='utf-8') as f:
        train_cfg = om.load(f)
    assert isinstance(train_cfg, DictConfig)

    # Set the dataset
    train_cfg.variables.data_local = dataset_name

    # Set batch size
    train_cfg.global_train_batch_size = 16
    train_cfg.device_eval_batch_size = 2
    train_cfg.device_train_microbatch_size = 2

    # Set duration
    train_cfg.max_duration = '16ba'
    train_cfg.eval_interval = '16ba'

    # TP needs unfused qkv (and we unfuse for no TP for a fair comparison)
    train_cfg.model.attn_cfg = {'fused_qkv': False}

    # loggers
    train_cfg.loggers = DictConfig({'inmemory': DictConfig({})})

    # default name
    train_cfg.variables.run_name = 'fsdp-test'

    if tp_strategy and tp_degree:
        train_cfg.variables.run_name = 'tp-test'
        train_cfg.tp_config = {
            'strategy': tp_strategy,
            'tensor_parallel_degree': tp_degree,
        }

        # when we replicate the data tp_degree times, each device must see tp_degree times more samples
        # because the samples are replicated
        train_cfg.train_loader.dataset.replication = tp_degree
        train_cfg.eval_loader.dataset.replication = tp_degree
        # train_cfg.device_train_microbatch_size *= tp_degree
        # train_cfg.device_eval_batch_size *= tp_degree
        train_cfg.global_train_batch_size *= tp_degree

    return train_cfg


def forward_pass(trainer):
    # reproducibility.seed_all(trainer.state.seed)
    batch = next(iter(trainer.state.train_dataloader))
    output = trainer.state.model.forward(batch)
    return output


def get_loss_array(trainer):
    logger = trainer.logger.destinations[0]
    loss_array = logger.get_timeseries(
        'loss/train/total',
    )['loss/train/total'],  # type: ignore
    return loss_array


def create_c4_dataset(data_dir: pathlib.Path):
    if data_dir.is_dir():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = create_c4_dataset_xxsmall(data_dir)
    return dataset_path


@pytest.mark.gpu
@pytest.mark.world_size(4)
def test_tp_train():
    """Test that we can train with FSDP-TP."""
    tp_degree = 2
    tp_strategy = 'ffn'

    # # create c4 dataset
    # my_dir = Path('/my-data-dir-2')
    # if my_dir.is_dir():
    #     shutil.rmtree(my_dir)
    # my_dir.mkdir(parents=True)
    # dataset_name_2 = create_c4_dataset_xxsmall(my_dir)
    fsdp_dataset_name = '/my-data-dir/my-copy-c4'
    tp_dataset_name = '/my-data-dir-2/my-copy-c4'

    # Get fsdp loss
    # fsdp_dataset_name = create_c4_dataset(pathlib.Path('/my-fsdp-data-dir/'))
    fsdp_cfg = get_cfg(fsdp_dataset_name)
    fsdp_trainer = train(fsdp_cfg)
    fsdp_trainer.close()
    fsdp_losses = get_loss_array(fsdp_trainer)
    ic(
        fsdp_losses
    )  #  fsdp_losses: (array([11.77620506, 11.76067352, 11.82744789, 11.71392155]),)

    # Get tp loss
    # tp_dataset_name = create_c4_dataset(pathlib.Path('/my-tp-data-dir/'))
    tp_cfg = get_cfg(tp_dataset_name, tp_strategy, tp_degree)
    tp_trainer = train(tp_cfg)
    tp_trainer.close()
    tp_losses = get_loss_array(tp_trainer)
    ic(
        tp_losses
    )  #tp_losses: (array([12.1361351 , 11.95866013, 11.91756916, 12.07012749]),)


@pytest.mark.gpu
def test_tp_train_with_one_gpu():
    """Test that when we have one GPU, we train DDP and not FSDP-TP."""
    with TemporaryDirectory() as tmp_path:
        # Make `train_cfg`` with a tensor parallelism strategy
        dataset_name = create_c4_dataset_xxsmall(Path(tmp_path))
        train_cfg = gpt_tiny_cfg(dataset_name, 'gpu')
        train_cfg.tp_config = {'strategy': 'ffn'}

        # Expect a warning
        with pytest.warns(
            UserWarning,
            match=
            r'FSDP\+TP is not applicable for single-GPU training. Reverting to DDP.',
        ):
            train(train_cfg)


@pytest.mark.gpu  # use gpu because `megablocks` only installed with `gpu` dependencies
def test_tp_train_with_moes():
    """Test that tensor parallelism is not compatible with MoEs."""
    # Make `cfg` for MoE model, fsdp, and tp
    train_cfg_path: str = 'scripts/train/yamls/pretrain/testing-moe.yaml'
    with open(train_cfg_path, 'r', encoding='utf-8') as f:
        train_cfg = om.load(f)
    model_cfg = train_cfg.model
    fsdp_cfg = train_cfg.fsdp_config
    tp_cfg = {'strategy': 'ffn'}

    # Expect an error
    with pytest.raises(
        ValueError,
        match='Tensor Parallelism is not currently supported for MoE models.',
    ):
        process_init_device(model_cfg, fsdp_cfg, tp_cfg)


# if __name__ == '__main__':
#     test_tp_train()
