# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Union

from composer.utils import dist
from omegaconf import DictConfig
from omegaconf import OmegaConf as om


def calculate_batch_size_info(global_batch_size: int,
                              device_microbatch_size: Union[int, str]):
    if global_batch_size % dist.get_world_size() != 0:
        raise ValueError(
            f'Global batch size {global_batch_size} is not divisible by {dist.get_world_size()} '
            'as a result, the batch size would be truncated, please adjust `global_batch_size` '
            f'to be divisible by world size, {dist.get_world_size()}.')
    device_batch_size = global_batch_size // dist.get_world_size()
    if device_microbatch_size == 'auto':
        device_grad_accum = 'auto'
    elif isinstance(device_microbatch_size, int):
        if device_microbatch_size > device_batch_size:
            print(
                f'WARNING: device_microbatch_size > device_batch_size, '
                f'will be reduced from {device_microbatch_size} -> {device_batch_size}.'
            )
            device_microbatch_size = device_batch_size
        device_grad_accum = math.ceil(device_batch_size /
                                      device_microbatch_size)
    else:
        raise ValueError(f'Not sure how to parse {device_microbatch_size=}')

    return device_batch_size, device_microbatch_size, device_grad_accum


# Coming soon: this conversion math will be done inside Composer Trainer
def update_batch_size_info(cfg: DictConfig):
    device_train_batch_size, device_train_microbatch_size, device_train_grad_accum = calculate_batch_size_info(
        cfg.global_train_batch_size, cfg.device_train_microbatch_size)
    cfg.n_gpus = dist.get_world_size()
    cfg.device_train_batch_size = device_train_batch_size
    cfg.device_train_microbatch_size = device_train_microbatch_size
    cfg.device_train_grad_accum = device_train_grad_accum
    # Safely set `device_eval_batch_size` if not provided by user
    if 'device_eval_batch_size' not in cfg:
        if cfg.device_train_microbatch_size == 'auto':
            cfg.device_eval_batch_size = 1  # TODO debug auto eval microbatching
        else:
            cfg.device_eval_batch_size = cfg.device_train_microbatch_size
    return cfg


def log_config(cfg: DictConfig):
    print(om.to_yaml(cfg))
    if 'wandb' in cfg.get('loggers', {}):
        try:
            import wandb
        except ImportError as e:
            raise e
        if wandb.run:
            wandb.config.update(om.to_container(cfg, resolve=True))
