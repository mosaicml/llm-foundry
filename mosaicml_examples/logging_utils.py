# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

from omegaconf import OmegaConf as om


def log_config(cfg):
    print(om.to_yaml(cfg))
    if 'wandb' in cfg.get('loggers', {}):
        try:
            import wandb
        except ImportError as e:
            raise e
        if wandb.run:
            wandb.config.update(om.to_container(cfg, resolve=True))
