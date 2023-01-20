# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from typing import Dict

import torch
from composer import Trainer
from composer.algorithms import BlurPool, ChannelsLast, LabelSmoothing, MixUp
from composer.callbacks import LRMonitor, MemoryMonitor, SpeedMonitor
from composer.loggers import ProgressBarLogger, WandBLogger
from composer.optim import DecoupledSGDW, MultiStepWithWarmupScheduler
from composer.utils import dist
from data import build_cifar10_dataspec
from model import build_composer_resnet_cifar
from omegaconf import DictConfig, OmegaConf


def build_logger(name: str, kwargs: Dict):
    if name == 'progress_bar':
        return ProgressBarLogger()
    elif name == 'wandb':
        return WandBLogger(**kwargs)
    else:
        raise ValueError(f'Not sure how to build logger: {name}')


def log_config(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    if 'wandb' in cfg.loggers:
        try:
            import wandb
        except ImportError as e:
            raise e
        if wandb.run:
            wandb.config.update(OmegaConf.to_container(cfg, resolve=True))


def main(config):
    if config.grad_accum == 'auto' and not torch.cuda.is_available():
        raise ValueError(
            'grad_accum="auto" requires training with a GPU; please specify grad_accum as an integer'
        )

    # Initialize dist to ensure CIFAR is only downloaded by rank 0
    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    if dist.get_world_size() > 1:
        dist.initialize_dist(device, 180)

    # Divide batch sizes by number of devices if running multi-gpu training
    train_batch_size = config.train_dataset.batch_size
    eval_batch_size = config.eval_dataset.batch_size
    if dist.get_world_size():
        train_batch_size //= dist.get_world_size()
        eval_batch_size //= dist.get_world_size()

    print('Building train dataloader')
    train_dataspec = build_cifar10_dataspec(
        data_path=config.train_dataset.path,
        is_streaming=config.train_dataset.is_streaming,
        batch_size=train_batch_size,
        local=config.train_dataset.local,
        is_train=True,
        download=config.train_dataset.download,
        drop_last=True,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True)
    print('Built train dataloader\n')

    print('Building evaluation dataloader')
    test_dataspec = build_cifar10_dataspec(
        data_path=config.eval_dataset.path,
        is_streaming=config.eval_dataset.is_streaming,
        batch_size=eval_batch_size,
        local=config.eval_dataset.local,
        is_train=False,
        download=config.eval_dataset.download,
        drop_last=False,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True)
    print('Built evaluation dataloader\n')

    print('Build Composer model')
    model = build_composer_resnet_cifar(model_name=config.model.name,
                                        num_classes=config.model.num_classes)
    print('Built Composer model\n')

    print('Building optimizer and learning rate scheduler')
    optimizer = DecoupledSGDW(model.parameters(),
                              lr=config.optimizer.lr,
                              momentum=config.optimizer.momentum,
                              weight_decay=config.optimizer.weight_decay)

    lr_scheduler = MultiStepWithWarmupScheduler(
        t_warmup=config.lr_scheduler.t_warmup,
        milestones=config.lr_scheduler.milestones,
        gamma=config.lr_scheduler.gamma)
    print('Built optimizer and learning rate scheduler')

    print('Building Speed, LR, and Memory monitoring callbacks')
    speed_monitor = SpeedMonitor(
        window_size=50
    )  # Measures throughput as samples/sec and tracks total training time
    lr_monitor = LRMonitor()  # Logs the learning rate
    memory_monitor = MemoryMonitor()  # Logs memory utilization
    print('Built Speed, LR, and Memory monitoring callbacks\n')

    print('Building algorithm recipes')
    if config.use_recipe:
        algorithms = [BlurPool(), ChannelsLast(), LabelSmoothing(), MixUp()]
    else:
        algorithms = None
    print('Built algorithm recipes\n')

    loggers = [
        build_logger(name, logger_config)
        for name, logger_config in config.loggers.items()
    ]

    print('Building Trainer')
    precision = 'amp' if device == 'gpu' else 'fp32'  # Mixed precision for fast training when using a GPU
    trainer = Trainer(
        run_name=config.run_name,
        model=model,
        train_dataloader=train_dataspec,
        eval_dataloader=test_dataspec,
        eval_interval='1ep',
        optimizers=optimizer,
        schedulers=lr_scheduler,
        algorithms=algorithms,
        loggers=loggers,
        max_duration=config.max_duration,
        callbacks=[speed_monitor, lr_monitor, memory_monitor],
        save_folder=config.save_folder,
        save_interval=config.save_interval,
        save_num_checkpoints_to_keep=config.save_num_checkpoints_to_keep,
        load_path=config.load_path,
        device=device,
        precision=precision,
        grad_accum=config.grad_accum,
        seed=config.seed)
    print('Built Trainer\n')

    print('Logging config')
    log_config(config)

    print('Run evaluation')
    trainer.eval()
    if config.is_train:
        print('Train!')
        trainer.fit()

    return trainer


if __name__ == '__main__':
    if len(sys.argv) < 2 or not os.path.exists(sys.argv[1]):
        raise ValueError('The first argument must be a path to a yaml config.')

    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open(yaml_path) as f:
        yaml_config = OmegaConf.load(f)
    cli_config = OmegaConf.from_cli(args_list)
    config = OmegaConf.merge(yaml_config, cli_config)
    main(config)
