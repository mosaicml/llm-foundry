# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""Example script to train a ResNet model on ImageNet."""

import os
import sys
from typing import Dict

import torch
from composer import Trainer
from composer.algorithms import (EMA, SAM, BlurPool, ChannelsLast, ColOut,
                                 LabelSmoothing, MixUp, ProgressiveResizing,
                                 RandAugment, StochasticDepth)
from composer.callbacks import LRMonitor, MemoryMonitor, SpeedMonitor
from composer.loggers import ProgressBarLogger, WandBLogger
from composer.optim import CosineAnnealingWithWarmupScheduler, DecoupledSGDW
from composer.utils import dist, reproducibility
from omegaconf import OmegaConf

from examples.common.config_utils import log_config
from examples.resnet_imagenet.data import build_imagenet_dataspec
from examples.resnet_imagenet.model import build_composer_resnet


def build_logger(name: str, kwargs: Dict):
    if name == 'progress_bar':
        return ProgressBarLogger()
    elif name == 'wandb':
        return WandBLogger(**kwargs)
    else:
        raise ValueError(f'Not sure how to build logger: {name}')


def main(config):
    reproducibility.seed_all(config.seed)
    if config.grad_accum == 'auto' and not torch.cuda.is_available():
        raise ValueError(
            'grad_accum="auto" requires training with a GPU; please specify grad_accum as an integer'
        )

    # If using a recipe, update the config's loss name, eval and train resize sizes, and the max duration
    if config.recipe_name:
        if config.recipe_name not in ['mild', 'medium', 'hot']:
            raise ValueError(
                f'recipe_name={config.recipe_name}, but must be one of ["mild", "medium", "hot"]'
            )
        recipe_config = config[config.recipe_name]
        for key, value in recipe_config.items():
            OmegaConf.update(config, key, value)

    # Divide batch sizes by number of devices if running multi-gpu training
    train_batch_size = config.train_dataset.batch_size
    eval_batch_size = config.eval_dataset.batch_size
    if dist.get_world_size():
        train_batch_size //= dist.get_world_size()
        eval_batch_size //= dist.get_world_size()

    # Train dataset
    print('Building train dataloader')
    train_dataspec = build_imagenet_dataspec(
        data_path=config.train_dataset.path,
        is_streaming=config.train_dataset.is_streaming,
        batch_size=train_batch_size,
        local=config.train_dataset.local,
        is_train=True,
        drop_last=True,
        shuffle=True,
        resize_size=config.train_dataset.resize_size,
        crop_size=config.train_dataset.crop_size,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True)

    print('Built train dataloader\n')

    # Validation dataset
    print('Building evaluation dataloader')
    eval_dataspec = build_imagenet_dataspec(
        data_path=config.eval_dataset.path,
        is_streaming=config.train_dataset.is_streaming,
        batch_size=eval_batch_size,
        local=config.eval_dataset.local,
        is_train=False,
        drop_last=False,
        shuffle=False,
        resize_size=config.eval_dataset.resize_size,
        crop_size=config.eval_dataset.crop_size,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True)
    print('Built evaluation dataloader\n')

    # Instantiate torchvision ResNet model
    print('Building Composer model')
    composer_model = build_composer_resnet(model_name=config.model.name,
                                           loss_name=config.model.loss_name,
                                           num_classes=config.model.num_classes)
    print('Built Composer model\n')

    # Optimizer
    print('Building optimizer and learning rate scheduler')
    optimizer = DecoupledSGDW(composer_model.parameters(),
                              lr=config.optimizer.lr,
                              momentum=config.optimizer.momentum,
                              weight_decay=config.optimizer.weight_decay)

    # Learning rate scheduler: LR warmup for 8 epochs, then cosine decay for the rest of training
    lr_scheduler = CosineAnnealingWithWarmupScheduler(
        t_warmup=config.scheduler.t_warmup, alpha_f=config.scheduler.alpha_f)
    print('Built optimizer and learning rate scheduler\n')

    # Callbacks for logging
    print('Building Speed, LR, and Memory monitoring callbacks')
    speed_monitor = SpeedMonitor(
        window_size=50
    )  # Measures throughput as samples/sec and tracks total training time
    lr_monitor = LRMonitor()  # Logs the learning rate
    memory_monitor = MemoryMonitor()  # Logs memory utilization

    # Callback for checkpointing
    print('Built Speed, LR, and Memory monitoring callbacks\n')

    # Recipes for training ResNet architectures on ImageNet in order of increasing training time and accuracy
    # To learn about individual methods, check out "Methods Overview" in our documentation: https://docs.mosaicml.com/
    print('Building algorithm recipes')
    if config.recipe_name == 'mild':
        algorithms = [
            BlurPool(),
            ChannelsLast(),
            EMA(half_life='100ba', update_interval='20ba'),
            ProgressiveResizing(initial_scale=0.5,
                                delay_fraction=0.4,
                                finetune_fraction=0.2),
            LabelSmoothing(smoothing=0.08),
        ]
    elif config.recipe_name == 'medium':
        algorithms = [
            BlurPool(),
            ChannelsLast(),
            EMA(half_life='100ba', update_interval='20ba'),
            ProgressiveResizing(initial_scale=0.5,
                                delay_fraction=0.4,
                                finetune_fraction=0.2),
            LabelSmoothing(smoothing=0.1),
            MixUp(alpha=0.2),
            SAM(rho=0.5, interval=10),
        ]
    elif config.recipe_name == 'hot':
        algorithms = [
            BlurPool(),
            ChannelsLast(),
            EMA(half_life='100ba', update_interval='20ba'),
            ProgressiveResizing(initial_scale=0.6,
                                delay_fraction=0.2,
                                finetune_fraction=0.2),
            LabelSmoothing(smoothing=0.13),
            MixUp(alpha=0.25),
            SAM(rho=0.5, interval=5),
            ColOut(p_col=0.05, p_row=0.05),
            RandAugment(depth=1, severity=9),
            StochasticDepth(target_layer_name='ResNetBottleneck',
                            stochastic_method='sample',
                            drop_distribution='linear',
                            drop_rate=0.1)
        ]
    else:
        algorithms = None
    print('Built algorithm recipes\n')

    loggers = [
        build_logger(name, logger_config)
        for name, logger_config in config.loggers.items()
    ]

    # Create the Trainer!
    print('Building Trainer')
    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    precision = 'amp' if device == 'gpu' else 'fp32'  # Mixed precision for fast training when using a GPU
    trainer = Trainer(
        run_name=config.run_name,
        model=composer_model,
        train_dataloader=train_dataspec,
        eval_dataloader=eval_dataspec,
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
        save_overwrite=config.get('save_overwrite', False),
        load_path=config.load_path,
        device=device,
        precision=precision,
        grad_accum=config.grad_accum,
        seed=config.seed,
        python_log_level=config.get('python_log_level', None),
    )
    print('Built Trainer\n')

    print('Logging config')
    log_config(config)

    print('Run evaluation')
    trainer.eval()
    if config.is_train:
        print('Train!')
        trainer.fit()

    # Return trainer for testing purposes
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
