# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import warnings

from composer import Trainer
from composer.utils import dist, reproducibility
from omegaconf import OmegaConf as om

from examples.common.builders import (build_algorithm, build_callback,
                                      build_dataloader, build_logger,
                                      build_optimizer, build_scheduler)
from examples.common.logging_utils import log_config
from examples.llm.src.model_registry import COMPOSER_MODEL_REGISTRY


def calculate_batch_size_info(global_batch_size, device_microbatch_size):
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
        device_grad_accum = device_batch_size // device_microbatch_size
    else:
        raise ValueError(f'Not sure how to parse {device_microbatch_size=}')

    return device_batch_size, device_microbatch_size, device_grad_accum


# Coming soon: this conversion math will be done inside Composer Trainer
def update_batch_size_info(cfg):
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


def build_composer_model(cfg):
    warnings.filterwarnings(
        action='ignore',
        message='Torchmetrics v0.9 introduced a new argument class property')
    try:
        return COMPOSER_MODEL_REGISTRY[cfg.name](cfg)
    except:
        raise ValueError(f'Not sure how to build model with name={cfg.name}')


def main(cfg):
    reproducibility.seed_all(cfg.seed)

    # Run Name
    cfg.run_name = cfg.get('run_name', os.environ.get('COMPOSER_RUN_NAME',
                                                      'llm'))

    # Get batch size info
    cfg = update_batch_size_info(cfg)

    # Read FSDP Config as a dict
    fsdp_config = cfg.get('fsdp_config', None)
    fsdp_config = om.to_container(fsdp_config,
                                  resolve=True) if fsdp_config else None

    # Restrict model init device to 'meta' and 'cpu',
    # using 'cuda' vs. 'cuda:id' is tricky and can lead to common user errors
    # when multiple GPUs are available.
    # Also 'meta' is only valid when using FSDP
    assert cfg.model.device in ['meta', 'cpu']
    if fsdp_config is None and cfg.model.device == 'meta':
        print(
            "Using init device `cfg.model.device='meta'` is only valid when using FSDP! "
            "Reverting to `cfg.model.device='cpu'`.")
        cfg.model.device = 'cpu'

    # Build Model
    # For fast initialization of MosaicGPT, use cfg.model.device='meta'
    print('Initializing model...')
    model = build_composer_model(cfg.model)
    cfg.n_params = sum(p.numel() for p in model.parameters())
    print(f'{cfg.n_params=:.2e}')
    if hasattr(model, 'num_fwd_flops'):
        print(f'{model.num_fwd_flops=:.2e}')

    # Dataloaders
    print('Building train loader...')
    train_loader = build_dataloader(cfg.train_loader,
                                    cfg.device_train_batch_size)
    print('Building eval loader...')
    eval_loader = build_dataloader(cfg.eval_loader, cfg.device_eval_batch_size)

    # Optimizer
    optimizer = build_optimizer(cfg.optimizer, model)

    # Scheduler
    scheduler = build_scheduler(cfg.scheduler)

    # we use (cfg.get(...) or {}) instead of cfg.get(..., {}) so that
    # .items() works even when the value is None
    # Loggers
    loggers = [
        build_logger(name, logger_cfg)
        for name, logger_cfg in cfg.get('loggers' or {}).items()
    ]

    # Callbacks
    callbacks = [
        build_callback(name, callback_cfg)
        for name, callback_cfg in cfg.get('callbacks' or {}).items()
    ]

    # Algorithms
    algorithms = [
        build_algorithm(name, algorithm_cfg)
        for name, algorithm_cfg in cfg.get('algorithms' or {}).items()
    ]

    # Build the Trainer
    trainer = Trainer(
        run_name=cfg.run_name,
        seed=cfg.seed,
        model=model,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        optimizers=optimizer,
        schedulers=scheduler,
        max_duration=cfg.max_duration,
        eval_interval=cfg.eval_interval,
        eval_subset_num_batches=cfg.get('eval_subset_num_batches', -1),
        progress_bar=cfg.get('progress_bar', False),
        log_to_console=cfg.get('log_to_console', True),
        console_log_interval=cfg.get('console_log_interval', '1ba'),
        loggers=loggers,
        callbacks=callbacks,
        precision=cfg.precision,
        algorithms=algorithms,
        device_train_microbatch_size=cfg.get('device_train_microbatch_size',
                                             'auto'),
        fsdp_config=fsdp_config,  # type: ignore
        save_folder=cfg.get('save_folder', None),
        save_interval=cfg.get('save_interval', '1000ba'),
        save_num_checkpoints_to_keep=cfg.get('save_num_checkpoints_to_keep',
                                             -1),
        save_overwrite=cfg.get('save_overwrite', False),
        load_path=cfg.get('load_path', None),
        load_weights_only=cfg.get('load_weights_only', False),
    )

    print('Logging config...')
    log_config(cfg)

    print('Starting training...')
    trainer.fit()

    print('Done.')


if __name__ == '__main__':
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg = om.merge(yaml_cfg, cli_cfg)
    main(cfg)
