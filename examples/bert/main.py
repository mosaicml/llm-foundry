# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from typing import Optional, cast

from composer import Trainer
from composer.utils import reproducibility
from omegaconf import DictConfig
from omegaconf import OmegaConf as om

from examples.bert.src.hf_bert import create_hf_bert_mlm
from examples.bert.src.mosaic_bert import create_mosaic_bert_mlm
from examples.common.builders import (build_algorithm, build_callback,
                                      build_dataloader, build_logger,
                                      build_optimizer, build_scheduler)
from examples.common.config_utils import log_config, update_batch_size_info


def build_model(cfg: DictConfig):
    if cfg.name == 'hf_bert':
        return create_hf_bert_mlm(
            pretrained_model_name=cfg.pretrained_model_name,
            use_pretrained=cfg.get('use_pretrained', None),
            model_config=cfg.get('model_config', None),
            tokenizer_name=cfg.get('tokenizer_name', None),
            gradient_checkpointing=cfg.get('gradient_checkpointing', None))
    elif cfg.name == 'mosaic_bert':
        return create_mosaic_bert_mlm(
            pretrained_model_name=cfg.pretrained_model_name,
            pretrained_checkpoint=cfg.get('pretrained_checkpoint', None),
            model_config=cfg.get('model_config', None),
            tokenizer_name=cfg.get('tokenizer_name', None),
            gradient_checkpointing=cfg.get('gradient_checkpointing', None))
    else:
        raise ValueError(f'Not sure how to build model with name={cfg.name}')


def main(cfg: DictConfig,
         return_trainer: bool = False,
         do_train: bool = True) -> Optional[Trainer]:
    print('Training using config: ')
    print(om.to_yaml(cfg))
    reproducibility.seed_all(cfg.seed)

    # Get batch size info
    cfg = update_batch_size_info(cfg)

    # Build Model
    print('Initializing model...')
    model = build_model(cfg.model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'{n_params=:.4e}')

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

    # Loggers
    loggers = [
        build_logger(name, logger_cfg)
        for name, logger_cfg in cfg.get('loggers', {}).items()
    ]

    # Callbacks
    callbacks = [
        build_callback(name, callback_cfg)
        for name, callback_cfg in cfg.get('callbacks', {}).items()
    ]

    # Algorithms
    algorithms = [
        build_algorithm(name, algorithm_cfg)
        for name, algorithm_cfg in cfg.get('algorithms', {}).items()
    ]

    if cfg.get('run_name') is None:
        cfg.run_name = os.environ.get('COMPOSER_RUN_NAME', 'bert')

    # Build the Trainer
    trainer = Trainer(
        run_name=cfg.run_name,
        seed=cfg.seed,
        model=model,
        algorithms=algorithms,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        train_subset_num_batches=cfg.get('train_subset_num_batches', -1),
        eval_subset_num_batches=cfg.get('eval_subset_num_batches', -1),
        optimizers=optimizer,
        schedulers=scheduler,
        max_duration=cfg.max_duration,
        eval_interval=cfg.eval_interval,
        progress_bar=cfg.progress_bar,
        log_to_console=cfg.log_to_console,
        console_log_interval=cfg.console_log_interval,
        loggers=loggers,
        callbacks=callbacks,
        precision=cfg.precision,
        device=cfg.get('device', None),
        device_train_microbatch_size=cfg.get('device_train_microbatch_size',
                                             'auto'),
        save_folder=cfg.get('save_folder', None),
        save_interval=cfg.get('save_interval', '1000ba'),
        save_num_checkpoints_to_keep=cfg.get('save_num_checkpoints_to_keep',
                                             -1),
        save_overwrite=cfg.get('save_overwrite', False),
        load_path=cfg.get('load_path', None),
        load_weights_only=cfg.get('load_weights_only', False),
        python_log_level=cfg.get('python_log_level', None),
    )

    print('Logging config...')
    log_config(cfg)

    if do_train:
        print('Starting training...')
        trainer.fit()

    if return_trainer:
        return trainer


if __name__ == '__main__':
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg = om.merge(yaml_cfg, cli_cfg)
    cfg = cast(DictConfig, cfg)  # for type checking
    main(cfg)
