# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""A starter script for fine-tuning a BERT model on your own dataset."""

import os
import sys
from typing import Optional, cast

import transformers
from composer import Trainer
from composer.core.types import Dataset
from composer.utils import dist, reproducibility
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from torch.utils.data import DataLoader

from examples.bert.src.glue.data import create_glue_dataset
from examples.bert.src.hf_bert import create_hf_bert_classification
from examples.bert.src.mosaic_bert import create_mosaic_bert_classification
from examples.common.builders import (build_algorithm, build_callback,
                                      build_logger, build_optimizer,
                                      build_scheduler)
from examples.common.config_utils import log_config, update_batch_size_info


def build_my_dataloader(cfg: DictConfig, device_batch_size: int):
    """Create a dataloader for classification.

    **Modify this function to train on your own dataset!**

    This function is provided as a starter code to simplify fine-tuning a BERT
    classifier on your dataset. We'll use the dataset for QNLI (one of the
    GLUE tasks) as a demonstration.

    Args:
        cfg (DictConfig): An omegaconf config that houses all the configuration
            variables needed to instruct dataset/dataloader creation.
        device_batch_size (int): The size of the batches that the dataloader
            should produce.

    Returns:
        dataloader: A dataloader set up for use of the Composer Trainer.
    """
    # As a demonstration, we're using the QNLI dataset from the GLUE suite
    # of tasks.
    #
    # Note: We create our dataset using the `create_glue_dataset` utility
    #   defined in `./src/glue/data.py`. If you inspect that code, you'll see
    #   that we're taking some extra steps so that our dataset yields examples
    #   that follow a particular format. In particular, the raw text is
    #   tokenized and some of the data columns are removed. The result is that
    #   each example is a dictionary with the following:
    #
    #     - 'input_ids': the tokenized raw text
    #     - 'label': the target class that the text belongs to
    #     - 'attention_mask': a list of 1s and 0s to indicate padding
    #
    # When you set up your own dataset, it should handle tokenization to yield
    # examples with a similar structure!
    #
    # REPLACE THIS WITH YOUR OWN DATASET:
    dataset = create_glue_dataset(
        task='qnli',
        split=cfg.split,
        tokenizer_name=cfg.tokenizer_name,
        max_seq_length=cfg.max_seq_len,
    )

    dataset = cast(Dataset, dataset)
    dataloader = DataLoader(
        dataset,
        # As an alternative to formatting the examples inside the dataloader,
        # you can write a custom data collator to do that instead.
        collate_fn=transformers.default_data_collator,
        batch_size=device_batch_size,
        sampler=dist.get_sampler(dataset,
                                 drop_last=cfg.drop_last,
                                 shuffle=cfg.shuffle),
        num_workers=cfg.num_workers,
        pin_memory=cfg.get('pin_memory', True),
        prefetch_factor=cfg.get('prefetch_factor', 2),
        persistent_workers=cfg.get('persistent_workers', True),
        timeout=cfg.get('timeout', 0),
    )

    return dataloader


def build_model(cfg: DictConfig):
    # Note: cfg.num_labels should match the number of classes in your dataset!
    if cfg.name == 'hf_bert':
        return create_hf_bert_classification(
            num_labels=cfg.num_labels,
            pretrained_model_name=cfg.pretrained_model_name,
            use_pretrained=cfg.get('use_pretrained', False),
            model_config=cfg.get('model_config'),
            tokenizer_name=cfg.get('tokenizer_name'),
            gradient_checkpointing=cfg.get('gradient_checkpointing'))
    elif cfg.name == 'mosaic_bert':
        return create_mosaic_bert_classification(
            num_labels=cfg.num_labels,
            pretrained_model_name=cfg.pretrained_model_name,
            pretrained_checkpoint=cfg.get('pretrained_checkpoint'),
            model_config=cfg.get('model_config'),
            tokenizer_name=cfg.get('tokenizer_name'),
            gradient_checkpointing=cfg.get('gradient_checkpointing'))
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
    train_loader = build_my_dataloader(cfg.train_loader,
                                       cfg.device_train_batch_size)
    print('Building eval loader...')
    eval_loader = build_my_dataloader(cfg.eval_loader,
                                      cfg.device_eval_batch_size)

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
        cfg.run_name = os.environ.get('COMPOSER_RUN_NAME',
                                      'sequence-classification')

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
        device=cfg.get('device'),
        device_train_microbatch_size=cfg.get('device_train_microbatch_size',
                                             'auto'),
        save_folder=cfg.get('save_folder'),
        save_interval=cfg.get('save_interval', '1000ba'),
        save_num_checkpoints_to_keep=cfg.get('save_num_checkpoints_to_keep',
                                             -1),
        save_overwrite=cfg.get('save_overwrite', False),
        load_path=cfg.get('load_path'),
        load_weights_only=True,
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
