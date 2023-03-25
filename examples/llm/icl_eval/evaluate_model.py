# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import sys
import time
from typing import List

import torch
from composer.loggers import InMemoryLogger, LoggerDestination
from composer.trainer import Trainer
from composer.utils import reproducibility
from omegaconf import DictConfig
from omegaconf import OmegaConf as om

from examples.common.builders import build_icl_evaluators, build_logger
from examples.llm.src.model_registry import COMPOSER_MODEL_REGISTRY

if __name__ == '__main__':
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg = DictConfig(om.merge(yaml_cfg, cli_cfg))

    reproducibility.seed_all(cfg.get('seed', 1234))

    composer_model = COMPOSER_MODEL_REGISTRY[cfg.model.name](cfg.model,
                                                             cfg.tokenizer)
    evaluators, logger_keys = build_icl_evaluators(cfg,
                                                   composer_model.tokenizer)

    in_memory_logger = InMemoryLogger()  # track metrics in the in_memory_logger
    loggers: List[LoggerDestination] = [
        build_logger(name, logger_cfg)
        for name, logger_cfg in (cfg.get('loggers') or {}).items()
    ]
    loggers.append(in_memory_logger)

    fsdp_config = cfg.get('fsdp_config', None)
    fsdp_config = om.to_container(
        fsdp_config, resolve=True) if fsdp_config is not None else None

    load_path = cfg.get('load_path', None)

    trainer = Trainer(
        model=composer_model,
        loggers=loggers,
        fsdp_config=fsdp_config,  # type: ignore
        load_path=load_path,
        load_weights_only=True,
        progress_bar=False,
        log_to_console=True)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    a = time.time()
    trainer.eval(eval_dataloader=evaluators)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    b = time.time()

    print(f'Ran eval in: {b-a} seconds')

    for key in logger_keys:
        if key in in_memory_logger.data:
            result = in_memory_logger.data[key][0][1].item()
            print(f'{key}: {result}')
