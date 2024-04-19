# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
import json
import os
from typing import Any, Dict, List, Optional, Union

from composer.loggers import MosaicMLLogger
from composer.loggers.logger_destination import LoggerDestination
from composer.loggers.mosaicml_logger import (MOSAICML_ACCESS_TOKEN_ENV_VAR,
                                              MOSAICML_PLATFORM_ENV_VAR)

_MODEL_KEYS_TO_LOG = [
    'pretrained_model_name_or_path',
    'pretrained',
    'vocab_size',
    'd_model',
    'n_heads',
    'n_layers',
    'expansion_ratio',
    'max_seq_len',
]


def maybe_create_mosaicml_logger() -> Optional[MosaicMLLogger]:
    """Creates a MosaicMLLogger if the run was sent from the Mosaic platform."""
    if os.environ.get(MOSAICML_PLATFORM_ENV_VAR, 'false').lower(
    ) == 'true' and os.environ.get(MOSAICML_ACCESS_TOKEN_ENV_VAR):
        return MosaicMLLogger()


def find_mosaicml_logger(
        loggers: List[LoggerDestination]) -> Optional[MosaicMLLogger]:
    """Returns the first MosaicMLLogger from a list, and None otherwise."""
    return next(
        (logger for logger in loggers if isinstance(logger, MosaicMLLogger)),
        None)


def log_eval_analytics(mosaicml_logger: MosaicMLLogger,
                       model_configs: List[Dict[str, Any]],
                       icl_tasks: Union[str, List[Dict[str, Any]]],
                       eval_gauntlet_config: Optional[Union[str, Dict[str,
                                                                      Any]]]):
    """Logs analytics for runs using the `eval.py` script."""
    metrics: Dict[str, Any] = {
        'llmfoundry/script': 'eval',
    }

    metrics['llmfoundry/gauntlet_configured'] = eval_gauntlet_config is not None
    metrics['llmfoundry/icl_configured'] = isinstance(icl_tasks,
                                                      str) or len(icl_tasks) > 0

    metrics['llmfoundry/model_configs'] = []
    for model_config in model_configs:
        nested_model_config = model_config.get('model', {})
        model_config_data = {}
        for key in _MODEL_KEYS_TO_LOG:
            if nested_model_config.get(key, None) is not None:
                model_config_data[key] = nested_model_config.get(key)

        if len(model_config_data) > 0:
            metrics['llmfoundry/model_configs'].append(
                json.dumps(model_config_data, sort_keys=True))

    mosaicml_logger.log_metrics(metrics)
    mosaicml_logger._flush_metadata(force_flush=True)


def log_train_analytics(mosaicml_logger: MosaicMLLogger,
                        model_config: Dict[str,
                                           Any], train_loader_config: Dict[str,
                                                                           Any],
                        eval_loader_config: Optional[Union[Dict[str, Any],
                                                           List[Dict[str,
                                                                     Any]]]],
                        callback_configs: Optional[Dict[str, Any]],
                        tokenizer_name: str, load_path: Optional[str],
                        icl_tasks_config: Optional[Union[List[Dict[str, Any]],
                                                         str]],
                        eval_gauntlet: Optional[Union[Dict[str, Any], str]]):
    """Logs analytics for runs using the `train.py` script."""
    train_loader_dataset = train_loader_config.get('dataset', {})
    metrics: Dict[str, Any] = {
        'llmfoundry/tokenizer_name': tokenizer_name,
        'llmfoundry/script': 'train',
        'llmfoundry/train_loader_name': train_loader_config.get('name'),
    }

    if callback_configs is not None:
        metrics['llmfoundry/callbacks'] = [
            name for name, _ in callback_configs.items()
        ]

    metrics['llmfoundry/gauntlet_configured'] = eval_gauntlet is not None
    metrics['llmfoundry/icl_configured'] = (icl_tasks_config is not None and (
        (isinstance(icl_tasks_config, str) or len(icl_tasks_config) > 0)))

    if train_loader_dataset.get('hf_name', None) is not None:
        metrics['llmfoundry/train_dataset_hf_name'] = train_loader_dataset.get(
            'hf_name', None)
    if train_loader_config.get('name') == 'finetuning':
        metrics['llmfoundry/train_task_type'] = 'INSTRUCTION_FINETUNE'
    elif train_loader_config.get('name') == 'text':
        if load_path is not None or model_config.get('pretrained') == True:
            metrics['llmfoundry/train_task_type'] = 'CONTINUED_PRETRAIN'
        else:
            metrics['llmfoundry/train_task_type'] = 'PRETRAIN'

    if eval_loader_config is not None:
        metrics['llmfoundry/eval_loaders'] = []

        if isinstance(eval_loader_config, list):
            eval_loader_configs: list = eval_loader_config
        else:
            eval_loader_configs = [eval_loader_config]

        for loader_config in eval_loader_configs:
            eval_loader_info = {}
            eval_loader_dataset = loader_config.get('dataset', {})
            eval_loader_info['name'] = loader_config.get('name')
            if eval_loader_dataset.get('hf_name', None) is not None:
                eval_loader_info['dataset_hf_name'] = eval_loader_dataset.get(
                    'hf_name')

            # Log as a key-sorted JSON string, so that we can easily parse it in Spark / SQL
            metrics['llmfoundry/eval_loaders'].append(
                json.dumps(eval_loader_info, sort_keys=True))

    model_config_data = {}
    for key in _MODEL_KEYS_TO_LOG:
        if model_config.get(key, None) is not None:
            model_config_data[f'llmfoundry/{key}'] = model_config.get(key)

    if len(model_config_data) > 0:
        metrics.update(model_config_data)

    mosaicml_logger.log_metrics(metrics)
    mosaicml_logger._flush_metadata(force_flush=True)
