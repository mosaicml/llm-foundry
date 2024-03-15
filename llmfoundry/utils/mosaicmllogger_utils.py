# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
import json
from typing import Any, Dict, List, Optional, Union

from composer.loggers import MosaicMLLogger
from composer.loggers.logger_destination import LoggerDestination
from omegaconf import DictConfig, ListConfig


def find_mosaicml_logger(
        loggers: List[LoggerDestination]) -> Union[MosaicMLLogger, None]:
    """Returns the first MosaicMLLogger from a list, and None otherwise."""
    return next(
        (logger for logger in loggers if isinstance(logger, MosaicMLLogger)),
        None)


def log_eval_analytics(mosaicml_logger: MosaicMLLogger,
                       model_configs: ListConfig, icl_tasks: Union[str,
                                                                   ListConfig],
                       eval_gauntlet_config: Optional[Union[str, DictConfig]]):
    """Logs analytics for runs using the `eval.py` script."""
    metrics: Dict[str, Any] = {
        'llmfoundry/script': 'eval',
    }

    if eval_gauntlet_config is not None:
        metrics['llmfoundry/gauntlet_configured'] = True
    else:
        metrics['llmfoundry/gauntlet_configured'] = False

    if isinstance(icl_tasks, str):
        metrics['llmfoundry/icl_configured'] = True
    elif len(icl_tasks) > 0:
        metrics['llmfoundry/icl_configured'] = True
    else:
        metrics['llmfoundry/icl_configured'] = False

    metrics['llmfoundry/model_configs'] = []
    for model_config in model_configs:
        model_config_data = {}
        if model_config.get('vocab_size', None) is not None:
            model_config_data['vocab_size'] = model_config.get('vocab_size')
        if model_config.get('d_model', None) is not None:
            model_config_data['d_model'] = model_config.get('d_model')
        if model_config.get('n_heads', None) is not None:
            model_config_data['n_heads'] = model_config.get('n_heads')

        if len(model_config_data) > 0:
            metrics['llmfoundry/model_configs'].append(
                json.dumps(model_config_data, sort_keys=True))
    mosaicml_logger.log_metrics(metrics)
    mosaicml_logger._flush_metadata(force_flush=True)


def log_train_analytics(mosaicml_logger: MosaicMLLogger,
                        model_config: DictConfig,
                        train_loader_config: DictConfig,
                        eval_loader_config: Union[DictConfig, ListConfig, None],
                        callback_configs: Union[DictConfig, None],
                        tokenizer_name: str, load_path: Union[str, None],
                        icl_tasks_config: Optional[Union[ListConfig, str]],
                        eval_gauntlet: Optional[Union[DictConfig, str]]):
    """Logs analytics for runs using the `train.py` script."""
    train_loader_dataset = train_loader_config.get('dataset', {})
    metrics: Dict[str, Any] = {
        'llmfoundry/tokenizer_name':
            tokenizer_name,
        'llmfoundry/script':
            'train',
        'llmfoundry/train_loader_name':
            train_loader_config.get('name'),
        'llmfoundry/train_loader_workers':
            train_loader_dataset.get('num_workers'),
    }

    if callback_configs is not None:
        metrics['llmfoundry/callbacks'] = [
            name for name, _ in callback_configs.items()
        ]

    if eval_gauntlet is not None:
        metrics['llmfoundry/gauntlet_configured'] = True
    else:
        metrics['llmfoundry/gauntlet_configured'] = False

    if icl_tasks_config is not None:
        if isinstance(icl_tasks_config, str):
            metrics['llmfoundry/icl_configured'] = True
        elif len(icl_tasks_config) > 0:
            metrics['llmfoundry/icl_configured'] = True
        else:
            metrics['llmfoundry/icl_configured'] = False
    else:
        metrics['llmfoundry/icl_configured'] = False

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

        if isinstance(eval_loader_config, ListConfig):
            eval_loader_configs: ListConfig = eval_loader_config
        else:
            eval_loader_configs = ListConfig([eval_loader_config])

        for loader_config in eval_loader_configs:
            eval_loader_info = {}
            eval_loader_dataset = loader_config.get('dataset', {})
            eval_loader_info['name'] = loader_config.get('name')
            eval_loader_info['num_workers'] = eval_loader_dataset.get(
                'num_workers', None)
            if eval_loader_dataset.get('hf_name', None) is not None:
                eval_loader_info['dataset_hf_name'] = eval_loader_dataset.get(
                    'hf_name')

            # Log as a key-sorted JSON string, so that we can easily parse it in Spark / SQL
            metrics['llmfoundry/eval_loaders'].append(
                json.dumps(eval_loader_info, sort_keys=True))

    if model_config['name'] == 'hf_casual_lm':
        metrics['llmfoundry/model_name'] = model_config.get(
            'pretrained_model_name_or_path')
    if model_config.get('vocab_size', None) is not None:
        metrics['llmfoundry/vocab_size'] = model_config.get('vocab_size'),
    if model_config.get('d_model', None) is not None:
        metrics['llmfoundry/d_model'] = model_config.get('d_model')
    if model_config.get('n_heads', None) is not None:
        metrics['llmfoundry/n_heads'] = model_config.get('n_heads')

    mosaicml_logger.log_metrics(metrics)
    mosaicml_logger._flush_metadata(force_flush=True)
