# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Union

from composer import algorithms
from composer.callbacks import (LRMonitor, MemoryMonitor, OptimizerMonitor,
                                RuntimeEstimator, SpeedMonitor)
from composer.core import Evaluator
from composer.datasets.in_context_learning_evaluation import \
    get_icl_task_dataloader
from composer.loggers import WandBLogger
from composer.optim import DecoupledAdamW
from composer.optim.scheduler import (ConstantWithWarmupScheduler,
                                      CosineAnnealingWithWarmupScheduler,
                                      LinearWithWarmupScheduler)
from composer.utils import dist
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from transformers import (AutoTokenizer, PreTrainedTokenizer,
                          PreTrainedTokenizerFast)

from llmfoundry.callbacks import (FDiffMetrics, Generate, GlobalLRScaling,
                                  LayerFreezing, MonolithicCheckpointSaver,
                                  ScheduledGarbageCollector)
from llmfoundry.optim import (DecoupledAdaLRLion, DecoupledClipLion,
                              DecoupledLionW)

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


def build_callback(name, kwargs):
    if name == 'lr_monitor':
        return LRMonitor()
    elif name == 'memory_monitor':
        return MemoryMonitor()
    elif name == 'speed_monitor':
        return SpeedMonitor(window_size=kwargs.get('window_size', 1),
                            gpu_flops_available=kwargs.get(
                                'gpu_flops_available', None))
    elif name == 'fdiff':
        return FDiffMetrics(**kwargs)
    elif name == 'runtime_estimator':
        return RuntimeEstimator()
    elif name == 'optimizer_monitor':
        return OptimizerMonitor(log_optimizer_metrics=kwargs.get(
            'log_optimizer_metrics', True),)
    elif name == 'generate_callback':
        prompts = kwargs.pop('prompts')
        return Generate(prompts=list(prompts), **kwargs)
    elif name == 'global_lr_scaling':
        return GlobalLRScaling(**kwargs)
    elif name == 'layer_freezing':
        return LayerFreezing(**kwargs)
    elif name == 'mono_ckpt_saver':
        return MonolithicCheckpointSaver(**kwargs)
    elif name == 'scheduled_gc':
        return ScheduledGarbageCollector(**kwargs)
    else:
        raise ValueError(f'Not sure how to build callback: {name}')


def build_logger(name, kwargs):
    if name == 'wandb':
        return WandBLogger(**kwargs)
    else:
        raise ValueError(f'Not sure how to build logger: {name}')


def build_algorithm(name, kwargs):
    if name == 'gradient_clipping':
        return algorithms.GradientClipping(**kwargs)
    elif name == 'alibi':
        return algorithms.Alibi(**kwargs)
    elif name == 'fused_layernorm':
        return algorithms.FusedLayerNorm(**kwargs)
    elif name == 'gated_linear_units':
        return algorithms.GatedLinearUnits(**kwargs)
    elif name == 'low_precision_layernorm':
        return algorithms.LowPrecisionLayerNorm(**kwargs)
    else:
        raise ValueError(f'Not sure how to build algorithm: {name}')


def build_optimizer(cfg, model):
    if cfg.name == 'decoupled_adamw':
        return DecoupledAdamW(model.parameters(),
                              lr=cfg.lr,
                              betas=cfg.betas,
                              eps=cfg.eps,
                              weight_decay=cfg.weight_decay)
    elif cfg.name == 'decoupled_lionw':
        return DecoupledLionW(model.parameters(),
                              lr=cfg.lr,
                              betas=cfg.betas,
                              weight_decay=cfg.weight_decay)
    elif cfg.name == 'clip_lion':
        return DecoupledClipLion(model.parameters(),
                                 lr=cfg.lr,
                                 betas=cfg.betas,
                                 weight_decay=cfg.weight_decay,
                                 outlier_threshold=cfg.outlier_threshold)
    elif cfg.name == 'adalr_lion':
        return DecoupledAdaLRLion(model.parameters(),
                                  lr=cfg.lr,
                                  betas=cfg.betas,
                                  weight_decay=cfg.weight_decay,
                                  outlier_threshold=cfg.outlier_threshold,
                                  timeout=cfg.timeout,
                                  lr_penalty=cfg.lr_penalty,
                                  min_scale=cfg.min_scale)
    else:
        raise ValueError(f'Not sure how to build optimizer: {cfg.name}')


def build_scheduler(cfg):
    if cfg.name == 'constant_with_warmup':
        return ConstantWithWarmupScheduler(t_warmup=cfg.t_warmup)
    elif cfg.name == 'cosine_with_warmup':
        return CosineAnnealingWithWarmupScheduler(t_warmup=cfg.t_warmup,
                                                  alpha_f=cfg.alpha_f)
    elif cfg.name == 'linear_decay_with_warmup':
        return LinearWithWarmupScheduler(t_warmup=cfg.t_warmup,
                                         alpha_f=cfg.alpha_f)
    else:
        raise ValueError(f'Not sure how to build scheduler: {cfg.name}')


def build_tokenizer(om_tokenizer_config: DictConfig,) -> Tokenizer:
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    resolved_om_tokenizer_config = om.to_container(om_tokenizer_config,
                                                   resolve=True)
    tokenizer_kwargs = resolved_om_tokenizer_config.get(  # type: ignore
        'kwargs', {})
    tokenizer_name = resolved_om_tokenizer_config['name']  # type: ignore
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,
                                              **tokenizer_kwargs)

    # HuggingFace does not respect the model_max_length kwarg, and overrides it with
    # min(kwargs['model_max_length'], original_config['model_max_length']), so we
    # explicitly set it here
    tokenizer.model_max_length = tokenizer_kwargs.get(
        'model_max_length',
        int(1e30),
    )

    return tokenizer


def build_icl_evaluators(icl_tasks,
                         tokenizer,
                         default_max_seq_len,
                         default_batch_size,
                         destination_dir=os.getcwd()):
    evaluators = []
    logger_keys = []
    if isinstance(icl_tasks, str):
        print(f'Extracting ICL task config from path: {icl_tasks}')
        with open(icl_tasks, 'r') as icl_f:
            icl_task_cfg = om.load(icl_f)
        icl_tasks = icl_task_cfg.icl_tasks

    def _validate_cfg(icl_cfg):
        assert 'label' in icl_cfg
        assert 'dataset_uri' in icl_cfg and icl_cfg.dataset_uri is not None
        assert 'icl_task_type' in icl_cfg
        assert 'num_fewshot' in icl_cfg

        if 'metric_names' not in icl_cfg:
            if icl_cfg.icl_task_type == 'language_modeling':
                icl_cfg.metric_names = ['InContextLearningLMAccuracy']
            elif icl_cfg.icl_task_type == 'multiple_choice':
                icl_cfg.metric_names = [
                    'InContextLearningMultipleChoiceAccuracy'
                ]
            elif icl_cfg.icl_task_type == 'schema':
                icl_cfg.metric_names = [
                    'InContextLearningMultipleChoiceAccuracy'
                ]
            elif icl_cfg.icl_task_type == 'question_answering':
                icl_cfg.metric_names = ['InContextLearningQAAccuracy']
            else:
                raise ValueError(
                    f'No metric_names defined, unable to build default metrics for icl_task_type={icl_cfg.icl_task_type}.'
                )

        if 'prompt_string' not in icl_cfg:
            icl_cfg.prompt_string = ''
        if 'example_delimiter' not in icl_cfg:
            icl_cfg.example_delimiter = '\n'
        if 'continuation_delimiter' not in icl_cfg:
            icl_cfg.continuation_delimiter = ' '
        if 'max_seq_len' not in icl_cfg:
            icl_cfg.max_seq_len = default_max_seq_len
        if 'batch_size' not in icl_cfg:
            icl_cfg.batch_size = default_batch_size

    for icl_cfg in icl_tasks:
        _validate_cfg(icl_cfg)
        for num_fewshot in list(icl_cfg.num_fewshot):
            if tokenizer.pad_token_id is None:
                # Current workaround to support GPT2 tokenizer with `pad_token_id = None`
                pad_tok_id = tokenizer.eos_token_id
            else:
                pad_tok_id = tokenizer.pad_token_id
            label = f'{icl_cfg.label}/{num_fewshot}-shot'
            metric_names = list(icl_cfg.metric_names)
            # TODO: fix Composer bug when copying local paths and destination exists
            destination_path = f'{destination_dir}/{icl_cfg.label}-{num_fewshot}.jsonl'
            with dist.run_local_rank_zero_first():
                if os.path.exists(destination_path):
                    os.remove(destination_path)
            dataloaders = get_icl_task_dataloader(
                icl_cfg.icl_task_type,
                icl_cfg.dataset_uri,
                tokenizer,
                batch_size=icl_cfg.batch_size,
                max_seq_len=icl_cfg.max_seq_len,
                pad_tok_id=pad_tok_id,
                num_fewshot=num_fewshot,
                prompt_string=icl_cfg.prompt_string,
                example_delimiter=icl_cfg.example_delimiter,
                continuation_delimiter=icl_cfg.continuation_delimiter,
                destination_path=destination_path,
                has_categories=icl_cfg.get('has_categories', False),
            )
            if hasattr(
                    icl_cfg,
                    'has_categories') and icl_cfg.has_categories and isinstance(
                        dataloaders, dict):
                for category in dataloaders.keys():
                    logger_keys.extend([
                        f'metrics/{label}/{category}/{m}' for m in metric_names
                    ])
                    evaluators.append(
                        Evaluator(label=f'{label}/{category}',
                                  dataloader=dataloaders[category],
                                  metric_names=metric_names),)
            else:
                logger_keys.extend(
                    [f'metrics/{label}/{m}' for m in metric_names])
                evaluators.append(
                    Evaluator(label=label,
                              dataloader=dataloaders,
                              metric_names=metric_names),)

    return evaluators, logger_keys
