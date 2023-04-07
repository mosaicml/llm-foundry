# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import os

from composer import algorithms
from composer.callbacks import (HealthChecker, LRMonitor, MemoryMonitor,
                                OptimizerMonitor, RuntimeEstimator,
                                SpeedMonitor)
from composer.core import Evaluator
from composer.datasets.in_context_learning_evaluation import \
    get_icl_task_dataloader
from composer.loggers import WandBLogger
from composer.optim import DecoupledAdamW
from composer.optim.scheduler import (ConstantWithWarmupScheduler,
                                      CosineAnnealingWithWarmupScheduler,
                                      LinearWithWarmupScheduler)

from examples.common.fdiff import FDiffMetrics
from examples.common.generate_callback import Generate
from examples.common.monolithic_ckpt_callback import MonolithicCheckpointSaver
from examples.common.optim import (DecoupledAdaLRLion, DecoupledClipLion,
                                   DecoupledLionW)
from examples.common.resumption_callbacks import GlobalLRScaling, LayerFreezing
from examples.common.text_data import build_text_dataloader


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
    elif name == 'health_checker':
        return HealthChecker(**kwargs)
    elif name == 'generate_callback':
        prompts = kwargs.pop('prompts')
        return Generate(prompts=list(prompts), **kwargs)
    elif name == 'global_lr_scaling':
        return GlobalLRScaling(**kwargs)
    elif name == 'layer_freezing':
        return LayerFreezing(**kwargs)
    elif name == 'mono_ckpt_saver':
        return MonolithicCheckpointSaver(**kwargs)
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


def build_dataloader(cfg, device_batch_size):
    if cfg.name == 'text':
        return build_text_dataloader(cfg, device_batch_size)
    else:
        raise ValueError(f'Not sure how to build dataloader with config: {cfg}')


def build_icl_evaluators(cfg, tokenizer):
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    evaluators = []
    logger_keys = []

    def _validate_cfg(icl_cfg):
        assert 'dataset_uri' in icl_cfg and icl_cfg.dataset_uri is not None
        assert 'icl_task_type' in icl_cfg
        assert 'num_fewshot' in icl_cfg
        assert 'batch_size' in icl_cfg
        assert 'metric_names' in icl_cfg
        assert 'prompt_string' in icl_cfg
        assert 'example_delimiter' in icl_cfg
        assert 'continuation_delimiter' in icl_cfg
        assert 'label' in icl_cfg

    for icl_cfg in cfg.icl_tasks:
        _validate_cfg(icl_cfg)
        for num_fewshot in list(icl_cfg.num_fewshot):
            if tokenizer.pad_token_id is None:
                # Current workaround to support GPT2 tokenizer with `pad_token_id = None`
                pad_tok_id = tokenizer.eos_token_id
            else:
                pad_tok_id = tokenizer.pad_token_id
            label = f'{icl_cfg.label}/{num_fewshot}-shot'
            metric_names = list(icl_cfg.metric_names)
            dataloader = get_icl_task_dataloader(
                icl_cfg.icl_task_type,
                icl_cfg.dataset_uri,
                tokenizer,
                batch_size=icl_cfg.batch_size,
                max_seq_len=tokenizer.model_max_length,
                pad_tok_id=pad_tok_id,
                num_fewshot=num_fewshot,
                prompt_string=icl_cfg.prompt_string,
                example_delimiter=icl_cfg.example_delimiter,
                continuation_delimiter=icl_cfg.continuation_delimiter,
                destination_path=f'{icl_cfg.label}-{num_fewshot}.jsonl',
            )
            logger_keys.extend([f'metrics/{label}/{m}' for m in metric_names])
            evaluators.append(
                Evaluator(label=label,
                          dataloader=dataloader,
                          metric_names=metric_names))

    return evaluators, logger_keys
