# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import functools
from typing import Any, Callable, Dict, Optional, Union

import catalogue
from composer.algorithms import (Alibi, GatedLinearUnits, GradientClipping,
                                 LowPrecisionLayerNorm)
from composer.callbacks import (EarlyStopper, Generate, LRMonitor,
                                MemoryMonitor, MemorySnapshot, OptimizerMonitor,
                                RuntimeEstimator, SpeedMonitor)
from composer.loggers import (InMemoryLogger, MLFlowLogger, TensorboardLogger,
                              WandBLogger)
from composer.optim import (ConstantWithWarmupScheduler,
                            CosineAnnealingWithWarmupScheduler, DecoupledAdamW,
                            LinearWithWarmupScheduler)

from llmfoundry.callbacks import (AsyncEval, CurriculumLearning, FDiffMetrics,
                                  GlobalLRScaling, HuggingFaceCheckpointer,
                                  LayerFreezing, MonolithicCheckpointSaver,
                                  ScheduledGarbageCollector)
from llmfoundry.optim import (DecoupledAdaLRLion, DecoupledClipLion,
                              DecoupledLionW, DecoupledLionW_8bit)
from llmfoundry.optim.scheduler import InverseSquareRootWithWarmupScheduler

loggers = catalogue.create('llm_foundry.loggers', entry_points=True)
loggers.register('wandb', WandBLogger)
loggers.register('tensorboard', TensorboardLogger)
loggers.register('inmemory', InMemoryLogger)
loggers.register('in_memory_logger',
                 InMemoryLogger)  # for backwards compatibility
loggers.register('mlflow', MLFlowLogger)

callbacks = catalogue.create('llm_foundry.callbacks', entry_points=True)
callbacks.register('lr_monitor', LRMonitor)
callbacks.register('memory_monitor', MemoryMonitor)
callbacks.register('memory_snapshot', MemorySnapshot)
callbacks.register('speed_monitor', SpeedMonitor)
callbacks.register('runtime_estimator', RuntimeEstimator)
callbacks.register('optimizer_monitor', OptimizerMonitor)
callbacks.register('generate_callback', Generate)
callbacks.register('early_stopper', EarlyStopper)
callbacks.register('fdiff_metrics', FDiffMetrics)
callbacks.register('huggingface_checkpointer', HuggingFaceCheckpointer)
callbacks.register('global_lr_scaling', GlobalLRScaling)
callbacks.register('layer_freezing', LayerFreezing)
callbacks.register('mono_checkpoint_saver', MonolithicCheckpointSaver)
callbacks.register('scheduled_garbage_collector', ScheduledGarbageCollector)

callbacks_with_config = catalogue.create('llm_foundry.callbacks_with_config',
                                         entry_points=True)
callbacks_with_config.register('async_eval', AsyncEval)
callbacks_with_config.register('curriculum_learning', CurriculumLearning)

optimizers = catalogue.create('llm_foundry.optimizers', entry_points=True)
optimizers.register('adalr_lion', DecoupledAdaLRLion)
optimizers.register('clip_lion', DecoupledClipLion)
optimizers.register('decoupled_lionw', DecoupledLionW)
optimizers.register('decoupled_lionw_8b', DecoupledLionW_8bit)
optimizers.register('decoupled_adamw', DecoupledAdamW)

algorithms = catalogue.create('llm_foundry.algorithms', entry_points=True)
algorithms.register('gradient_clipping', GradientClipping)
algorithms.register('alibi', Alibi)
algorithms.register('gated_linear_units', GatedLinearUnits)
algorithms.register('low_precision_layernorm', LowPrecisionLayerNorm)

schedulers = catalogue.create('llm_foundry.schedulers', entry_points=True)
schedulers.register('constant_with_warmup', ConstantWithWarmupScheduler)
schedulers.register('cosine_with_warmup', CosineAnnealingWithWarmupScheduler)
schedulers.register('linear_decay_with_warmup', LinearWithWarmupScheduler)
schedulers.register('inv_sqrt_with_warmup',
                    InverseSquareRootWithWarmupScheduler)


def contains(
    name: str,
    registry: catalogue.Registry,
) -> bool:
    return name in registry


def builder(
    name: str,
    registry: catalogue.Registry,
    pre_validation_function: Optional[Union[Callable[[Any], None], type]],
    post_validation_function: Optional[Callable[[Any], None]],
    kwargs: Dict[str, Any],
) -> Any:
    registered_item = registry.get(name)

    if pre_validation_function is not None:
        if isinstance(pre_validation_function, Callable):
            pre_validation_function(registered_item)
        elif isinstance(pre_validation_function, type):
            if not issubclass(registered_item, pre_validation_function):
                raise ValueError(
                    f'Expected {name} to be of type {pre_validation_function}, but got {type(registered_item)}'
                )
        else:
            raise ValueError(
                f'Expected pre_validation_function to be a callable or a type, but got {type(pre_validation_function)}'
            )

    # If it is a class, construct the class with kwargs
    # If it is a function, create a partial with kwargs
    if isinstance(registered_item, type):
        constructed_item = registered_item(**kwargs)
    elif callable(registered_item):
        constructed_item = functools.partial(registered_item, **kwargs)
    else:
        raise ValueError(
            f'Expected {name} to be a class or function, but got {type(registered_item)}'
        )

    if post_validation_function is not None:
        post_validation_function(registered_item)

    return constructed_item
