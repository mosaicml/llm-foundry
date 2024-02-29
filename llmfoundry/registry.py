# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import functools
from typing import Any, Callable, Dict, Generic, Optional, Type, TypeVar, Union

import catalogue
from composer.algorithms import (Alibi, GatedLinearUnits, GradientClipping,
                                 LowPrecisionLayerNorm)
from composer.callbacks import (EarlyStopper, Generate, LRMonitor,
                                MemoryMonitor, MemorySnapshot, OOMObserver,
                                OptimizerMonitor, RuntimeEstimator,
                                SpeedMonitor)
from composer.core import Algorithm, Callback
from composer.loggers import LoggerDestination
from composer.optim import (ComposerScheduler, ConstantWithWarmupScheduler,
                            CosineAnnealingWithWarmupScheduler, DecoupledAdamW,
                            LinearWithWarmupScheduler)
from torch.optim import Optimizer

from llmfoundry.callbacks import (AsyncEval, CurriculumLearning, FDiffMetrics,
                                  GlobalLRScaling, HuggingFaceCheckpointer,
                                  LayerFreezing, MonolithicCheckpointSaver,
                                  ScheduledGarbageCollector)
from llmfoundry.optim import (DecoupledAdaLRLion, DecoupledClipLion,
                              DecoupledLionW)
from llmfoundry.optim.scheduler import InverseSquareRootWithWarmupScheduler

T = TypeVar('T')
S = TypeVar('S')


class TypedRegistry(catalogue.Registry, Generic[T]):
    """A thin wrapper around catalogue.Registry to add static typing."""

    def __call__(self, name: str, func: T) -> Callable[[T], T]:
        return super().__call__(name, func)

    def register(self, name: str, func: T) -> T:
        return super().__call__(name, func=func)

    def get(self, name: str) -> T:
        return super().get(name)

    def get_all(self) -> Dict[str, T]:
        return super().get_all()

    def get_entry_point(self, name: str, default: Optional[T] = None) -> T:
        return super().get_entry_point(name, default=default)

    def get_entry_points(self) -> Dict[str, T]:
        return super().get_entry_points()


def create(
    namespace: str,
    generic_type: Type[S],
    entry_points: bool = False,
) -> 'TypedRegistry[S]':
    """Create a new registry.

    Args:
        namespace (str): The namespace, e.g. "llmfoundry.loggers"
        entry_points (bool): Accept registered functions from entry points.

    Returns:
        The TypedRegistry object.
    """
    if catalogue.check_exists(*namespace):
        raise catalogue.RegistryError(f'Namespace already exists: {namespace}')

    return TypedRegistry[generic_type](namespace, entry_points=entry_points)


def builder(
    name: str,
    registry: catalogue.Registry,
    partial_function: bool = True,
    pre_validation_function: Optional[Union[Callable[[Any], None],
                                            type]] = None,
    post_validation_function: Optional[Callable[[Any], None]] = None,
    **kwargs: Dict[str, Any],
) -> Any:
    """Helper function to build an item from the registry.

    Args:
        name (str): The name of the registered item
        registry (catalogue.Registry): The registry to fetch the item from
        partial_function (bool, optional): Whether to return a partial function for registered callables. Defaults to True.
        pre_validation_function (Optional[Union[Callable[[Any], None], type]], optional): An optional validation function called
            before constructing the item to return. Defaults to None.
        post_validation_function (Optional[Callable[[Any], None]], optional): An optional validation function called after
            constructing the item to return. Defaults to None.

    Raises:
        ValueError: If the validation functions failed or the registered item is invalid

    Returns:
        Any: The constructed item from the registry
    """
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

    # If it is a class, or a builder function, construct the class with kwargs
    # If it is a function, create a partial with kwargs
    if isinstance(registered_item,
                  type) or callable(registered_item) and not partial_function:
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


loggers = create('llm_foundry.loggers',
                 generic_type=Type[LoggerDestination],
                 entry_points=True)
callbacks = create('llm_foundry.callbacks', Type[Callback], entry_points=True)
callbacks_with_config = create('llm_foundry.callbacks_with_config',
                               Type[Callback],
                               entry_points=True)
optimizers = catalogue.create('llm_foundry.optimizers',
                              Type[Optimizer],
                              entry_points=True)
algorithms = catalogue.create('llm_foundry.algorithms',
                              Type[Algorithm],
                              entry_points=True)
schedulers = catalogue.create('llm_foundry.schedulers',
                              Type[ComposerScheduler],
                              entry_points=True)

callbacks.register('lr_monitor', func=LRMonitor)
callbacks.register('memory_monitor', func=MemoryMonitor)
callbacks.register('memory_snapshot', func=MemorySnapshot)
callbacks.register('speed_monitor', func=SpeedMonitor)
callbacks.register('runtime_estimator', func=RuntimeEstimator)
callbacks.register('optimizer_monitor', func=OptimizerMonitor)
callbacks.register('generate_callback', func=Generate)
callbacks.register('early_stopper', func=EarlyStopper)
callbacks.register('fdiff_metrics', func=FDiffMetrics)
callbacks.register('huggingface_checkpointer', func=HuggingFaceCheckpointer)
callbacks.register('global_lr_scaling', func=GlobalLRScaling)
callbacks.register('layer_freezing', func=LayerFreezing)
callbacks.register('mono_checkpoint_saver', func=MonolithicCheckpointSaver)
callbacks.register('scheduled_garbage_collector',
                   func=ScheduledGarbageCollector)
callbacks.register('oom_observer', func=OOMObserver)

callbacks_with_config.register('async_eval', func=AsyncEval)
callbacks_with_config.register('curriculum_learning', func=CurriculumLearning)

optimizers.register('adalr_lion', func=DecoupledAdaLRLion)
optimizers.register('clip_lion', func=DecoupledClipLion)
optimizers.register('decoupled_lionw', func=DecoupledLionW)
optimizers.register('decoupled_adamw', func=DecoupledAdamW)

algorithms.register('gradient_clipping', func=GradientClipping)
algorithms.register('alibi', func=Alibi)
algorithms.register('gated_linear_units', func=GatedLinearUnits)
algorithms.register('low_precision_layernorm', func=LowPrecisionLayerNorm)

schedulers.register('constant_with_warmup', func=ConstantWithWarmupScheduler)
schedulers.register('cosine_with_warmup',
                    func=CosineAnnealingWithWarmupScheduler)
schedulers.register('linear_decay_with_warmup', func=LinearWithWarmupScheduler)
schedulers.register('inv_sqrt_with_warmup',
                    func=InverseSquareRootWithWarmupScheduler)
