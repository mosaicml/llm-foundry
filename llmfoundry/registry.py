# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Callable, Iterable, Union

from composer.core import Algorithm, Callback, DataSpec
from composer.loggers import LoggerDestination
from composer.models import ComposerModel
from composer.optim import ComposerScheduler
from torch.distributed.checkpoint import LoadPlanner, SavePlanner
from torch.distributed.tensor.parallel.style import ParallelStyle
from torch.optim import Optimizer
from torch.utils.data import DataLoader as TorchDataloader
from torch.utils.data import Dataset
from torchmetrics import Metric
from transformers import PreTrainedTokenizerBase

from llmfoundry.interfaces import CallbackWithConfig
from llmfoundry.layers_registry import (
    attention_classes,
    attention_implementations,
    fcs,
    ffns,
    ffns_with_megablocks,
    ffns_with_norm,
    module_init_fns,
    norms,
    param_init_fns,
)
from llmfoundry.utils.registry_utils import create_registry

_loggers_description = (
    """The loggers registry is used to register classes that implement the LoggerDestination interface.

    These classes are used to log data from the training loop, and will be passed to the loggers arg of the Trainer. The loggers
    will be constructed by directly passing along the specified kwargs to the constructor. See loggers/ for examples.

    Args:
        kwargs (Dict[str, Any]): The kwargs to pass to the LoggerDestination constructor.

    Returns:
        LoggerDestination: The logger destination.
    """
)
loggers = create_registry(
    'llmfoundry',
    'loggers',
    generic_type=type[LoggerDestination],
    entry_points=True,
    description=_loggers_description,
)

_callbacks_description = (
    """The callbacks registry is used to register classes that implement the Callback interface.

    These classes are used to interact with the Composer event system, and will be passed to the callbacks arg of the Trainer.
    The callbacks will be constructed by directly passing along the specified kwargs to the constructor. See callbacks/ for examples.

    Args:
        kwargs (Dict[str, Any]): The kwargs to pass to the Callback constructor.

    Returns:
        Callback: The callback.
    """
)
callbacks = create_registry(
    'llmfoundry',
    'callbacks',
    generic_type=type[Callback],
    entry_points=True,
    description=_callbacks_description,
)

_callbacks_with_config_description = (
    """The callbacks_with_config registry is used to register classes that implement the CallbackWithConfig interface.

    These are the same as the callbacks registry, except that they additionally take the full training config as an argument to their constructor.
    See callbacks/ for examples.

    Args:
        config (DictConfig): The training config.
        kwargs (Dict[str, Any]): The kwargs to pass to the Callback constructor.

    Returns:
        Callback: The callback.
    """
)
callbacks_with_config = create_registry(
    'llmfoundry',
    'callbacks_with_config',
    generic_type=type[CallbackWithConfig],
    entry_points=True,
    description=_callbacks_with_config_description,
)

_optimizers_description = (
    """The optimizers registry is used to register classes that implement the Optimizer interface.

    The optimizer will be passed to the optimizers arg of the Trainer. The optimizer will be constructed by directly passing along the
    specified kwargs to the constructor, along with the model parameters. See optim/ for examples.

    Args:
        params (Iterable[torch.nn.Parameter]): The model parameters.
        kwargs (Dict[str, Any]): The kwargs to pass to the Optimizer constructor.

    Returns:
        Optimizer: The optimizer.
    """
)
optimizers = create_registry(
    'llmfoundry',
    'optimizers',
    generic_type=type[Optimizer],
    entry_points=True,
    description=_optimizers_description,
)

_algorithms_description = (
    """The algorithms registry is used to register classes that implement the Algorithm interface.

    The algorithm will be passed to the algorithms arg of the Trainer. The algorithm will be constructed by directly passing along the
    specified kwargs to the constructor. See algorithms/ for examples.

    Args:
        kwargs (Dict[str, Any]): The kwargs to pass to the Algorithm constructor.

    Returns:
        Algorithm: The algorithm.
    """
)
algorithms = create_registry(
    'llmfoundry',
    'algorithms',
    generic_type=type[Algorithm],
    entry_points=True,
    description=_algorithms_description,
)

_schedulers_description = (
    """The schedulers registry is used to register classes that implement the ComposerScheduler interface.

    The scheduler will be passed to the schedulers arg of the Trainer. The scheduler will be constructed by directly passing along the
    specified kwargs to the constructor. See optim/ for examples.

    Args:
        kwargs (Dict[str, Any]): The kwargs to pass to the ComposerScheduler constructor.

    Returns:
        ComposerScheduler: The scheduler.
    """
)
schedulers = create_registry(
    'llmfoundry',
    'schedulers',
    generic_type=type[ComposerScheduler],
    entry_points=True,
    description=_schedulers_description,
)

_tokenizers_description = (
    'The tokenizers registry is used to register tokenizers that implement the transformers.PreTrainedTokenizerBase interface. '
    +
    'The tokenizer will be passed to the build_dataloader() and build_composer_model() methods in train.py.'
)
tokenizers = create_registry(
    'llmfoundry',
    'tokenizers',
    generic_type=type[PreTrainedTokenizerBase],
    entry_points=True,
    description=_tokenizers_description,
)

_models_description = (
    """The models registry is used to register classes that implement the ComposerModel interface.

    The model constructor should accept a PreTrainedTokenizerBase named `tokenizer`, and the rest of its constructor kwargs.
    See models/ for examples.

    Args:
        tokenizer (PreTrainedTokenizerBase): The tokenizer.
        kwargs (Dict[str, Any]): The kwargs to pass to the Composer

    Returns:
        ComposerModel: The model.
    """
)
models = create_registry(
    'llmfoundry',
    'models',
    generic_type=type[ComposerModel],
    entry_points=True,
    description=_models_description,
)

_dataloaders_description = (
    """The dataloaders registry is used to register functions that create a DataSpec given a config.

    The function should take a PreTrainedTokenizerBase, a device batch size, and the rest of its constructor kwargs,
    and return a DataSpec. See data/ for examples.

    Args:
        tokenizer (PreTrainedTokenizerBase): The tokenizer
        device_batch_size (Union[int, float]): The device batch size.
        kwargs (Dict[str, Any]): The kwargs to pass to the builder function.

    Returns:
        DataSpec: The dataspec.
    """
)
dataloaders = create_registry(
    'llmfoundry',
    'dataloaders',
    generic_type=Callable[
        ...,
        DataSpec],  # The arguments to the dataloader may vary depending on the contents of the config.
    entry_points=True,
    description=_dataloaders_description,
)

_dataset_replication_validators_description = (
    """The dataset_replication_validators registry is used to register functions that validate replication factor.

    The function should return the replication factor and the dataset device batch size. See data/ for examples.

    Args:
        cfg (DictConfig): The dataloader config.
        tokenizer (PreTrainedTokenizerBase): The tokenizer
        device_batch_size (Union[int, float]): The device batch size.

    Returns:
        replication_factor (int): The replication factor for dataset.
        dataset_batch_size (int): The dataset device batch size.
    """
)
dataset_replication_validators = create_registry(
    'llmfoundry',
    'dataset_replication_validators',
    generic_type=Callable[
        [dict[str, Any], PreTrainedTokenizerBase, Union[int, float]],
        tuple[int, int]],
    entry_points=True,
    description=_dataset_replication_validators_description,
)

_collators_description = (
    """The collators registry is used to register functions that create the collate function for the DataLoader.

    See data/ for examples.

    Args:
        cfg (DictConfig): The dataloader config.
        tokenizer (PreTrainedTokenizerBase): The tokenizer
        dataset_batch_size (Union[int, float]): The dataset device batch size.

    Returns:
        collate_fn  (Any): The collate function.
        dataloader_batch_size (int): The batch size for dataloader. In case of packing, this might be the packing ratio times the dataset device batch size.
    """
)
collators = create_registry(
    'llmfoundry',
    'collators',
    generic_type=Callable[[dict[str, Any], PreTrainedTokenizerBase, int],
                          tuple[Any, int]],
    entry_points=True,
    description=_collators_description,
)

_data_specs_description = (
    """The data_specs registry is used to register functions that create a DataSpec given a dataloader.

    See data/ for examples.

    Args:
        dl (Union[Iterable, TorchDataloader): The dataloader.
        dataset_cfg (DictConfig): The dataset config.

    Returns:
        dataspec (DataSpec): The dataspec.
    """
)
data_specs = create_registry(
    'llmfoundry',
    'data_specs',
    generic_type=Callable[[Union[Iterable, TorchDataloader], dict[str, Any]],
                          DataSpec],
    entry_points=True,
    description=_data_specs_description,
)

_metrics_description = (
    """The metrics registry is used to register classes that implement the torchmetrics.Metric interface.

    The metric will be passed to the metrics arg of the Trainer. The metric will be constructed by directly passing along the
    specified kwargs to the constructor. See metrics/ for examples.

    Args:
        kwargs (Dict[str, Any]): The kwargs to pass to the Metric constructor.

    Returns:
        Metric: The metric.
    """
)
metrics = create_registry(
    'llmfoundry',
    'metrics',
    generic_type=type[Metric],
    entry_points=True,
    description=_metrics_description,
)

_icl_datasets_description = (
    """The ICL datasets registry is used to register classes that implement the InContextLearningDataset interface.

    The dataset will be constructed along with an Evaluator. The dataset will be constructed by directly passing along the
    specified kwargs to the constructor. See eval/ for examples.

    Args:
        kwargs (Dict[str, Any]): The kwargs to pass to the Dataset constructor.

    Returns:
        InContextLearningDataset: The dataset.
    """
)
icl_datasets = create_registry(
    'llmfoundry',
    'icl_datasets',
    # TODO: Change type from Dataset to
    # llmfoundry.eval.InContextLearningDataset.
    # Using ICL dataset here introduces a circular import dependency between
    # the registry and eval packages right now, thus needs some refactoring.
    generic_type=type[Dataset],
    entry_points=True,
    description=_icl_datasets_description,
)

_config_transforms_description = (
    """The config_transforms registry is used to register functions that transform the training config

    The config will be transformed before it is used anywhere else. Note: By default ALL registered transforms will be applied to the train config
    and NONE to the eval config. Each transform should return the modified config. See utils/config_utils.py for examples.

    Args:
        cfg (Dict[str, Any]): The training config.

    Returns:
        cfg (Dict[str, Any]): The modified training config.
    """
)
config_transforms = create_registry(
    'llmfoundry',
    'config_transforms',
    generic_type=Callable[[dict[str, Any]], dict[str, Any]],
    entry_points=True,
    description=_config_transforms_description,
)

_load_planners_description = (
    """The load_planners registry is used to register classes that implement the LoadPlanner interface.

    The LoadPlanner will be passed as part of the FSDP config arg of the Trainer. It will be used to load distributed checkpoints.

    Returns:
        LoadPlanner: The load planner.
    """
)

load_planners = create_registry(
    'llmfoundry',
    'load_planners',
    generic_type=type[LoadPlanner],
    entry_points=True,
    description=_load_planners_description,
)

_save_planners_description = (
    """The save_planners registry is used to register classes that implement the SavePlanner interface.

    The savePlanner will be passed as part of the FSDP config arg of the Trainer. It will be used to save distributed checkpoints.

    Returns:
        SavePlanner: The save planner.
    """
)

save_planners = create_registry(
    'llmfoundry',
    'save_planners',
    generic_type=type[SavePlanner],
    entry_points=True,
    description=_save_planners_description,
)

_tp_strategies_description = (
    """The tp_strategies registry is used to register strategies for tensor parallelism.

    Args:
        model (ComposerModel): The model.

    Returns:
        layer_plan (Dict[str, ParallelStyle]): The plan used to parallelize the model.
        model (ComposerModel): The model.
    """
)

tp_strategies = create_registry(
    'llmfoundry',
    'tp_strategies',
    generic_type=Callable[[ComposerModel], dict[str, ParallelStyle]],
    entry_points=True,
    description=_tp_strategies_description,
)

__all__ = [
    'loggers',
    'callbacks',
    'callbacks_with_config',
    'optimizers',
    'algorithms',
    'schedulers',
    'tokenizers',
    'models',
    'dataset_replication_validators',
    'collators',
    'data_specs',
    'metrics',
    'dataloaders',
    'norms',
    'param_init_fns',
    'module_init_fns',
    'ffns',
    'ffns_with_norm',
    'ffns_with_megablocks',
    'attention_classes',
    'attention_implementations',
    'fcs',
    'icl_datasets',
    'config_transforms',
    'load_planners',
    'save_planners',
    'tp_strategies',
]
