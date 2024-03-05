# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import functools
from typing import (Any, Callable, Dict, Generic, Optional, Sequence, Type,
                    TypeVar, Union)

import catalogue
from composer.core import Algorithm, Callback
from composer.loggers import LoggerDestination
from composer.optim import ComposerScheduler
from torch.optim import Optimizer

from llmfoundry.interfaces import CallbackWithConfig

T = TypeVar('T')
S = TypeVar('S')


class TypedRegistry(catalogue.Registry, Generic[T]):
    """A thin wrapper around catalogue.Registry to add static typing."""

    def __init__(self,
                 namespace: Sequence[str],
                 entry_points: bool = False,
                 description: str = '') -> None:
        super().__init__(namespace, entry_points=entry_points)

        self.description = description

    def __call__(self, name: str, func: Optional[T] = None) -> Callable[[T], T]:
        return super().__call__(name, func)

    def register(self, name: str, *, func: Optional[T] = None) -> T:
        return super().register(name, func=func)

    def get(self, name: str) -> T:
        return super().get(name)

    def get_all(self) -> Dict[str, T]:
        return super().get_all()

    def get_entry_point(self, name: str, default: Optional[T] = None) -> T:
        return super().get_entry_point(name, default=default)

    def get_entry_points(self) -> Dict[str, T]:
        return super().get_entry_points()


def create(
    *namespace: str,
    generic_type: Type[S],
    entry_points: bool = False,
    description: str = '',
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

    return TypedRegistry[generic_type](namespace,
                                       entry_points=entry_points,
                                       description=description)


def builder(
    name: str,
    registry: catalogue.Registry,
    partial_function: bool = True,
    pre_validation_function: Optional[Union[Callable[[Any], None],
                                            type]] = None,
    post_validation_function: Optional[Callable[[Any], None]] = None,
    kwargs: Optional[Dict[str, Any]] = None,
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
    if kwargs is None:
        kwargs = {}

    registered_item = registry.get(name)

    if pre_validation_function is not None:
        if isinstance(pre_validation_function, type):
            if not issubclass(registered_item, pre_validation_function):
                raise ValueError(
                    f'Expected {name} to be of type {pre_validation_function}, but got {type(registered_item)}'
                )
        elif isinstance(pre_validation_function, Callable):
            pre_validation_function(registered_item)
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


_loggers_description = """The loggers registry is used to register classes that implement the LoggerDestination interface.
These classes are used to log data from the training loop, and will be passed to the loggers arg of the Trainer. The loggers
will be constructed by directly passing along the specified kwargs to the constructor."""
loggers = create('llmfoundry',
                 'loggers',
                 generic_type=Type[LoggerDestination],
                 entry_points=True,
                 description=_loggers_description)

_callbacks_description = """The callbacks registry is used to register classes that implement the Callback interface.
These classes are used to interact with the Composer event system, and will be passed to the callbacks arg of the Trainer.
The callbacks will be constructed by directly passing along the specified kwargs to the constructor."""
callbacks = create('llmfoundry',
                   'callbacks',
                   generic_type=Type[Callback],
                   entry_points=True,
                   description=_callbacks_description)

_callbacks_with_config_description = """The callbacks_with_config registry is used to register classes that implement the CallbackWithConfig interface.
These are the same as the callbacks registry, except that they additionally take the full training config as an argument to their constructor."""
callbacks_with_config = create('llm_foundry.callbacks_with_config',
                               generic_type=Type[CallbackWithConfig],
                               entry_points=True,
                               description=_callbacks_with_config_description)

_optimizers_description = """The optimizers registry is used to register classes that implement the Optimizer interface.
The optimizer will be passed to the optimizers arg of the Trainer. The optimizer will be constructed by directly passing along the
specified kwargs to the constructor, along with the model parameters."""
optimizers = create('llmfoundry',
                    'optimizers',
                    generic_type=Type[Optimizer],
                    entry_points=True,
                    description=_optimizers_description)

_algorithms_description = """The algorithms registry is used to register classes that implement the Algorithm interface.
The algorithm will be passed to the algorithms arg of the Trainer. The algorithm will be constructed by directly passing along the
specified kwargs to the constructor."""
algorithms = create('llmfoundry',
                    'algorithms',
                    generic_type=Type[Algorithm],
                    entry_points=True,
                    description=_algorithms_description)

_schedulers_description = """The schedulers registry is used to register classes that implement the ComposerScheduler interface.
The scheduler will be passed to the schedulers arg of the Trainer. The scheduler will be constructed by directly passing along the
specified kwargs to the constructor."""
schedulers = create('llmfoundry',
                    'schedulers',
                    generic_type=Type[ComposerScheduler],
                    entry_points=True,
                    description=_schedulers_description)

__all__ = [
    'loggers',
    'callbacks',
    'callbacks_with_config',
    'optimizers',
    'algorithms',
    'schedulers',
    'create',
    'builder',
    'TypedRegistry',
]
