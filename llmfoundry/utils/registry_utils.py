# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import functools
import importlib.util
import os
from pathlib import Path
from types import ModuleType
from typing import (Any, Callable, Dict, Generic, Optional, Sequence, Type,
                    TypeVar, Union)

import catalogue

__all__ = ['TypedRegistry', 'create', 'builder']

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
        generic_type (Type[S]): The type of the registry.
        description (str): A description of the registry.

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


def import_file(loc: Union[str, Path]) -> ModuleType:
    """Import module from a file.

    Used to run arbitrary python code.
    Args:
        name (str): Name of module to load.
        loc (str / Path): Path to the file.

    Returns:
        ModuleType: The module object.
    """
    if not os.path.exists(loc):
        raise FileNotFoundError(f'File {loc} does not exist.')

    spec = importlib.util.spec_from_file_location('python_code', str(loc))

    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)

    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f'Error executing {loc}') from e
    return module
