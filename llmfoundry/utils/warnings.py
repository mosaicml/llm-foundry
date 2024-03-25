# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
import functools
import warnings
from typing import Any, Callable, Type, TypeVar, cast

__all__ = [
    'VersionedDeprecationWarning',
]


class VersionedDeprecationWarning(UserWarning):
    """A custom deprecation warning class that includes version information.

    Attributes:
        message (str): The deprecation message describing why the feature is deprecated.
        remove_version (str): The version in which the feature will be removed.

    Example:
        >>> def deprecated_function():
        ...     warnings.warn(
        ...         VersionedDeprecationWarning(
        ...             "Function XYZ is deprecated.",
        ...             remove_version="2.0.0"
        ...         )
        ...     )
        ...
        >>> deprecated_function()
        DeprecationWarning: Function XYZ is deprecated. It will be removed in version 2.0.0.
    """

    def __init__(self, message: str, remove_version: str) -> None:
        super().__init__(message +
                         f' It will be removed in version {remove_version}.')


class ExperimentalWarning(Warning):
    """A warning for experimental features.

    Attributes:
        feature_name (str): The name of the experimental feature.
    """

    def __init__(self, feature_name: str) -> None:
        super().__init__(
            f'{feature_name} is experimental and may change with future versions.'
        )


F = TypeVar('F', bound=Callable[..., Any])


# Decorator version of experimental warning
def experimental_function(feature_name: str) -> Callable[[F], F]:
    """Decorator to mark a function as experimental.

    The message displayed will be {feature_name} is experimental and may change with future versions.

    Args:
        feature_name (str): The name of the experimental feature.

    Returns:
        The decorated function.
    """

    def decorator(func: Callable):

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any):
            warnings.warn(ExperimentalWarning(feature_name))
            return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorator


def experimental_class(feature_name: str) -> Callable[[Type], Type]:
    """Class decorator to mark a class as experimental."""

    def class_decorator(cls: Type):
        original_init = cls.__init__

        def new_init(self: Any, *args: Any, **kwargs: Any):
            warnings.warn(ExperimentalWarning(feature_name))
            original_init(self, *args, **kwargs)

        cls.__init__ = new_init
        return cls

    return class_decorator
