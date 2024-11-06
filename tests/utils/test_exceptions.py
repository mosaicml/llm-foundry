# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import inspect
import pickle
from typing import Any, Optional, get_type_hints

import pytest

import llmfoundry.utils.exceptions as foundry_exceptions


def create_exception_object(
    exception_class: type[foundry_exceptions.BaseContextualError],
):

    def get_init_annotations(cls: type):
        try:
            return get_type_hints(cls.__init__)
        except (AttributeError, TypeError):
            # Handle cases where __init__ does not exist or has no annotations
            return {}

    # First, try to get annotations from the class itself
    required_args = get_init_annotations(exception_class)

    # If the annotations are empty, look at parent classes
    if not required_args:
        for parent in exception_class.__bases__:
            if parent == object:
                break
            parent_args = get_init_annotations(parent)
            if parent_args:
                required_args = parent_args
                break

    # Remove self, return, and kwargs
    required_args.pop('self', None)
    required_args.pop('return', None)
    required_args.pop('kwargs', None)

    def get_default_value(arg_type: Optional[type] = None):
        if arg_type == dict[str,
                            str] or arg_type == dict[str,
                                                     Any] or arg_type == dict:
            return {'key': 'value'}
        elif arg_type == str:
            return 'string'
        elif arg_type == int:
            return 1
        elif arg_type == float:
            return 1.0
        elif arg_type == list[float]:
            return [1.0]
        elif arg_type == set[str]:
            return {'set'}
        elif arg_type == list[str]:
            return ['list']
        elif arg_type == None:
            return None
        elif arg_type == type:
            return bool
        elif arg_type == list[dict[str, Any]]:
            return [{'key': 'value'}]
        elif arg_type == Optional[str]:
            return 'string_but_optional'
        raise ValueError(f'Unsupported arg type: {arg_type}')

    kwargs = {
        arg: get_default_value(arg_type)
        for arg, arg_type in required_args.items()
    }
    return exception_class(**kwargs)  # type: ignore


def filter_exceptions(possible_exceptions: list[str]):
    attrs = [
        getattr(foundry_exceptions, exception)
        for exception in possible_exceptions
    ]
    classes = [attr for attr in attrs if inspect.isclass(attr)]
    exceptions = [
        exception_class for exception_class in classes
        if issubclass(exception_class, foundry_exceptions.BaseContextualError)
    ]
    return exceptions


@pytest.mark.parametrize(
    'exception_class',
    filter_exceptions(dir(foundry_exceptions)),
)
def test_exception_serialization(
    exception_class: type[foundry_exceptions.BaseContextualError],
):
    print(f'Testing serialization for {exception_class.__name__}')
    excluded_base_classes = [
        foundry_exceptions.InternalError,
        foundry_exceptions.UserError,
        foundry_exceptions.NetworkError,
        foundry_exceptions.BaseContextualError,
    ]

    exception = create_exception_object(exception_class)
    print(f'Created exception object: {exception}')

    expect_reduce_error = exception.__class__ in excluded_base_classes
    error_context = pytest.raises(
        NotImplementedError,
    ) if expect_reduce_error else contextlib.nullcontext()

    exc_str = str(exception)
    print(f'Exception string: {exc_str}')
    with error_context:
        pkl = pickle.dumps(exception)
        unpickled_exc = pickle.loads(pkl)
        unpickled_exc_str = str(unpickled_exc)
        assert exc_str == unpickled_exc_str
        assert exception.location == unpickled_exc.location
        assert exception.error_attribution == unpickled_exc.error_attribution
