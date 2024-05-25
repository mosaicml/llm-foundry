# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import inspect
import pickle
from typing import Any, Dict, List, Optional, Type

import pytest

import llmfoundry.utils.exceptions as foundry_exceptions


def create_exception_object(
    exception_class: Type[foundry_exceptions.BaseContextualError],
):
    # get required arg types of exception class by inspecting its __init__ method

    if hasattr(inspect, 'get_annotations'):
        required_args = inspect.get_annotations( # type: ignore
            exception_class.__init__,
        )  # type: ignore
    else:
        required_args = exception_class.__init__.__annotations__  # python 3.9 and below

    # create a dictionary of required args with default values
    required_args.pop('kwargs', None)

    def get_default_value(arg_type: Optional[type] = None):
        if arg_type == Dict[str,
                            str] or arg_type == Dict[str,
                                                     Any] or arg_type == Dict:
            return {'key': 'value'}
        elif arg_type == str:
            return 'string'
        elif arg_type == int:
            return 1
        elif arg_type == set[str]:
            return {'set'}
        elif arg_type == List[str]:
            return ['list']
        elif arg_type == None:
            return None
        elif arg_type == type:
            return bool
        elif arg_type == List[Dict[str, Any]]:
            return [{'key': 'value'}]
        raise ValueError(f'Unsupported arg type: {arg_type}')

    required_args.pop('self', None)
    required_args.pop('return', None)
    kwargs = {
        arg: get_default_value(arg_type)
        for arg, arg_type in required_args.items()
    }
    return exception_class(**kwargs)  # type: ignore


def filter_exceptions(possible_exceptions: List[str]):
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
    exception_class: Type[foundry_exceptions.BaseContextualError],
):
    excluded_base_classes = [
        foundry_exceptions.InternalError,
        foundry_exceptions.UserError,
        foundry_exceptions.NetworkError,
        foundry_exceptions.BaseContextualError,
    ]

    exception = create_exception_object(exception_class)

    expect_reduce_error = exception.__class__ in excluded_base_classes
    error_context = pytest.raises(
        NotImplementedError,
    ) if expect_reduce_error else contextlib.nullcontext()

    exc_str = str(exception)
    with error_context:
        pkl = pickle.dumps(exception)
        unpickled_exc = pickle.loads(pkl)
        unpickled_exc_str = str(unpickled_exc)
        assert exc_str == unpickled_exc_str
        assert exception.location == unpickled_exc.location
        assert exception.error_attribution == unpickled_exc.error_attribution
