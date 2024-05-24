# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import inspect
import pickle
from typing import Any, Dict, List, Optional

import pytest

import llmfoundry.utils.exceptions as foundry_exceptions


def create_exception_object(exception_name: str):
    exception_class = getattr(
        __import__('llmfoundry.utils.exceptions', fromlist=[exception_name]),
        exception_name,
    )
    # get required arg types of exception class by inspecting its __init__ method

    if hasattr(inspect, 'get_annotations'):
        required_args = inspect.get_annotations(
            exception_class.__init__,
        )  # type: ignore
    else:
        required_args = exception_class.__init__.__annotations__  # python 3.9 and below

    # create a dictionary of required args with default values

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
        raise ValueError(f'Unsupported arg type: {arg_type}')

    required_args.pop('self', None)
    required_args.pop('return', None)
    kwargs = {
        arg: get_default_value(arg_type)
        for arg, arg_type in required_args.items()
    }
    return exception_class(*kwargs.values())


def filter_exceptions(exceptions: List[str]):
    return [
        exception for exception in exceptions
        if ('Error' in exception or 'Exception' in exception) and
        ('Base' not in exception)
    ]


@pytest.mark.parametrize(
    'exception_name',
    filter_exceptions(dir(foundry_exceptions)),
)
def test_exception_serialization(exception_name: str):
    exception = create_exception_object(exception_name)

    exc_str = str(exception)
    pkl = pickle.dumps(exception)
    unpickled_exc = pickle.loads(pkl)
    unpickled_exc_str = str(unpickled_exc)
    assert exc_str == unpickled_exc_str
