# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import inspect
import typing

import pytest
from composer.core import Callback

from llmfoundry.callbacks.async_eval_callback import AsyncEval
from llmfoundry.callbacks.curriculum_learning_callback import CurriculumLearning
from llmfoundry.interfaces.callback_with_config import CallbackWithConfig
from llmfoundry.registry import callbacks, callbacks_with_config
from llmfoundry.utils.builders import build_callback

primitive_types = {int, float, str, bool, dict, list}

# Callbacks that we skip during testing because they require more complex inputs.
# They should be tested separately.
skip_callbacks = [
    AsyncEval,
    CurriculumLearning,
]


def get_default_value(
    param: str,
    tpe: type,
    inspected_param: typing.Optional[inspect.Parameter],
):
    if typing.get_origin(tpe) is typing.Union:
        args = typing.get_args(tpe)
        return get_default_value(param, args[0], None)
    elif typing.get_origin(tpe) is list or typing.get_origin(tpe) is list:
        return []
    elif typing.get_origin(tpe) is dict or typing.get_origin(tpe) is dict:
        return {}
    elif tpe is int:
        return 0
    elif tpe is float:
        return 0.0
    elif tpe is str:
        return ''
    elif tpe is bool:
        return False
    elif tpe is dict:
        return {}
    elif tpe is list:
        return []
    elif inspected_param is not None and tpe is typing.Any and inspected_param.kind is inspect.Parameter.VAR_KEYWORD:
        return None
    elif inspected_param is not None and tpe is typing.Any and inspected_param.kind is inspect.Parameter.VAR_POSITIONAL:
        return None
    else:
        raise ValueError(f'Unsupported type: {tpe} for parameter {param}')


def get_default_kwargs(callback_class: type):
    type_hints = typing.get_type_hints(callback_class.__init__)
    inspected_params = inspect.signature(callback_class.__init__).parameters

    default_kwargs = {}

    for param, tpe in type_hints.items():
        if param == 'self' or param == 'return' or param == 'train_config':
            continue
        if inspected_params[param].default == inspect.Parameter.empty:
            default_value = get_default_value(
                param,
                tpe,
                inspected_params[param],
            )
            if default_value is not None:
                default_kwargs[param] = default_value
    return default_kwargs


def maybe_skip_callback_test(callback_class: type):
    if hasattr(
        callback_class,
        'is_experimental',
    ) and callback_class.is_experimental:  # type: ignore
        pytest.skip(
            f'Skipping test for {callback_class.__name__} because it is experimental.',
        )
    if callback_class in skip_callbacks:
        pytest.skip(
            f'Skipping test for {callback_class.__name__}. It should be tested elsewhere.',
        )


@pytest.mark.parametrize(
    'callback_name,callback_class',
    callbacks.get_all().items(),
)
def test_build_callback(callback_name: str, callback_class: type):
    maybe_skip_callback_test(callback_class)
    get_default_kwargs(callback_class)

    callback = build_callback(
        callback_name,
        kwargs=get_default_kwargs(callback_class),
    )

    assert isinstance(callback, callback_class)
    assert isinstance(callback, Callback)


@pytest.mark.parametrize(
    'callback_name,callback_class',
    callbacks_with_config.get_all().items(),
)
def test_build_callback_with_config(callback_name: str, callback_class: type):
    maybe_skip_callback_test(callback_class)
    get_default_kwargs(callback_class)

    callback = build_callback(
        callback_name,
        kwargs=get_default_kwargs(callback_class),
        train_config={
            'save_folder': 'test',
            'save_interval': '1ba',
        },
    )

    assert isinstance(callback, callback_class)
    assert isinstance(callback, CallbackWithConfig)
