# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import importlib.metadata
import pathlib
from importlib.metadata import EntryPoint
from typing import Any, Callable, Type, Union

import catalogue
import pytest
from composer.loggers import InMemoryLogger, LoggerDestination

from llmfoundry import registry
from scripts.train.train import import_file


def test_expected_registries_exist():
    existing_registries = {
        name for name in dir(registry)
        if isinstance(getattr(registry, name), registry.TypedRegistry)
    }
    expected_registry_names = {
        'loggers', 'optimizers', 'schedulers', 'callbacks', 'algorithms',
        'callbacks_with_config'
    }

    assert existing_registries == expected_registry_names


def test_registry_create(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(catalogue, 'Registry', {})

    new_registry = registry.create('llmfoundry',
                                   'test_registry',
                                   generic_type=str,
                                   entry_points=False)

    assert new_registry.namespace == ('llmfoundry', 'test_registry')
    assert isinstance(new_registry, registry.TypedRegistry)


def test_registry_typing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(catalogue, 'Registry', {})
    new_registry = registry.create('llmfoundry',
                                   'test_registry',
                                   generic_type=str,
                                   entry_points=False)
    new_registry.register('test_name', func='test')

    # This would fail type checking without the type ignore
    # It is here to show that the TypedRegistry is working (gives a type error without the ignore),
    # although this would not catch a regression in this regard
    new_registry.register('test_name', func=1)  # type: ignore


def test_registry_add(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(catalogue, 'Registry', {})
    new_registry = registry.create('llmfoundry',
                                   'test_registry',
                                   generic_type=str,
                                   entry_points=False)
    new_registry.register('test_name', func='test')

    assert new_registry.get('test_name') == 'test'


def test_registry_overwrite(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(catalogue, 'Registry', {})
    new_registry = registry.create('llmfoundry',
                                   'test_registry',
                                   generic_type=str,
                                   entry_points=False)
    new_registry.register('test_name', func='test')
    new_registry.register('test_name', func='test2')

    assert new_registry.get('test_name') == 'test2'


def test_registry_init_code(tmp_path: pathlib.Path):
    register_code = """
from llmfoundry.registry import loggers
from composer.loggers import InMemoryLogger

@loggers.register('test_logger')
class TestLogger(InMemoryLogger):
    pass
"""

    with open(tmp_path / 'init_code.py', 'w') as _f:
        _f.write(register_code)

    import_file(tmp_path / 'init_code.py')

    assert issubclass(registry.loggers.get('test_logger'), InMemoryLogger)

    del catalogue.REGISTRY[('llmfoundry', 'loggers', 'test_logger')]

    assert 'test_logger' not in registry.loggers


def test_registry_entrypoint(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(catalogue, 'Registry', {})

    # monkeypatch.setattr(
    #     importlib.metadata, 'entry_points', lambda: SelectableGroups([(
    #         'llmfoundry_test_registry',
    #         EntryPoints([
    #             EntryPoint(name='test_entry',
    #                        value='composer.loggers:InMemoryLogger',
    #                        group='llmfoundry_test_registry')
    #         ]))]))
    monkeypatch.setattr(
        importlib.metadata, 'entry_points', lambda: {
            'llmfoundry_test_registry': [
                EntryPoint(name='test_entry',
                           value='composer.loggers:InMemoryLogger',
                           group='llmfoundry_test_registry')
            ]
        })

    monkeypatch.setattr(catalogue, 'AVAILABLE_ENTRY_POINTS',
                        importlib.metadata.entry_points())
    new_registry = registry.create('llmfoundry',
                                   'test_registry',
                                   generic_type=str,
                                   entry_points=True)
    assert new_registry.get('test_entry') == InMemoryLogger


def test_registry_builder(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(catalogue, 'Registry', {})

    new_registry = registry.create(
        'llmfoundry',
        'test_registry',
        entry_points=False,
        generic_type=Union[Type[LoggerDestination],
                           Callable[..., LoggerDestination]])

    class TestLoggerDestination(LoggerDestination):
        pass

    new_registry.register('test_destination', func=TestLoggerDestination)

    # Valid, no validation
    valid_class = registry.builder(
        'test_destination',
        new_registry,
        pre_validation_function=TestLoggerDestination)
    assert isinstance(valid_class, TestLoggerDestination)

    # Invalid, class validation
    with pytest.raises(ValueError,
                       match='Expected test_destination to be of type'):
        registry.builder('test_destination',
                         new_registry,
                         pre_validation_function=InMemoryLogger)

    # Invalid, function pre-validation
    with pytest.raises(ValueError, match='Invalid'):

        def pre_validation_function(x: Any):
            raise ValueError('Invalid')

        registry.builder('test_destination',
                         new_registry,
                         pre_validation_function=pre_validation_function)

    # Invalid, function post-validation
    with pytest.raises(ValueError, match='Invalid'):

        def post_validation_function(x: Any):
            raise ValueError('Invalid')

        registry.builder('test_destination',
                         new_registry,
                         post_validation_function=post_validation_function)

    # Invalid, not a class or function
    new_registry.register('non_callable', func=1)  # type: ignore
    with pytest.raises(ValueError,
                       match='Expected non_callable to be a class or function'):
        registry.builder('non_callable', new_registry)

    # Valid, partial function
    new_registry.register('partial_func',
                          func=lambda x, y: x * y)  # type: ignore
    partial_func = registry.builder('partial_func',
                                    new_registry,
                                    partial_function=True,
                                    kwargs={'x': 2})
    assert partial_func(y=3) == 6

    # Valid, builder function
    new_registry.register('builder_func', func=lambda: TestLoggerDestination())
    valid_built_class = registry.builder('builder_func',
                                         new_registry,
                                         partial_function=False)
    assert isinstance(valid_built_class, TestLoggerDestination)
