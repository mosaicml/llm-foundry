from llmfoundry import registry
import catalogue
import pytest

def test_expected_registries_exist():
    existing_registries = {name for name in dir(registry) if isinstance(getattr(registry, name), registry.TypedRegistry)}
    expected_registry_names = {'loggers', 'optimizers', 'schedulers', 'callbacks', 'algorithms', 'callbacks_with_config'}
    
    assert existing_registries == expected_registry_names

def test_registry_create(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(catalogue, 'Registry', {})

    new_registry = registry.create('llmfoundry', 'test_registry', generic_type=str, entry_points=False)

    assert new_registry.namespace == ('llmfoundry', 'test_registry')
    assert isinstance(new_registry, registry.TypedRegistry)

def test_registry_typing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(catalogue, 'Registry', {})
    new_registry = registry.create('llmfoundry', 'test_registry', generic_type=str, entry_points=False)
    new_registry.register('test_name', func='test')

    # This would fail type checking without the type ignore
    # It is here to show that the TypedRegistry is working (gives a type error without the ignore),
    # although this would not catch a regression in this regard
    new_registry.register('test_name', func=1)  # type: ignore

def test_registry_add(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(catalogue, 'Registry', {})
    new_registry = registry.create('llmfoundry', 'test_registry', generic_type=str, entry_points=False)
    new_registry.register('test_name', func='test')

    assert new_registry.get('test_name') == 'test'
    

def test_registry_overwrite(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(catalogue, 'Registry', {})
    new_registry = registry.create('llmfoundry', 'test_registry', generic_type=str, entry_points=False)
    new_registry.register('test_name', func='test')
    new_registry.register('test_name', func='test2')

    assert new_registry.get('test_name') == 'test2'

def test_registry_init_code():
    pass

def test_registry_entrypoint():
    pass

def test_registry_builder():
    pass