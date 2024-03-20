from llmfoundry.cli.registry_cli import _get_registries
from llmfoundry.utils.registry_utils import TypedRegistry
from llmfoundry import registry

def test_get_registries():
    available_registries = _get_registries()
    expected_registries = [
        getattr(registry, r) for r in dir(registry) if isinstance(getattr(registry, r), TypedRegistry)
    ]
    assert available_registries == expected_registries

def test_get_registries_group():
    available_registries = _get_registries('loggers')
    assert len(available_registries) == 1
    assert available_registries[0].namespace == ('llmfoundry', 'loggers')
