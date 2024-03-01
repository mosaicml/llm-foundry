from llmfoundry.registry import import_file
import os
import pathlib

def test_registry_init_code(tmp_path: pathlib.Path):
    register_code = """
import os
os.environ['TEST_ENVIRON_REGISTRY_KEY'] = 'test'
"""

    with open(tmp_path / 'init_code.py', 'w') as _f:
        _f.write(register_code)

    import_file(tmp_path / 'init_code.py')

    assert os.environ['TEST_ENVIRON_REGISTRY_KEY'] == 'test'

    del os.environ['TEST_ENVIRON_REGISTRY_KEY']