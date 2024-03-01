# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib

import pytest

from llmfoundry.registry import import_file


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


def test_registry_init_code_fails(tmp_path: pathlib.Path):
    register_code = """
import os
os.environ['TEST_ENVIRON_REGISTRY_KEY'] = 'test'
asdf
"""

    with open(tmp_path / 'init_code.py', 'w') as _f:
        _f.write(register_code)

    with pytest.raises(RuntimeError, match='Error executing .*init_code.py'):
        import_file(tmp_path / 'init_code.py')


def test_registry_init_code_dne(tmp_path: pathlib.Path):
    with pytest.raises(FileNotFoundError, match='File .* does not exist'):
        import_file(tmp_path / 'init_code.py')
