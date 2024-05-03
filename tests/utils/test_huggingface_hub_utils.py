# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import ast

import pytest

from llmfoundry.utils.huggingface_hub_utils import (
    _flatten_import,
    _remove_import,
)


def test_flatten_import_true():
    node = ast.ImportFrom('y', ['x', 'y', 'z'])
    assert _flatten_import(node, ('x', 'y', 'z'))


def test_flatten_import_false():
    node = ast.ImportFrom('y', ['x', 'y', 'z'])
    assert not _flatten_import(node, ('x', 'z'))


@pytest.mark.parametrize(
    'prefix_to_remove,expected_imports_remaining',
    [('llmfoundry', 1), ('llmfoundry.utils', 2)],
)
def test_remove_imports(prefix_to_remove: str, expected_imports_remaining: int):
    source_code = """
from llmfoundry import a
from llmfoundry.utils import b
from other_package import c
"""

    tree = ast.parse(source_code)
    assert len(tree.body) == 3

    imports_kept = 0
    for node in ast.walk(tree):
        if isinstance(
            node,
            ast.ImportFrom,
        ) and not _remove_import(node, [prefix_to_remove]):
            imports_kept += 1

    assert imports_kept == expected_imports_remaining
