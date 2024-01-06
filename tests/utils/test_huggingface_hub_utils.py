# Copyright 2022-2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import ast

from llmfoundry.utils.huggingface_hub_utils import _flatten_import


def test_flatten_import_true():
    node = ast.ImportFrom('y', ['x', 'y', 'z'])
    assert _flatten_import(node, ('x', 'y', 'z'))


def test_flatten_import_false():
    node = ast.ImportFrom('y', ['x', 'y', 'z'])
    assert not _flatten_import(node, ('x', 'z'))
