# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os
from pathlib import Path
from types import ModuleType
from typing import Union

__all__ = ['import_file']


def import_file(loc: Union[str, Path]) -> ModuleType:
    """Import module from a file. Used to run arbitrary python code.

    Args:
        name (str): Name of module to load.
        loc (str / Path): Path to the file.

    Returns:
        ModuleType: The module object.
    """
    if not os.path.exists(loc):
        raise FileNotFoundError(f'File {loc} does not exist.')

    spec = importlib.util.spec_from_file_location('python_code', str(loc))

    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)

    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f'Error executing {loc}') from e
    return module
