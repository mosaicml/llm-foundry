from typing import Union
from pathlib import Path
from types import ModuleType
import importlib

__all__ = ['import_file']

def import_file(loc: Union[str, Path]) -> ModuleType:
    """Import module from a file. Used to run arbitrary python code.

    Args:
        name (str): Name of module to load.
        loc (str / Path): Path to the file.

    Returns:
        ModuleType: The module object.
    """
    spec = importlib.util.spec_from_file_location(
        'python_code', str(loc))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module