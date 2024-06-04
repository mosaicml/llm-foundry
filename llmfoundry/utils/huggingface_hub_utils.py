# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import ast
import importlib
import json
import os
from typing import Optional, Sequence

__all__ = ['edit_files_for_hf_compatibility']


class DeleteSpecificNodes(ast.NodeTransformer):

    def __init__(self, nodes_to_remove: list[ast.AST]):
        self.nodes_to_remove = nodes_to_remove

    def visit(self, node: ast.AST) -> Optional[ast.AST]:
        if node in self.nodes_to_remove:
            return None

        return super().visit(node)


def convert_to_relative_import(
    module_name: str,
    original_parent_module_name: Optional[str],
) -> str:
    parts = module_name.split('.')
    if parts[-1] == original_parent_module_name:
        return '.'
    return '.' + parts[-1]


def find_module_file(module_name: str) -> str:
    if not module_name:
        raise ValueError(f'Invalid input: {module_name=}')
    module = importlib.import_module(module_name)
    module_file = module.__file__
    if module_file is None:
        raise ValueError(f'Could not find file for module: {module_name}')
    return module_file


def _flatten_import(
    node: ast.ImportFrom,
    flatten_imports_prefix: Sequence[str],
) -> bool:
    """Returns True if import should be flattened.

    Checks whether the node starts the same as any of the imports in
    flatten_imports_prefix.
    """
    for import_prefix in flatten_imports_prefix:
        if node.module is not None and node.module.startswith(import_prefix):
            return True
    return False


def _remove_import(
    node: ast.ImportFrom,
    remove_imports_prefix: Sequence[str],
) -> bool:
    """Returns True if import should be removed.

    Checks whether the node starts the same as any of the imports in
    remove_imports_prefix.
    """
    for import_prefix in remove_imports_prefix:
        if node.module is not None and node.module.startswith(import_prefix):
            return True
    return False


def process_file(
    file_path: str,
    folder_path: str,
    flatten_imports_prefix: Sequence[str],
    remove_imports_prefix: Sequence[str],
) -> list[str]:
    with open(file_path, 'r', encoding='utf-8') as f:
        source = f.read()

    parent_module_name = None
    if os.path.basename(file_path) == '__init__.py':
        parent_module_name = os.path.basename(os.path.dirname(file_path))

    tree = ast.parse(source)
    new_files_to_process = []
    nodes_to_remove = []
    for node in ast.walk(tree):
        # Remove any imports matching the remove_imports_prefix
        if isinstance(
            node,
            ast.ImportFrom,
        ) and node.module is not None and _remove_import(
            node,
            remove_imports_prefix,
        ):
            nodes_to_remove.append(node)
        # Convert any (remaining) imports matching the flatten_imports_prefix
        # to relative imports
        elif (
            isinstance(node, ast.ImportFrom) and node.module is not None and
            _flatten_import(node, flatten_imports_prefix)
        ):
            module_path = find_module_file(node.module)
            node.module = convert_to_relative_import(
                node.module,
                parent_module_name,
            )
            # Recursively process any llmfoundry files
            new_files_to_process.append(module_path)
        # Remove the Composer* class
        elif (
            isinstance(node, ast.ClassDef) and node.name.startswith('Composer')
        ):
            nodes_to_remove.append(node)
        # Remove the __all__ declaration in any __init__.py files, whose
        # enclosing module will be converted to a single file of the same name
        elif (
            isinstance(node, ast.Assign) and len(node.targets) == 1 and
            isinstance(node.targets[0], ast.Name) and
            node.targets[0].id == '__all__'
        ):
            nodes_to_remove.append(node)

    transformer = DeleteSpecificNodes(nodes_to_remove)
    new_tree = transformer.visit(tree)

    new_filename = os.path.basename(file_path)
    # Special case for __init__.py to mimic the original submodule
    if new_filename == '__init__.py':
        new_filename = file_path.split('/')[-2] + '.py'
    new_file_path = os.path.join(folder_path, new_filename)
    with open(new_file_path, 'w', encoding='utf-8') as f:
        assert new_tree is not None
        f.write(ast.unparse(new_tree))

    return new_files_to_process


def get_all_relative_imports(file_path: str) -> set[str]:
    """Get all relative imports from a file.

    Args:
        file_path (str): The file to process.

    Returns:
        set[str]: The relative imports.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        source = f.read()

    tree = ast.parse(source)
    relative_imports = set()
    for node in ast.walk(tree):
        if isinstance(
            node,
            ast.ImportFrom,
        ) and node.module is not None and node.level == 1:
            relative_imports.add(node.module)

    return relative_imports


def add_relative_imports(
    file_path: str,
    relative_imports_to_add: set[str],
) -> None:
    """Add relative imports to a file.

    Args:
        file_path (str): The file to add to.
        relative_imports_to_add (set[str]): The set of relative imports to add
    """
    # Get the directory name where all the files are located
    dir_name = os.path.dirname(file_path)

    with open(file_path, 'r', encoding='utf-8') as f:
        source = f.read()

    tree = ast.parse(source)

    for relative_import in relative_imports_to_add:
        import_path = os.path.join(dir_name, relative_import + '.py')
        # Open up the file we are adding an import to find something to import from it
        with open(import_path, 'r', encoding='utf-8') as f:
            import_source = f.read()
        import_tree = ast.parse(import_source)
        item_to_import = None
        for node in ast.walk(import_tree):
            # Find the first function or class
            if isinstance(node,
                          ast.FunctionDef) or isinstance(node, ast.ClassDef):
                # Get its name to import it
                item_to_import = node.name
                break

        if item_to_import is None:
            item_to_import = '*'

        # This will look like `from .relative_import import item_to_import`
        import_node = ast.ImportFrom(
            module=relative_import,
            names=[ast.alias(name=item_to_import, asname=None)],
            level=1,
        )

        # Insert near the top of the file, but past the from __future__ import
        tree.body.insert(2, import_node)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(ast.unparse(tree))


def edit_files_for_hf_compatibility(
    folder: str,
    flatten_imports_prefix: Sequence[str] = ('llmfoundry',),
    remove_imports_prefix: Sequence[str] = (
        'composer',
        'omegaconf',
        'llmfoundry.metrics',
        'llmfoundry.eval',
        'llmfoundry.utils.builders',
    ),
) -> None:
    """Edit files to be compatible with Hugging Face Hub.

    Args:
        folder (str): The folder to process.
        flatten_imports_prefix (Sequence[str], optional): Sequence of prefixes to flatten. Defaults to ('llmfoundry',).
        remove_imports_prefix (Sequence[str], optional): Sequence of prefixes to remove. Takes precedence over flattening.
            Defaults to ('composer', 'omegaconf', 'llmfoundry.metrics', 'llmfoundry.utils.builders').
    """
    listed_dir = os.listdir(folder)

    # Try to acquire the config file to determine which python file is the entrypoint file
    config_file_exists = 'config.json' in listed_dir
    with open(os.path.join(folder, 'config.json'), 'r') as _f:
        config = json.load(_f)

    # If the config file exists, the entrypoint files would be specified in the auto map
    entrypoint_files = set()
    if config_file_exists:
        for key, value in config.get('auto_map', {}).items():
            # Only keep the modeling entrypoints, e.g. AutoModelForCausalLM
            if 'model' not in key.lower():
                continue
            split_path = value.split('.')
            if len(split_path) > 1:
                entrypoint_files.add(split_path[0] + '.py')

    files_to_process = [
        os.path.join(folder, filename)
        for filename in listed_dir
        if filename.endswith('.py')
    ]
    files_processed_and_queued = set(files_to_process)

    while len(files_to_process) > 0:
        to_process = files_to_process.pop()
        if os.path.isfile(to_process) and to_process.endswith('.py'):
            to_add = process_file(
                to_process,
                folder,
                flatten_imports_prefix,
                remove_imports_prefix,
            )
            for file in to_add:
                if file not in files_processed_and_queued:
                    files_to_process.append(file)
                    files_processed_and_queued.add(file)

    # For each entrypoint, determine which imports are missing, and add them
    # This is because HF does not recursively search imports when determining
    # which files to copy into its modules cache
    all_relative_imports = {
        os.path.splitext(os.path.basename(f))[0]
        for f in files_processed_and_queued
    }
    for entrypoint in entrypoint_files:
        existing_relative_imports = get_all_relative_imports(
            os.path.join(folder, entrypoint),
        )
        # Add in self so we don't create a circular import
        existing_relative_imports.add(os.path.splitext(entrypoint)[0])
        missing_relative_imports = all_relative_imports - existing_relative_imports
        add_relative_imports(
            os.path.join(folder, entrypoint),
            missing_relative_imports,
        )
