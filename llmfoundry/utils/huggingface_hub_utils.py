# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import ast
import importlib
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
        module_name: str, original_parent_module_name: Optional[str]) -> str:
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
                ast.ImportFrom) and node.module is not None and _remove_import(
                    node, remove_imports_prefix):
            nodes_to_remove.append(node)
        # Convert any (remaining) imports matching the flatten_imports_prefix
        # to relative imports
        elif (isinstance(node, ast.ImportFrom) and node.module is not None and
              _flatten_import(node, flatten_imports_prefix)):
            module_path = find_module_file(node.module)
            node.module = convert_to_relative_import(node.module,
                                                     parent_module_name)
            # Recursively process any llmfoundry files
            new_files_to_process.append(module_path)
        # Remove the Composer* class
        elif (isinstance(node, ast.ClassDef) and
              node.name.startswith('Composer')):
            nodes_to_remove.append(node)
        # Remove the __all__ declaration in any __init__.py files, whose
        # enclosing module will be converted to a single file of the same name
        elif (isinstance(node, ast.Assign) and len(node.targets) == 1 and
              isinstance(node.targets[0], ast.Name) and
              node.targets[0].id == '__all__'):
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


def edit_files_for_hf_compatibility(
    folder: str,
    flatten_imports_prefix: Sequence[str] = ('llmfoundry',),
    remove_imports_prefix: Sequence[str] = ('composer', 'omegaconf',
                                            'llmfoundry.metrics'),
) -> None:
    """Edit files to be compatible with Hugging Face Hub.

    Args:
        folder (str): The folder to process.
        flatten_imports_prefix (Sequence[str], optional): Sequence of prefixes to flatten. Defaults to ('llmfoundry',).
        remove_imports_prefix (Sequence[str], optional): Sequence of prefixes to remove. Takes precedence over flattening.
            Defaults to ('composer', 'omegaconf', 'llmfoundry.metrics').
    """
    files_to_process = [
        os.path.join(folder, filename)
        for filename in os.listdir(folder)
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
