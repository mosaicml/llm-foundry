# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import ast
import importlib
import os
from typing import List, Optional

__all__ = ['edit_files_for_hf_compatibility']


class DeleteSpecificNodes(ast.NodeTransformer):

    def __init__(self, nodes_to_remove: List[ast.AST]):
        self.nodes_to_remove = nodes_to_remove

    def visit(self, node: ast.AST):
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
    module = importlib.import_module(module_name)
    module_file = module.__file__
    return module_file


def process_file(file_path: str, folder_path: str) -> List[str]:
    with open(file_path, 'r') as f:
        source = f.read()

    parent_module_name = None
    if os.path.basename(file_path) == '__init__.py':
        parent_module_name = os.path.basename(os.path.dirname(file_path))

    tree = ast.parse(source)
    new_files_to_process = []
    nodes_to_remove = []
    for node in ast.walk(tree):
        # convert any llmfoundry imports into relative imports
        if isinstance(node,
                      ast.ImportFrom) and node.module.startswith('llmfoundry'):
            module_path = find_module_file(node.module)
            node.module = convert_to_relative_import(node.module,
                                                     parent_module_name)
            # recursively process any llmfoundry files
            new_files_to_process.append(module_path)
        # remove any imports from composer or omegaconf
        elif isinstance(
                node, ast.ImportFrom) and (node.module.startswith('composer') or
                                           node.module.startswith('omegaconf')):
            nodes_to_remove.append(node)
        # remove the Composer* class
        elif isinstance(node,
                        ast.ClassDef) and node.name.startswith('Composer'):
            nodes_to_remove.append(node)
        # remove the __all__ declaration in any __init__.py files, whose enclosing module
        # will be converted to a single file of the same name
        elif isinstance(node,
                        ast.Assign) and len(node.targets) == 1 and isinstance(
                            node.targets[0],
                            ast.Name) and node.targets[0].id == '__all__':
            nodes_to_remove.append(node)

    transformer = DeleteSpecificNodes(nodes_to_remove)
    new_tree = transformer.visit(tree)

    new_filename = os.path.basename(file_path)
    # special case for __init__.py to mimic the original submodule
    if new_filename == '__init__.py':
        new_filename = file_path.split('/')[-2] + '.py'
    new_file_path = os.path.join(folder_path, new_filename)
    with open(new_file_path, 'w') as f:
        f.write(ast.unparse(new_tree))

    return new_files_to_process


def edit_files_for_hf_compatibility(folder: str):
    files_to_process = [
        os.path.join(folder, filename)
        for filename in os.listdir(folder)
        if filename.endswith('.py')
    ]
    files_processed_and_queued = set(files_to_process)

    while len(files_to_process) > 0:
        to_process = files_to_process.pop()
        if os.path.isfile(to_process) and to_process.endswith('.py'):
            to_add = process_file(to_process, folder)
            for file in to_add:
                if file not in files_processed_and_queued:
                    files_to_process.append(file)
                    files_processed_and_queued.add(file)
