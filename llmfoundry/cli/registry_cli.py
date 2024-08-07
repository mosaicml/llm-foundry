# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

from rich.console import Console
from rich.table import Table
from typer import Typer

from llmfoundry import registry
from llmfoundry.utils.registry_utils import TypedRegistry

console = Console()
app = Typer(pretty_exceptions_show_locals=False)


def _get_registries(group: Optional[str] = None) -> list[TypedRegistry]:
    registry_attr_names = dir(registry)
    registry_attrs = [getattr(registry, name) for name in registry_attr_names]
    available_registries = [
        r for r in registry_attrs if isinstance(r, TypedRegistry)
    ]

    if group is not None and group not in registry_attr_names:
        console.print(
            f'Group {group} not found in registry. Run `llmfoundry registry get` to see available groups.',
        )
        return []

    if group is not None:
        available_registries = [getattr(registry, group)]

    return available_registries


@app.command()
def get(group: Optional[str] = None):
    """Get the available registries.

    Args:
        group (Optional[str], optional): The group to get. If not provided, all groups will be shown. Defaults to None.
    """
    available_registries = _get_registries(group)

    table = Table('Registry', 'Description', 'Options', show_lines=True)
    for r in available_registries:
        table.add_row(
            '.'.join(r.namespace),
            r.description,
            ', '.join(r.get_all()),
        )

    console.print(table)


@app.command()
def find(group: str, name: str):
    """Find a registry entry by name.

    Args:
        group (str): The group to search.
        name (str): The name of the entry to search for.
    """
    available_registries = _get_registries(group)
    if not available_registries:
        return

    r = available_registries[0]
    find_output = r.find(name)

    table = Table('Module', 'File', 'Line number', 'Docstring')
    table.add_row(
        find_output['module'],
        find_output['file'],
        str(find_output['line_no']),
        find_output['docstring'],
    )

    console.print(table)
