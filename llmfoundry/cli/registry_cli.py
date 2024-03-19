import typer
from rich.console import Console
from rich.table import Table

console = Console()
from typing import Optional

from llmfoundry.utils import registry_utils
from llmfoundry import registry

app = typer.Typer()

@app.command()
def get(group: Optional[str] = None):
    registries = [
        getattr(registry, name)
        for name in (dir(registry) if group is None else [group])
        if isinstance(getattr(registry, name), registry_utils.TypedRegistry)
    ]

    options = [
        r.get_all() for r in registries
    ]

    table = Table("Registry", "Description", "Options")
    for r, options in zip(registries, options):
        table.add_row('.'.join(r.namespace), r.description, ", ".join(options))

    console.print(table)

@app.command()
def find(group: str, name: str):
    r = getattr(registry, group)
    find_output = r.find(name)

    table = Table(width=160)
    table.add_column("Module")
    table.add_column("File", overflow='fold', max_width=40)
    table.add_column("Line number")
    table.add_column("Docstring")
    table.add_row(find_output['module'], find_output['file'], str(find_output['line_no']), find_output['docstring'])

    console.print(table)
