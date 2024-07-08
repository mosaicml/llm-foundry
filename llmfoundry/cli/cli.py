# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from typing import List

import typer

from llmfoundry.cli import registry_cli
from llmfoundry.train.train import run_trainer

app = typer.Typer(pretty_exceptions_show_locals=False)
app.add_typer(registry_cli.app, name='registry')


@app.command(name='train')
def train(
    yaml_path: str = typer.Argument(
        ..., help='Path to the YAML configuration file'
    ),
    additional_args: List[str] = typer.Option(
        [],
        '--extra',
        help='Additional command line arguments',
        show_default=False,
    ),
):
    """Run the training with the given configuration and optional overrides from
    command line."""
    run_trainer(yaml_path, additional_args)


if __name__ == '__main__':
    app()
