# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated, Optional

from typer import Argument, Typer

from llmfoundry.cli import (
    data_prep_cli,
    registry_cli,
)
from llmfoundry.command_utils import (
    eval_from_yaml,
    train_from_yaml,
)

app = Typer(pretty_exceptions_show_locals=False)
app.add_typer(registry_cli.app, name='registry')
app.add_typer(data_prep_cli.app, name='data_prep')


@app.command(name='train')
def train(
    yaml_path: Annotated[str,
                         Argument(
                             ...,
                             help='Path to the YAML configuration file',
                         )],
    args_list: Annotated[
        Optional[list[str]],
        Argument(help='Additional command line arguments')] = None,
):
    """Run the training with optional overrides from CLI."""
    train_from_yaml(yaml_path, args_list)


@app.command(name='eval')
def eval(
    yaml_path: Annotated[str,
                         Argument(
                             ...,
                             help='Path to the YAML configuration file',
                         )],
    args_list: Annotated[
        Optional[list[str]],
        Argument(help='Additional command line arguments')] = None,
):
    """Run the eval with optional overrides from CLI."""
    eval_from_yaml(yaml_path, args_list)


if __name__ == '__main__':
    app()
