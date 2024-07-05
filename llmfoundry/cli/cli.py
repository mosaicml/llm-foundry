# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import typer
import sys

from llmfoundry.cli import registry_cli
from llmfoundry.train.train import run_trainer

app = typer.Typer(pretty_exceptions_show_locals=False)
app.add_typer(registry_cli.app, name='registry')

@app.command(name="train")
def train(yaml_path: str = typer.Argument(..., help="Path to the YAML configuration file")):
    """Run the training with the given configuration and optional overrides from command line."""
    args_list = sys.argv[2:]  # assuming additional arguments may follow the yaml_path
    run_trainer(yaml_path, args_list)

if __name__ == '__main__':
    app()
