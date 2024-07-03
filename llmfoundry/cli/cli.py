# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import typer

from llmfoundry.cli import registry_cli
from llmfoundry.train import train

app = typer.Typer(pretty_exceptions_show_locals=False)
app.add_typer(registry_cli.app, name='registry')
app.add_typer(train.app, name='train')

if __name__ == '__main__':
    app()
