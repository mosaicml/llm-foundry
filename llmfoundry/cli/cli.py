# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import typer

from llmfoundry.cli import registry_cli

app = typer.Typer(pretty_exceptions_show_locals=False)
app.add_typer(registry_cli.app, name='registry')

if __name__ == '__main__':
    app()
