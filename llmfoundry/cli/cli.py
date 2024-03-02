import typer

from llmfoundry.cli import registry_cli

app = typer.Typer()
app.add_typer(registry_cli.app, name='registry')


if __name__ == "__main__":
    app()