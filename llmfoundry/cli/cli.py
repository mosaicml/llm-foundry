# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from argparse import Namespace
from typing import Optional

import typer

from llmfoundry.cli import registry_cli
from llmfoundry.data_prep import convert_dataset_json_from_args
from llmfoundry.train import train_from_yaml

app = typer.Typer(pretty_exceptions_show_locals=False)
app.add_typer(registry_cli.app, name='registry')


@app.command(name='train')
def train(
    yaml_path: str = typer.Argument(
        ...,
        help='Path to the YAML configuration file',
    ),  # type: ignore
    args_list: Optional[list[str]] = typer.
    Argument(None, help='Additional command line arguments'),  # type: ignore
):
    """Run the training with optional overrides from CLI."""
    train_from_yaml(yaml_path, args_list)


@app.command(name='convert_dataset_json')
def convert_dataset_json_from_args_cli(
    path: str = typer.Option(
        ...,
        '--path',
        help='Path to the input data file',
    ),  # type: ignore
    out_root: str = typer.
    Option(..., '--out_root', help='Output root directory'),  # type: ignore
    compression: str = typer.Option(
        None,
        '--compression',
        help='Compression type, if any',
    ),  # type: ignore
    concat_tokens: int = typer.Option(
        None,
        '--concat_tokens',
        help='Convert text to tokens and concatenate up to this many tokens',
    ),  # type: ignore
    split: str = typer.
    Option('train', '--split', help='Dataset split to process'),  # type: ignore
    tokenizer: Optional[str] = typer.
    Option(None, '--tokenizer', help='Tokenizer name'),  # type: ignore
    bos_text: Optional[str] = typer.Option(
        None,
        '--bos_text',
        help='Text to insert at the beginning of each sequence',
    ),  # type: ignore
    eos_text: Optional[str] = typer.Option(
        None,
        '--eos_text',
        help='Text to insert at the end of each sequence',
    ),  # type: ignore
    no_wrap: bool = typer.Option(
        False,
        '--no_wrap',
        help='Do not wrap text across max_length boundaries',
    ),  # type: ignore
    num_workers: int = typer.Option(
        None,
        '--num_workers',
        help='Number of workers for data loading',
    ),  # type: ignore
):
    args = Namespace(
        path=path,
        out_root=out_root,
        compression=compression,
        concat_tokens=concat_tokens,
        split=split,
        tokenizer=tokenizer,
        bos_text=bos_text,
        eos_text=eos_text,
        no_wrap=no_wrap,
        num_workers=num_workers,
    )
    convert_dataset_json_from_args(args)


if __name__ == '__main__':
    app()
