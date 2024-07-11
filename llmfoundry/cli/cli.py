# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from argparse import Namespace
from typing import Optional

import typer

from llmfoundry.cli import registry_cli
from llmfoundry.data_prep import convert_dataset_hf_from_args
from llmfoundry.train import train_from_yaml

app = typer.Typer(pretty_exceptions_show_locals=False)
app.add_typer(registry_cli.app, name='registry')


@app.command(name='train')
def train_from_yaml_cli(
    yaml_path: str = typer.Argument(
        ...,
        help='Path to the YAML configuration file',
    ),  # type: ignore
    args_list: Optional[list[str]] = typer.
    Argument(None, help='Additional command line arguments'),  # type: ignore
):
    """Run the training with optional overrides from CLI."""
    train_from_yaml(yaml_path, args_list)


@app.command(name='convert_dataset_hf')
def convert_dataset_hf_cli(
    dataset: str = typer.Option(..., '--dataset',
                                help='Name of the dataset'),  # type: ignore
    data_subset: Optional[str] = typer.Option(
        None,
        '--data_subset',
        help='Subset of the dataset (e.g., "all" or "en")',
    ),  # type: ignore
    splits: str = typer.Option(
        'train, train_small, val, val_small, val_xsmall',
        '--splits',
        help='Comma-separated list of dataset splits',
    ),  # type: ignore
    out_root: str = typer.
    Option(..., '--out_root', help='Output root directory'),  # type: ignore
    compression: Optional[str] = typer.
    Option(None, '--compression', help='Compression type'),  # type: ignore
    concat_tokens: Optional[int] = typer.Option(
        None,
        '--concat_tokens',
        help='Concatenate tokens up to this many tokens',
    ),  # type: ignore
    tokenizer: Optional[str] = typer.
    Option(None, '--tokenizer', help='Tokenizer name'),  # type: ignore
    tokenizer_kwargs: Optional[str] = typer.Option(
        None,
        '--tokenizer_kwargs',
        help='Tokenizer keyword arguments in JSON format',
    ),  # type: ignore
    bos_text: Optional[str] = typer.Option(None, '--bos_text',
                                           help='BOS text'),  # type: ignore
    eos_text: Optional[str] = typer.Option(None, '--eos_text',
                                           help='EOS text'),  # type: ignore
    no_wrap: bool = typer.Option(
        False,
        '--no_wrap',
        help='Do not wrap text across max_length boundaries',
    ),  # type: ignore
    num_workers: Optional[int] = typer.
    Option(None, '--num_workers', help='Number of workers'),  # type: ignore
):
    # Convert comma-separated splits into a list
    splits_list = splits.split(',') if splits else []

    args = Namespace(
        dataset=dataset,
        data_subset=data_subset,
        splits=splits_list,
        out_root=out_root,
        compression=compression,
        concat_tokens=concat_tokens,
        tokenizer=tokenizer,
        tokenizer_kwargs=tokenizer,
        bos_text=bos_text,
        eos_text=eos_text,
        no_wrap=no_wrap,
        num_workers=num_workers,
    )
    convert_dataset_hf_from_args(args)


if __name__ == '__main__':
    app()
