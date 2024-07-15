# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated, Optional

from typer import Argument, Option, Typer

from llmfoundry.cli import registry_cli
from llmfoundry.data_prep import convert_dataset_json_from_args
from llmfoundry.train import train_from_yaml

app = Typer(pretty_exceptions_show_locals=False)
app.add_typer(registry_cli.app, name='registry')


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


@app.command(name='convert_dataset_json')
def convert_dataset_json(
    path: Annotated[str, Option(..., help='Path to the input data file')],
    out_root: Annotated[str, Option(..., help='Output root directory')],
    concat_tokens: Annotated[
        int,
        Option(
            ...,
            help='Convert text to tokens and concatenate up to this many tokens',
        )],
    tokenizer: Annotated[str, Option(..., help='Tokenizer name')],
    compression: Annotated[Optional[str],
                           Option(help='Compression type, if any')] = 'zstd',
    split: Annotated[str, Option(help='Dataset split to process')] = 'train',
    bos_text: Annotated[
        Optional[str],
        Option(help='Text to insert at the beginning of each sequence')] = None,
    eos_text: Annotated[
        Optional[str],
        Option(help='Text to insert at the end of each sequence')] = None,
    no_wrap: Annotated[
        bool,
        Option(help='Do not wrap text across max_length boundaries')] = False,
    num_workers: Annotated[
        Optional[int],
        Option(help='Number of workers for data loading')] = None,
):
    """Convert a dataset from JSON to MDS streaming format."""
    convert_dataset_json_from_args(
        path=path,
        split=split,
        out_root=out_root,
        compression=compression,
        concat_tokens=concat_tokens,
        tokenizer=tokenizer,
        bos_text=bos_text,
        eos_text=eos_text,
        no_wrap=no_wrap,
        num_workers=num_workers,
    )


if __name__ == '__main__':
    app()
