# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import os
from argparse import Namespace
from typing import Optional

import typer

from llmfoundry.cli import registry_cli
from llmfoundry.data_prep.convert_dataset_json import \
    convert_delta_to_json_from_args
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


@app.command(name='convert_delta_to_json')
def convert_delta_to_json_cli(
    delta_table_name: str = typer.Option(
        ...,
        help='UC table <catalog>.<schema>.<table name>',
    ),  # type: ignore
    json_output_folder: str = typer.Option(
        ...,
        help='Local path to save the converted json',
    ),  # type: ignore
    http_path: Optional[str] = typer.Option(
        None,
        help='If set, dbsql method is used',
    ),  # type: ignore
    batch_size: int = typer.Option(
        1 << 30,
        help='Row chunks to transmit a time to avoid OOM',
    ),  # type: ignore
    processes: int = typer.Option(
        os.cpu_count(), # type: ignore
        help='Number of processes allowed to use',
    ),  # type: ignore
    cluster_id: Optional[str] = typer.Option(
        None,
        help=
        'Cluster ID with runtime newer than 14.1.0 and access mode of either assigned or shared can use databricks-connect.',
    ),  # type: ignore
    use_serverless: bool = typer.Option(
        False,
        help=
        'Use serverless or not. Make sure the workspace is entitled with serverless',
    ),  # type: ignore
    json_output_filename: str = typer.Option(
        'train-00000-of-00001.jsonl',
        help=
        'The name of the combined final jsonl that combines all partitioned jsonl',
    ),  # type: ignore
):
    args = Namespace(
        delta_table_name=delta_table_name,
        json_output_folder=json_output_folder,
        http_path=http_path,
        batch_size=batch_size,
        processes=processes,
        cluster_id=cluster_id,
        use_serverless=use_serverless,
        json_output_filename=json_output_filename,
    )

    convert_delta_to_json_from_args(args)


if __name__ == '__main__':
    app()
