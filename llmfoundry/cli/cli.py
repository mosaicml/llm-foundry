# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Annotated, Optional

from typer import Argument, Option, Typer

from llmfoundry.cli import registry_cli
from llmfoundry.data_prep.convert_delta_to_json import \
    convert_delta_to_json_from_args
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


@app.command(name='convert_delta_to_json')
def convert_delta_to_json_cli(
    delta_table_name: Annotated[str, Option(..., help='UC table <catalog>.<schema>.<table name>')],
    json_output_folder: Annotated[str, Option(..., help='Local path to save the converted json')],
    http_path: Annotated[Optional[str], Option(help='If set, dbsql method is used')] = None,
    batch_size: Annotated[int, Option(help='Row chunks to transmit a time to avoid OOM')] = 1 << 30,
    processes: Annotated[int, Option(help='Number of processes allowed to use')] = os.cpu_count(), # type: ignore
    cluster_id: Annotated[Optional[str], Option(help='Cluster ID with runtime newer than 14.1.0 and access mode of either assigned or shared can use databricks-connect.')] = None,
    use_serverless: Annotated[bool, Option(help='Use serverless or not. Make sure the workspace is entitled with serverless')] = False,
    json_output_filename: Annotated[str, Option(help='The name of the combined final jsonl that combines all partitioned jsonl')] = 'train-00000-of-00001.jsonl',
):
    convert_delta_to_json_from_args(
        delta_table_name=delta_table_name,
        json_output_folder=json_output_folder,
        http_path=http_path,
        batch_size=batch_size,
        processes=processes,
        cluster_id=cluster_id,
        use_serverless=use_serverless,
        json_output_filename=json_output_filename,
    )


if __name__ == '__main__':
    app()
