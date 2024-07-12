# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from argparse import Namespace
from typing import Optional

import typer

from llmfoundry.cli import registry_cli
from llmfoundry.data_prep import convert_finetuning_dataset_from_args
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


@app.command(name='convert_text_to_mds')
def convert_text_to_mds_cli(
    dataset: str = typer.Option(
        ...,
        '--dataset',
        help=
        'Name of the dataset (e.g., first argument to `datasets.load_dataset`, for jsonl data format, it is `json`).',
    ),  # type: ignore
    data_subset: Optional[str] = typer.Option(
        None,
        '--data_subset',
        help='(Optional) subset of data to use.',
    ),  # type: ignore
    splits: str = typer.Option(
        'train,validation',
        '--splits',
        help='Comma-separated list of dataset splits',
    ),  # type: ignore
    preprocessor: Optional[str] = typer.Option(
        None,
        '--preprocessor',
        help=
        'Name or import path of function used to preprocess (reformat) the dataset.',
    ),  # type: ignore
    data_files: str = typer.Option(
        '',
        '--data_files',
        help='Data file for each split. Comma-separated.',
    ),  # type: ignore
    skip_preprocessing: bool = typer.Option(
        False,
        '--skip_preprocessing',
        help='Whether to skip preprocessing.',
    ),  # type: ignore
    out_root: str = typer.Option(
        ...,
        '--out_root',
        help=
        'Root path of output directory where MDS shards will be stored. Can be a remote URI.',
    ),  # type: ignore
    local: Optional[str] = typer.Option(
        None,
        '--local',
        help=
        '(Optional) root path of local directory if you want to keep a local copy when out_root is remote.',
    ),  # type: ignore
    compression: Optional[str] = typer.Option(
        None,
        '--compression',
        help='(Optional) name of compression algorithm to use.',
    ),  # type: ignore
    num_workers: Optional[int] = typer.
    Option(None, '--num_workers', help='Number of workers.'),  # type: ignore
    tokenizer: Optional[str] = typer.Option(
        None,
        '--tokenizer',
        help='Tokenizer used for processing.',
    ),  # type: ignore
    tokenizer_kwargs: Optional[str] = typer.Option(
        None,
        '--tokenizer_kwargs',
        help='Keyword arguments for tokenizer initialization in JSON format.',
    ),  # type: ignore
    max_seq_len: int = typer.Option(
        2048,
        '--max_seq_len',
        help='Maximum sequence length.',
    ),  # type: ignore
    target_prompts: str = typer.Option(
        'none',
        '--target_prompts',
        help='Policy for when to use prompts as training targets.',
    ),  # type: ignore
    target_responses: str = typer.Option(
        'last',
        '--target_responses',
        help='Policy for which responses to treat as training targets.',
    ),  # type: ignore
    encoder_decoder: bool = typer.Option(
        False,
        '--encoder_decoder',
        help=
        'Set if the data are intended to be used to train an encoder-decoder model.',
    ),  # type: ignore
):
    # Convert comma-separated args
    splits_list = splits.split(',') if splits else []
    data_files_list = data_files.split(',') if data_files else []

    args = Namespace(
        dataset=dataset,
        data_subset=data_subset,
        splits=splits_list,
        preprocessor=preprocessor,
        data_files=data_files_list,
        skip_preprocessing=skip_preprocessing,
        out_root=out_root,
        local=local,
        compression=compression,
        num_workers=num_workers,
        tokenizer=tokenizer,
        tokenizer_kwargs=tokenizer_kwargs,
        max_seq_len=max_seq_len,
        target_prompts=target_prompts,
        target_responses=target_responses,
        encoder_decoder=encoder_decoder,
    )

    convert_finetuning_dataset_from_args(args)


if __name__ == '__main__':
    app()
