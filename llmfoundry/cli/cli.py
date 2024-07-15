# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated, Optional

from typer import Argument, Option, Typer

from llmfoundry.cli import registry_cli
from llmfoundry.data_prep import convert_finetuning_dataset_from_args
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


@app.command(name='convert_finetuning_dataset')
def convert_finetuning_dataset_cli(
    dataset: Annotated[
        str,
        Option(
            ...,
            help=
            'Name of the dataset (e.g., first argument to `datasets.load_dataset`, for jsonl data format, it is `json`).',
        )],
    data_subset: Annotated[
        Optional[str],
        Option(help='(Optional) subset of data to use.',)] = None,
    splits: Annotated[str,
                      Option(help='Comma-separated list of dataset splits'),
                     ] = 'train,validation',
    preprocessor: Annotated[
        Optional[str],
        Option(
            help=
            'Name or import path of function used to preprocess (reformat) the dataset.',
        )] = None,
    data_files: Annotated[
        str, Option(help='Data file for each split. Comma-separated.')] = '',
    skip_preprocessing: Annotated[
        bool, Option(help='Whether to skip preprocessing.')] = False,
    out_root: Annotated[
        str,
        Option(
            ...,
            help=
            'Root path of output directory where MDS shards will be stored. Can be a remote URI.',
        )] = '',
    local: Annotated[
        Optional[str],
        Option(
            help=
            '(Optional) root path of local directory if you want to keep a local copy when out_root is remote.',
        )] = None,
    compression: Annotated[
        Optional[str],
        Option(help='(Optional) name of compression algorithm to use.')] = None,
    num_workers: Annotated[Optional[int],
                           Option(help='Number of workers.')] = None,
    tokenizer: Annotated[Optional[str],
                         Option(help='Tokenizer used for processing.')] = None,
    tokenizer_kwargs: Annotated[
        Optional[str],
        Option(
            help=
            'Keyword arguments for tokenizer initialization in JSON format.',
        )] = None,
    max_seq_len: Annotated[int, Option(help='Maximum sequence length.')] = 2048,
    target_prompts: Annotated[
        str,
        Option(help='Policy for when to use prompts as training targets.'),
    ] = 'none',
    target_responses: Annotated[
        str,
        Option(help='Policy for which responses to treat as training targets.'),
    ] = 'last',
    encoder_decoder: Annotated[
        bool,
        Option(
            help=
            'Set if the data are intended to be used to train an encoder-decoder model.',
        )] = False,
):
    """Convert a Finetuning Dataset to MDS streaming format."""
    # Convert comma-separated args
    splits_list = splits.split(',') if splits else []
    data_files_list = data_files.split(',') if data_files else []
    convert_finetuning_dataset_from_args(
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


if __name__ == '__main__':
    app()
