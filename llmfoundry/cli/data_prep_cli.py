# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Annotated, Optional

import psutil
from typer import Option, Typer

from llmfoundry.command_utils import (
    convert_dataset_hf_from_args,
    convert_dataset_json_from_args,
    convert_delta_to_json_from_args,
    convert_finetuning_dataset_from_args,
    convert_text_to_mds_from_args,
)

app = Typer(pretty_exceptions_show_locals=False)


@app.command(name='convert_dataset_hf')
def convert_dataset_hf(
    dataset: Annotated[str, Option(..., help='Name of the dataset')],
    out_root: Annotated[str, Option(..., help='Output root directory')],
    data_subset: Annotated[
        Optional[str],
        Option(help='Subset of the dataset (e.g., "all" or "en")'),
    ] = None,
    splits: Annotated[str,
                      Option(help='Comma-separated list of dataset splits',),
                     ] = 'train, train_small, val, val_small, val_xsmall',
    compression: Annotated[Optional[str],
                           Option(help='Compression type')] = None,
    concat_tokens: Annotated[
        Optional[int],
        Option(help='Concatenate tokens up to this many tokens')] = None,
    tokenizer: Annotated[Optional[str],
                         Option(help='Tokenizer name')] = None,
    tokenizer_kwargs: Annotated[
        Optional[str],
        Option(help='Tokenizer keyword arguments in JSON format')] = None,
    bos_text: Annotated[Optional[str], Option(help='BOS text')] = None,
    eos_text: Annotated[Optional[str], Option(help='EOS text')] = None,
    no_wrap: Annotated[
        bool,
        Option(help='Do not wrap text across max_length boundaries'),
    ] = False,
    num_workers: Annotated[Optional[int],
                           Option(help='Number of workers')] = None,
):
    """Converts dataset from HuggingFace into JSON files."""
    # Convert comma-separated splits into a list
    splits_list = splits.split(',') if splits else []
    convert_dataset_hf_from_args(
        dataset=dataset,
        data_subset=data_subset,
        splits=splits_list,
        out_root=out_root,
        compression=compression,
        concat_tokens=concat_tokens,
        tokenizer=tokenizer,
        tokenizer_kwargs=tokenizer_kwargs,
        bos_text=bos_text,
        eos_text=eos_text,
        no_wrap=no_wrap,
        num_workers=num_workers,
    )


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


@app.command(name='convert_text_to_mds')
def convert_text_to_mds(
    output_folder: Annotated[str, Option(..., help='The folder to write output to')],
    input_folder: Annotated[str, Option(..., help='The folder with text files to convert to MDS')],
    concat_tokens: Annotated[int, Option(..., help='Convert text to tokens and concatenate up to this many tokens')],
    tokenizer: Annotated[str, Option(..., help='The name of the tokenizer to use')],
    bos_text: Annotated[Optional[str], Option(help='The text to prepend to each example to separate concatenated examples')] = None,
    eos_text: Annotated[Optional[str], Option(help='The text to append to each example to separate concatenated examples')] = None,
    compression: Annotated[str, Option(help='The compression algorithm to use for MDS writing')] = 'zstd',
    use_tokenizer_eos: Annotated[bool, Option(help='Use the EOS text from the tokenizer')] = False,
    no_wrap: Annotated[bool, Option(help='Whether to let text examples wrap across multiple training examples')] = False,
    processes: Annotated[int, Option(
        help='The number of processes to use to download and convert the dataset',
    )] = min(max(psutil.cpu_count() - 2, 1), 32), # type: ignore
    reprocess: Annotated[bool, Option(
        help=
        'If true, reprocess the input_folder to MDS format. Otherwise, only reprocess upon changes to the input folder or dataset creation parameters.',
    )] = False,
    trust_remote_code: Annotated[bool, Option(
        help='If true, allows custom code to be executed to load the tokenizer',
    )] = False,
    logging_level: Annotated[str, Option(
        help='Logging level for the script. Default is INFO.',
    )] = 'INFO',

):
    """Convert text files to MDS streaming format."""
    convert_text_to_mds_from_args(
        output_folder=output_folder,
        input_folder=input_folder,
        compression=compression,
        concat_tokens=concat_tokens,
        tokenizer_name=tokenizer,
        bos_text=bos_text,
        eos_text=eos_text,
        use_tokenizer_eos=use_tokenizer_eos,
        no_wrap=no_wrap,
        processes=processes,
        reprocess=reprocess,
        trust_remote_code=trust_remote_code,
        logging_level=logging_level,
    )


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
    """Convert a Delta table into JSON files."""
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
