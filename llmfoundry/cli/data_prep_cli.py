# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated, Optional

from typer import Option, Typer

from llmfoundry.command_utils import (
    convert_dataset_hf_from_args,
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
