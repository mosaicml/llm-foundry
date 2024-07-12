# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from argparse import Namespace
from typing import Optional

import psutil
import typer

from llmfoundry.cli import registry_cli
from llmfoundry.data_prep import convert_text_to_mds_from_args
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
    output_folder: str = typer.
    Option(..., '--output_folder', help='The folder to write output to'),  # type: ignore
    input_folder: str = typer.Option(
        ..., '--input_folder', help='The folder with text files to convert to MDS',
    ),  # type: ignore
    compression: str = typer.Option(
        'zstd', '--compression', help='The compression algorithm to use for MDS writing',
    ),  # type: ignore
    concat_tokens: int = typer.Option(
        ...,
        '--concat_tokens',
        help='Convert text to tokens and concatenate up to this many tokens',
    ),  # type: ignore
    tokenizer: str = typer.Option(..., '--tokenizer', help='The name of the tokenizer to use',
                                 ),  # type: ignore
    bos_text: Optional[str] = typer.Option(
        None,
        '--bos_text',
        help=
        'The text to prepend to each example to separate concatenated examples',
    ),  # type: ignore
    eos_text: Optional[str] = typer.Option(
        None,
        '--eos_text',
        help=
        'The text to append to each example to separate concatenated examples',
    ),  # type: ignore
    use_tokenizer_eos: bool = typer.
    Option(False, '--use_tokenizer_eos', help='Use the EOS text from the tokenizer.'),  # type: ignore
    no_wrap: bool = typer.Option(
        False,
        '--no_wrap',
        help='Whether to let text examples wrap across multiple training examples',
    ),  # type: ignore
    processes: int = typer.Option(
        min(max(psutil.cpu_count() - 2, 1), 32), # type: ignore
        '--processes',
        help='The number of processes to use to download and convert the dataset',
    ),  # type: ignore
    reprocess: bool = typer.Option(
        False,
        '--reprocess',
        help=
        'If true, reprocess the input_folder to MDS format. Otherwise, only reprocess upon changes to the input folder or dataset creation parameters.',
    ),  # type: ignore
    trust_remote_code: bool = typer.Option(
        False,
        '--trust_remote_code',
        help='If true, allows custom code to be executed to load the tokenizer',
    ),  # type: ignore
    logging_level: str = typer.Option(
        'INFO', '--logging_level', help='Logging level for the script. Default is INFO.',
    ),  # type: ignore
):
    args = Namespace(
        output_folder=output_folder,
        input_folder=input_folder,
        compression=compression,
        concat_tokens=concat_tokens,
        tokenizer=tokenizer,
        bos_text=bos_text,
        eos_text=eos_text,
        use_tokenizer_eos=use_tokenizer_eos,
        no_wrap=no_wrap,
        processes=processes,
        reprocess=reprocess,
        trust_remote_code=trust_remote_code,
        logging_level=logging_level,
    )
    convert_text_to_mds_from_args(args)


if __name__ == '__main__':
    app()
