# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated, Optional

import psutil
from typer import Argument, Option, Typer

from llmfoundry.cli import registry_cli
from llmfoundry.data_prep import convert_text_to_mds_from_args
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


if __name__ == '__main__':
    app()
