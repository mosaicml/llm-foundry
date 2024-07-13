# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import logging
from argparse import ArgumentParser, Namespace

import psutil

from llmfoundry.data_prep import convert_text_to_mds_from_args

log = logging.getLogger(__name__)

DONE_FILENAME = '.text_to_mds_conversion_done'


def parse_args() -> Namespace:
    """Parse commandline arguments."""
    parser = ArgumentParser(
        description=
        'Convert text files into MDS format, optionally concatenating and tokenizing',
    )
    parser.add_argument(
        '--output_folder',
        type=str,
        required=True,
        help='The folder to write output to',
    )
    parser.add_argument(
        '--input_folder',
        type=str,
        required=True,
        help='The folder with text files to convert to mds',
    )
    parser.add_argument(
        '--compression',
        type=str,
        default='zstd',
        required=False,
        help='The compression algorithm to use for MDS writing',
    )

    parser.add_argument(
        '--concat_tokens',
        type=int,
        required=True,
        help='Convert text to tokens and concatenate up to this many tokens',
    )

    parser.add_argument(
        '--tokenizer',
        type=str,
        required=True,
        help='The name of the tokenizer to use',
    )
    parser.add_argument(
        '--bos_text',
        type=str,
        required=False,
        default=None,
        help=
        'The text to prepend to each example to separate concatenated examples',
    )
    parser.add_argument(
        '--eos_text',
        type=str,
        required=False,
        default=None,
        help=
        'The text to append to each example to separate concatenated examples',
    )
    parser.add_argument(
        '--use_tokenizer_eos',
        required=False,
        action='store_true',
        default=False,
        help='Use the EOS text from the tokenizer.',
    )
    parser.add_argument(
        '--no_wrap',
        default=False,
        action='store_true',
        help=
        'Whether to let text examples wrap across multiple training examples',
    )
    parser.add_argument(
        '--processes',
        type=int,
        required=False,
        default=min(max(psutil.cpu_count() - 2, 1), 32),
        help=
        'The number of processes to use to download and convert the dataset',
    )
    parser.add_argument(
        '--reprocess',
        type=bool,
        required=False,
        default=False,
        help='If true, reprocess the input_folder to mds format. Otherwise, ' +
        'only reprocess upon changes to the input folder or dataset creation parameters.',
    )
    parser.add_argument(
        '--trust-remote-code',
        type=bool,
        required=False,
        default=False,
        help='If true, allows custom code to be executed to load the tokenizer',
    )
    parser.add_argument(
        '--logging-level',
        type=str,
        required=False,
        default='INFO',
        help='Logging level for the script. Default is INFO.',
    )
    parsed = parser.parse_args()
    return parsed


if __name__ == '__main__':
    args = parse_args()
    convert_text_to_mds_from_args(
        output_folder=args.output_folder,
        input_folder=args.input_folder,
        compression=args.compression,
        concat_tokens=args.concat_tokens,
        tokenizer=args.tokenizer,
        bos_text=args.bos_text,
        eos_text=args.eos_text,
        use_tokenizer_eos=args.use_tokenizer_eos,
        no_wrap=args.no_wrap,
        processes=args.processes,
        reprocess=args.reprocess,
        trust_remote_code=args.trust_remote_code,
        logging_level=args.logging,
    )
