# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import logging
from argparse import ArgumentParser

from llmfoundry.command_utils import convert_delta_to_mds_from_args

log = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = ArgumentParser(
        description=
        'Download Delta Table from UC and save as MDS shards in local folder',
    )
    parser.add_argument(
        '--delta_table_name',
        required=True,
        type=str,
        help='UC table <catalog>.<schema>.<table>',
    )
    parser.add_argument(
        '--mds_output_folder',
        required=True,
        type=str,
        help='Local path to save the converted MDS shards',
    )
    args = parser.parse_args()
    convert_delta_to_mds_from_args(
        delta_table_name=args.delta_table_name,
        mds_output_folder=args.mds_output_folder,
    )
