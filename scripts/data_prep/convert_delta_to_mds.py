# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import json
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
        help='Local path to save the MDS shards',
    )
    parser.add_argument(
        '--http_path',
        required=False,
        type=str,
        help='http_path is set then dbsql method is used',
    )
    parser.add_argument(
        '--batch_size',
        required=False,
        type=int,
        default=1 << 30,
        help='row chunks to transmit a time to avoid OOM',
    )
    parser.add_argument(
        '--processes',
        required=False,
        type=int,
        default=1,
        help='number of processes allowed to use',
    )
    parser.add_argument(
        '--cluster_id',
        required=False,
        type=str,
        help='cluster id to use for serverless',
    )
    parser.add_argument(
        '--use_serverless',
        required=False,
        action='store_true',
        help='use serverless cluster',
    )
    parser.add_argument(
        '--dtypes',
        required=False,
        type=str,
        help='data types for columns',
    )
    args = parser.parse_args()

    if args.dtypes is not None:
        args.dtypes = json.loads(args.dtypes)

    convert_delta_to_mds_from_args(
        delta_table_name=args.delta_table_name,
        mds_output_folder=args.mds_output_folder,
        http_path=args.http_path,
        cluster_id=args.cluster_id,
        use_serverless=args.use_serverless,
        batch_size=args.batch_size,
        processes=args.processes,
        dtypes=args.dtypes,
    )
