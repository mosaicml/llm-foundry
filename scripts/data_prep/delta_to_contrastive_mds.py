# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from argparse import ArgumentParser, Namespace

from llmfoundry.command_utils import convert_delta_to_contrastive_mds

logger = logging.getLogger(__name__)


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description=
        'Download Delta table from UC and convert to JSON to save locally.',
    )
    parser.add_argument(
        '--delta_table_name',
        required=True,
        type=str,
        help='UC table <catalog>.<schema>.<table name>',
    )
    parser.add_argument(
        '--output_path',
        required=True,
        type=str,
        help='Local path to save the converted JSON',
    )
    parser.add_argument(
        '--http_path',
        required=False,
        type=str,
        help='http_path is set then dbsql method is used',
    )
    parser.add_argument(
        '--cluster_id',
        required=False,
        type=str,
        help=
        'Cluster ID with runtime newer than 14.1.0 and access mode of either assigned or shared can use databricks-connect.',
    )
    parser.add_argument(
        '--use_serverless',
        required=False,
        type=bool,
        default=False,
        help=
        'Use serverless or not. Make sure the workspace is entitled with serverless',
    )
    parser.add_argument(
        '--batch_size',
        required=False,
        type=int,
        default=1 << 30,
        help='Batch size for processing the data',
    )
    parser.add_argument(
        '--processes',
        required=False,
        type=int,
        default=os.cpu_count(),
        help='Number of processes to use for parallel processing',
    )
    parsed = parser.parse_args()
    return parsed


if __name__ == '__main__':
    args = parse_args()
    convert_delta_to_contrastive_mds(
        delta_table_name=args.delta_table_name,
        http_path=args.http_path,
        cluster_id=args.cluster_id,
        use_serverless=args.use_serverless,
        output_path=args.output_path,
        batch_size=args.batch_size,
        processes=args.processes,
    )
