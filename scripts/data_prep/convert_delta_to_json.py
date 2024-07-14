# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import re
from argparse import ArgumentParser

from databricks.sql.client import Connection as Connection
from databricks.sql.client import Cursor as Cursor

from llmfoundry.data_prep import convert_delta_to_json_from_args

MINIMUM_DB_CONNECT_DBR_VERSION = '14.1'
MINIMUM_SQ_CONNECT_DBR_VERSION = '12.2'

TABLENAME_PATTERN = re.compile(r'(\S+)\.(\S+)\.(\S+)')

log = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = ArgumentParser(
        description=
        'Download delta table from UC and convert to json to save local',
    )
    parser.add_argument(
        '--delta_table_name',
        required=True,
        type=str,
        help='UC table <catalog>.<schema>.<table name>',
    )
    parser.add_argument(
        '--json_output_folder',
        required=True,
        type=str,
        help='Local path to save the converted json',
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
        default=os.cpu_count(),
        help='number of processes allowed to use',
    )
    parser.add_argument(
        '--cluster_id',
        required=False,
        type=str,
        help=
        'cluster id has runtime newer than 14.1.0 and access mode of either assigned or shared can use databricks-connect.',
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
        '--json_output_filename',
        required=False,
        type=str,
        default='train-00000-of-00001.jsonl',
        help=
        'The name of the combined final jsonl that combines all partitioned jsonl',
    )
    args = parser.parse_args()
    convert_delta_to_json_from_args(
        delta_table_name=args.delta_table_name,
        json_output_folder=args.json_output_folder,
        http_path=args.http_path,
        batch_size=args.batch_size,
        processes=args.processes,
        cluster_id=args.cluster_id,
        use_serverless=args.use_serverless,
        json_output_filename=args.json_output_filename,
    )
