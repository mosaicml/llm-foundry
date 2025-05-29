# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import tempfile
from enum import Enum
from typing import Callable, Optional
from argparse import ArgumentParser

import numpy as np
from streaming import MDSWriter

from llmfoundry.command_utils.data_prep.convert_delta_to_json import (
    _check_imports,
    fetch_DT,
    format_tablename,
    get_columns_info,
    validate_and_get_cluster_info,
)

logger = logging.getLogger(__name__)


class FinetuneTaskType(Enum):
    CHAT_COMPLETION = 'CHAT_COMPLETION'
    CONTINUED_PRETRAIN = 'CONTINUED_PRETRAIN'
    INSTRUCTION_FINETUNE = 'INSTRUCTION_FINETUNE'
    SPECIAL = 'SPECIAL'


def get_conversion_config(
    columns: list[str],
    task_type: FinetuneTaskType,
) -> tuple[dict, Callable]:
    """
    Returns dtypes and a conversion function for a given task type.

    Args:
        columns (list[str]): The list of column names.
        task_type (str): One of FinetuneTaskType.

    Returns:
        tuple[dict, Callable]: A tuple of the dtypes and a conversion function.
    """
    if task_type in [
            FinetuneTaskType.INSTRUCTION_FINETUNE,
            FinetuneTaskType.CHAT_COMPLETION,
    ]:
        dtypes = {
            'input_ids': 'ndarray',
            'attention_mask': 'ndarray',
            'labels': 'ndarray',
        }
        convert_x = lambda x: (
            ValueError('More than one turn found')
            if len(x['turns']) > 1 else {
                'input_ids': np.array(x['turns'][0]['input_ids']),
                'attention_mask': np.array(x['turns'][0]['attention_mask']),
                'labels': np.array(x['turns'][0]['labels']),
            })
    elif task_type == FinetuneTaskType.CONTINUED_PRETRAIN:
        dtypes = {
            'tokens': 'ndarray',
        }
        convert_x = lambda x: {'tokens': np.array(x['concat_tokens'])}
    elif task_type == FinetuneTaskType.SPECIAL:
        dtypes = {
            'prompt': 'bytes',
            'chosen': 'bytes',
            'rejected': 'bytes',
            'chosen_reward': 'float32',  # Spark FloatType is 4 bytes
            'rejected_reward': 'float32',
        }
        convert_x = lambda x: {
            'prompt': np.array(x['prompt']).tobytes(),
            'chosen': np.array(x['chosen']).tobytes(),
            'rejected': np.array(x['rejected']).tobytes(),
            'chosen_reward': np.float32(x['chosen_reward']),
            'rejected_reward': np.float32(x['rejected_reward']),
        }
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    return dtypes, convert_x


def convert_delta_to_mds_from_args(
    delta_table_name: str,
    mds_output_folder: str,
    http_path: Optional[str],
    cluster_id: Optional[str],
    use_serverless: bool,
    batch_size: int,
    processes: int,
    task_type: FinetuneTaskType,
) -> None:
    """A wrapper for convert_delta_to_mds that parses arguments.

    Args:
        delta_table_name (str): The name of the delta table to convert.
        mds_output_folder (str): The folder to output MDS shards.
        http_path (Optional[str]): If set, dbsql method is used
        batch_size (int): Row chunks to transmit a time to avoid OOM
        processes (int): Number of processes allowed to use
        cluster_id (Optional[str]): Cluster ID with runtime newer than 14.1.0 and access mode of either assigned or shared can use databricks-connect.
        use_serverless (bool): Use serverless or not. Make sure the workspace is entitled with serverless
        task_type (FinetuneTaskType): The type of finetune task.
    """
    _check_imports()
    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient()
    DATABRICKS_HOST = w.config.host
    DATABRICKS_TOKEN = w.config.token

    method, dbsql, sparkSession = validate_and_get_cluster_info(
        cluster_id=cluster_id,
        databricks_host=DATABRICKS_HOST,
        databricks_token=DATABRICKS_TOKEN,
        http_path=http_path,
        use_serverless=use_serverless,
    )
    cursor = dbsql.cursor() if dbsql is not None else None
    columns, _, _ = get_columns_info(
        tablename=format_tablename(delta_table_name),
        method=method,
        cursor=cursor,
        sparkSession=sparkSession,
    )
    logger.info(f'Columns: {columns}')

    dtypes, convert_x = get_conversion_config(columns, task_type)

    compression = 'zstd:7'
    hashes = ['sha1']
    limit = '10mb'

    logging.info(f'Fetching data from Delta Table {delta_table_name}...')

    with tempfile.TemporaryDirectory() as json_out_folder:
        json_out_filename = 'train.jsonl'
        json_full_filepath = os.path.join(json_out_folder, json_out_filename)
        try:
            fetch_DT(
                delta_table_name=delta_table_name,
                json_output_folder=json_out_folder,
                http_path=http_path,
                batch_size=batch_size,
                processes=processes,
                cluster_id=cluster_id,
                use_serverless=use_serverless,
                json_output_filename=json_out_filename,
                DATABRICKS_HOST=DATABRICKS_HOST,
                DATABRICKS_TOKEN=DATABRICKS_TOKEN,
            )
        except Exception as e:
            logger.error(f'Error fetching data from Delta Table: {e}')
            raise e
        
        json_num_lines = sum(1 for _ in open(json_full_filepath, 'r'))
        print(f'Number of lines in JSON file: {json_num_lines}', flush=True)
        
        with MDSWriter(
                out=mds_output_folder,
                columns=dtypes,
                compression=compression,
                hashes=hashes,
                size_limit=limit,
        ) as out:
            try:
                with open(json_full_filepath, 'r') as f:
                    for idx, line in enumerate(f):
                        print(f'Processing line {idx}: {line}', flush=True)
                        converted = convert_x(json.loads(line))
                        print(f'Wrote converted data: {converted}', flush=True)
                        out.write(converted)

            except FileNotFoundError as e:
                logger.error(f'JSON output file not found: {e}')
                raise e

    logging.info(f'Wrote to MDS at {mds_output_folder}')

    # print out the files in mds_output_folder and their sizes
    print("FILES IN MDS OUTPUT FOLDER AND THEIR SIZES:", flush=True)
    for root, _, files in os.walk(mds_output_folder):
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            print(f'File: {file_path}, Size: {file_size} bytes', flush=True)


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
        '--task_type',
        required=True,
        type=str.upper,
        choices=FinetuneTaskType.__members__.keys(),
        help='The type of finetune task',
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
    args = parser.parse_args()

    task_type = FinetuneTaskType(args.task_type)

    convert_delta_to_mds_from_args(
        delta_table_name=args.delta_table_name,
        mds_output_folder=args.mds_output_folder,
        http_path=args.http_path,
        cluster_id=args.cluster_id,
        use_serverless=args.use_serverless,
        batch_size=args.batch_size,
        processes=args.processes,
        task_type=task_type,
    )