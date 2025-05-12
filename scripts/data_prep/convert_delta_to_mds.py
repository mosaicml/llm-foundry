# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import tempfile
import traceback
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
    CONTINUED_PRETRAIN = 'CPT'
    INSTRUCTION_FINETUNE = 'IFT'
    CHAT_COMPLETION = 'CHAT'
    COMPARATIVE_EVALUATION = 'COMP_EVAL'


def get_conversion_config(
    columns: list[str],
    task_type: Optional[FinetuneTaskType] = None,
) -> tuple[dict, Callable]:
    """
    Returns dtyles and a conversion function for a given task type.
    If no task type is provided, attempts to infer config based on column names.

    Args:
        columns (list[str]): The list of column names.
        task_type (Optional[str]): One of FinetuneTaskType. If not provided, will attempt to infer config based on column names.

    Returns:
        tuple[dict, Callable]: A tuple of the dtypes and a conversion function.
    """
    if task_type is None:
        if len(columns) == 1:
            if 'turns' in columns[0]:
                task_type = FinetuneTaskType.CHAT_COMPLETION
            elif 'tokens' in columns[0]:
                task_type = FinetuneTaskType.CONTINUED_PRETRAIN
            else:
                raise ValueError(
                    'Unable to infer task type from columns and no task_type provided.',
                )
        elif len(columns) == 5 and set(columns) == {'prompt', 'chosen', 'rejected', 'chosen_reward', 'rejected_reward'}:
            task_type = FinetuneTaskType.COMPARATIVE_EVALUATION
        else:
            raise ValueError(
                'Unable to infer task type from columns and no task_type provided.',
            )
        logger.info(f'No task_type provided. Inferred task type: {task_type}')

    if task_type in [FinetuneTaskType.INSTRUCTION_FINETUNE, FinetuneTaskType.CHAT_COMPLETION]:
        dtypes = {
            'input_ids': 'ndarray',
            'attention_mask': 'ndarray',
            'labels': 'ndarray',
        }
        
        def convert_x(x):
            if len(x['turns']) > 1:
                raise ValueError('More than one turn found')
            
            # Debug the raw data
            print(f"IFT/CHAT turn data types: input_ids={type(x['turns'][0]['input_ids'])}, attention_mask={type(x['turns'][0]['attention_mask'])}, labels={type(x['turns'][0]['labels'])}")
            
            return {
                'input_ids': np.array(x['turns'][0]['input_ids'], dtype=np.int32),
                'attention_mask': np.array(x['turns'][0]['attention_mask'], dtype=np.int32),
                'labels': np.array(x['turns'][0]['labels'], dtype=np.int32),
            }
    
    elif task_type == FinetuneTaskType.CONTINUED_PRETRAIN:
        dtypes = {
            'tokens': 'ndarray',
        }
        
        def convert_x(x):
            # Debug the raw data
            print(f"CPT tokens type: {type(x['concat_tokens'])}, sample: {str(x['concat_tokens'][:10]) if x['concat_tokens'] else 'empty'}")
            
            return {'tokens': np.array(x['concat_tokens'], dtype=np.int32)}
    
    elif task_type == FinetuneTaskType.COMPARATIVE_EVALUATION:
        dtypes = {
            'prompt': 'bytes',
            'chosen': 'bytes',
            'rejected': 'bytes',
            'chosen_reward': 'float32',  # Spark FloatType is 4 bytes
            'rejected_reward': 'float32',
        }
        
        def convert_x(x):
            # Debug the raw data
            print(f"COMP_EVAL data types: prompt={type(x['prompt'])}, chosen={type(x['chosen'])}, rejected={type(x['rejected'])}")
            
            try:
                result = {
                    'prompt': np.array(x['prompt'], dtype=np.int32).tobytes(),
                    'chosen': np.array(x['chosen'], dtype=np.int32).tobytes(),
                    'rejected': np.array(x['rejected'], dtype=np.int32).tobytes(),
                    'chosen_reward': np.float32(x['chosen_reward']), 
                    'rejected_reward': np.float32(x['rejected_reward']),
                }
                
                # Debug the converted data
                print(f"Converted COMP_EVAL data types: prompt={result['prompt'].dtype}, chosen={result['chosen'].dtype}, rejected={result['rejected'].dtype}")
                
                return result
            except Exception as e:
                logger.error(f"Error in COMP_EVAL convert_x: {e}")
                logger.error(f"Problematic data: prompt={str(x['prompt'])[:50]}, chosen={str(x['chosen'])[:50]}, rejected={str(x['rejected'])[:50]}")
                raise
    
    else:
        raise ValueError(
            'Unable to infer dtypes from columns and no dtypes provided.',
        )

    return dtypes, convert_x


def debug_data_sample(data, prefix=""):
    """Helper function to debug data structures"""
    if not data:
        print(f"{prefix} Data is empty or None")
        return
    
    if isinstance(data, dict):
        print(f"{prefix} Dict with keys: {list(data.keys())}")
        for key, value in data.items():
            sample_value = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
            print(f"{prefix} - {key}: {type(value)}, sample: {sample_value}")
    elif isinstance(data, list):
        print(f"{prefix} List of length {len(data)}")
        if data:
            first_item = data[0]
            sample_value = str(first_item)[:100] + "..." if len(str(first_item)) > 100 else str(first_item)
            print(f"{prefix} - First item type: {type(first_item)}, sample: {sample_value}")
    else:
        print(f"{prefix} Data type: {type(data)}, sample: {str(data)[:100]}")


def safe_write(writer, data, record_index=None):
    """Helper function to safely write to MDS with better error reporting"""
    try:
        writer.write(data)
    except ValueError as e:
        error_msg = str(e)
        logger.error(f"ValueError when writing record {record_index}: {error_msg}")
        
        if "Unsupported dtype" in error_msg:
            logger.error("Detailed data type information:")
            for key, value in data.items():
                if hasattr(value, 'dtype'):
                    logger.error(f"Field {key}: dtype={value.dtype}, shape={value.shape}")
                    # If it's a string dtype, print some values to help debugging
                    if 'str' in str(value.dtype):
                        logger.error(f"First few values of {key}: {value.tolist()[:5] if value.size > 0 else 'empty'}")
                        
                        # Try to inspect the original values
                        logger.error(f"Attempting to fix string dtype in {key}")
                        try:
                            # Convert strings to integers if possible
                            if value.size > 0:
                                # Check if the first few values are numeric strings
                                sample_values = value.tolist()[:5]
                                logger.error(f"Sample values types: {[type(x) for x in sample_values]}")
                                
                                # Try to convert to int if it's a numeric string
                                fixed_val = np.array([int(x) if isinstance(x, str) and x.isdigit() else x for x in value], dtype=np.int32)
                                data[key] = fixed_val
                                logger.error(f"Fixed {key} to dtype={fixed_val.dtype}")
                        except Exception as fix_error:
                            logger.error(f"Failed to fix {key}: {fix_error}")
            
            # Try again with potentially fixed data
            try:
                writer.write(data)
                logger.info("Successfully wrote after attempting to fix data types")
            except Exception as retry_error:
                logger.error(f"Still failed after fixing: {retry_error}")
                # Print full traceback
                logger.error(traceback.format_exc())
                raise
        else:
            logger.error(traceback.format_exc())
            raise


def convert_delta_to_mds_from_args(
    delta_table_name: str,
    mds_output_folder: str,
    http_path: Optional[str],
    cluster_id: Optional[str],
    use_serverless: bool,
    batch_size: int,
    processes: int,
    task_type: Optional[FinetuneTaskType],
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
        task_type (Optional[FinetuneTaskType]): The type of finetune task. If not provided, the function will attempt to infer the task type from the column names.
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
    logger.info(f'Using dtypes configuration: {dtypes}')

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
        
        with MDSWriter(
            out=mds_output_folder,
            columns=dtypes,
            compression=compression,
            hashes=hashes,
            size_limit=limit,
        ) as out:
            try:
                with open(json_full_filepath, 'r') as f:
                    for i, line in enumerate(f):
                        try:
                            # Parse the JSON
                            data = json.loads(line)
                            
                            # Debug the first few records
                            if i < 3:
                                logger.info(f"Record {i} raw data sample:")
                                debug_data_sample(data, prefix=f"Record {i}")
                                
                                # Check for string values in arrays
                                for key, value in data.items():
                                    if isinstance(value, list) and value:
                                        if any(isinstance(item, str) for item in value[:10]):
                                            logger.warning(f"String values found in {key} array at record {i}")
                                            logger.warning(f"First few items: {value[:5]}")
                            
                            # Convert the data
                            try:
                                converted = convert_x(data)
                                
                                # Debug the converted data
                                if i < 3:
                                    logger.info(f"Record {i} converted data:")
                                    for key, value in converted.items():
                                        if hasattr(value, 'dtype'):
                                            logger.info(f"  - {key}: type={type(value)}, dtype={value.dtype}, shape={value.shape}")
                                            if 'str' in str(value.dtype):
                                                logger.warning(f"WARNING: String dtype detected in {key}: {value.dtype}")
                                
                                # Write with enhanced error handling
                                safe_write(out, converted, i)
                                
                            except Exception as convert_error:
                                logger.error(f"Error converting record {i}: {convert_error}")
                                logger.error(f"Problematic data: {str(data)[:500]}")
                                logger.error(traceback.format_exc())
                                raise
                                
                        except Exception as record_error:
                            logger.error(f"Error processing record {i}: {record_error}")
                            logger.error(traceback.format_exc())
                            raise
                            
            except FileNotFoundError as e:
                logger.error(f'JSON output file not found: {e}')
                raise e

    logging.info(f'Wrote to MDS at {mds_output_folder}')


if __name__ == '__main__':
    # Configure more verbose logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
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
        '--task_type',
        required=False,
        type=str,
        help='The type of finetune task',
    )
    args = parser.parse_args()

    try:
        task_type = FinetuneTaskType(args.task_type.upper()) if args.task_type else None
    except ValueError:
        logger.error(f'Invalid task type: {args.task_type}, will attempt to automatically infer task type.')
        task_type = None
        
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