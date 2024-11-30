# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import tempfile
from typing import Callable, Optional

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


def get_conversion_config(
    columns: list[str],
    provided_dtypes: Optional[dict],
) -> tuple[dict, Callable]:
    """If no dtypes is provided, attempts to infer config based on column names.

    Args:
        columns (List[str]): The list of column names.
        provided_dtypes (Optional[Dict]): The provided dtypes.
    """
    if provided_dtypes is not None:
        convert_x = lambda x: {
            k: np.array(v, dtype=provided_dtypes.get(k)) for k, v in x.items()
        }
        return provided_dtypes, convert_x

    if len(columns) != 1:
        raise ValueError(
            'Unable to infer dtypes from columns and no dtypes provided.',
        )

    if 'turns' in columns[0]:
        logging.info('Identified IFT data')
        dtypes = {
            'input_ids': 'ndarray',
            'attention_mask': 'ndarray',
            'labels': 'ndarray',
        }
        convert_x = lambda x: {
            # join the turns into a single array
            'input_ids':
                np.concatenate([
                    np.array(turn['input_ids']) for turn in x['turns']
                ]),
            'attention_mask':
                np.concatenate([
                    np.array(turn['attention_mask']) for turn in x['turns']
                ]),
            'labels':
                np.
                concatenate([np.array(turn['labels']) for turn in x['turns']]),
        }
    elif 'concat_tokens' in columns[0]:
        logging.info('Identified CPT data')
        dtypes = {
            'tokens': 'ndarray',
        }
        convert_x = lambda x: {'tokens': np.array(x['concat_tokens'])}
    else:
        raise ValueError(
            'Unable to infer dtypes from columns and no dtypes provided.',
        )

    return dtypes, convert_x


def convert_delta_to_mds_from_args(
    delta_table_name: str,
    mds_output_folder: str,
    http_path: Optional[str],
    cluster_id: Optional[str],
    use_serverless: bool,
    batch_size: int,
    processes: int,
    dtypes: Optional[dict[str, str]],
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
        dtypes (Optional[Dict[str, str]]): Mapping between column name and dtype, where dtype is supported for MDS conversion.
                                           If not provided, the function will attempt to infer the dtype.
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

    dtypes, convert_x = get_conversion_config(columns, dtypes)

    compression = 'zstd:7'
    hashes = ['sha1']
    limit = '10mb'

    logging.info(f'Fetching data from Delta Table {delta_table_name}...')

    with tempfile.TemporaryDirectory() as json_out_folder:
        json_out_filename = 'temp.jsonl'
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
                    for line in f:
                        out.write(convert_x(json.loads(line)))
            except FileNotFoundError as e:
                logger.error(f'JSON output file not found: {e}')
                raise e

    logging.info(f'Wrote to MDS at {mds_output_folder}')
