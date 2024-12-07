# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import tempfile
from typing import Optional

from llmfoundry.command_utils.data_prep.convert_delta_to_json import (
    _check_imports,
    fetch_DT,
    format_tablename,
    get_columns_info,
    validate_and_get_cluster_info,
)
from llmfoundry.command_utils.data_prep.convert_finetuning_dataset import \
    convert_finetuning_dataset_from_args

logger = logging.getLogger(__name__)


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

        convert_finetuning_dataset_from_args(
            dataset='json',
            data_subset=None,
            splits=[''],
            preprocessor=None,
            data_files=[json_full_filepath],
            skip_preprocessing=True,
            out_root=mds_output_folder,
            local=None,
            compression='zstd:7',
            num_workers=processes,
            tokenizer=None,
            tokenizer_kwargs=None,
            max_seq_len=-1,
            target_prompts='',
            target_responses='',
            encoder_decoder=False,
        )

        logging.info(f'Wrote to MDS at {mds_output_folder}')
