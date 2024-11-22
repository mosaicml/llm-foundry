# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import time

from streaming.base.converters import dataframe_to_mds

from llmfoundry.command_utils.data_prep.convert_delta_to_json import (
    _check_imports,
    format_tablename,
    handle_fetch_exception,
    validate_and_get_cluster_info,
    validate_output_folder,
)

log = logging.getLogger(__name__)


def fetch_DT_mds(
    delta_table_name: str,
    mds_output_folder: str,
    DATABRICKS_HOST: str,
    DATABRICKS_TOKEN: str,
) -> None:
    """Fetch UC Delta Table and convert to MDS shards."""
    log.info(f'Converting Delta Table {delta_table_name} to MDS shards.')

    validate_output_folder(mds_output_folder)

    method, _, sparkSession = validate_and_get_cluster_info(
        cluster_id=None,
        databricks_host=DATABRICKS_HOST,
        databricks_token=DATABRICKS_TOKEN,
        http_path=None,
        use_serverless=True,
    )

    formatted_delta_table_name = format_tablename(delta_table_name)

    try:
        if method == 'dbconnect' and sparkSession is not None:
            df = sparkSession.table(formatted_delta_table_name)

            mds_kwargs = {
                'out': mds_output_folder,
                'keep_local': True,
                'compression': None,
            }
            dataframe_to_mds(
                df,
                merge_index=True,
                mds_kwargs=mds_kwargs,
            )
        else:
            raise NotImplementedError('Currently only dbconnect is supported.')
    except Exception as e:
        handle_fetch_exception(e, formatted_delta_table_name)


def convert_delta_to_mds_from_args(
    delta_table_name: str,
    mds_output_folder: str,
) -> None:
    """A wrapper for convert_delta_to_mds that parses arguments. Currently only
    supports dbconnect on severless compute and not dbsql.

    Args:
        delta_table_name (str): The name of the delta table to convert.
        mds_output_folder (str): The folder to output MDS shards.
    """
    os.environ['WORLD_SIZE'] = '1'
    _check_imports()
    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient()
    DATABRICKS_HOST = w.config.host
    DATABRICKS_TOKEN = w.config.token

    tik = time.time()
    fetch_DT_mds(
        delta_table_name=delta_table_name,
        mds_output_folder=mds_output_folder,
        DATABRICKS_HOST=DATABRICKS_HOST,
        DATABRICKS_TOKEN=DATABRICKS_TOKEN,
    )
    log.info(f'convert_delta_to_mds took {time.time() - tik} seconds.')
