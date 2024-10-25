# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import tempfile
from argparse import ArgumentParser
from typing import Optional

from databricks.sql.client import Cursor
from pyspark.sql import SparkSession
from streaming import MDSWriter

from llmfoundry.command_utils.data_prep.convert_delta_to_json import (
    _check_imports,
    fetch_DT,
    run_query,
    validate_and_get_cluster_info,
)

logger = logging.getLogger(__name__)


def validate_columns_in_table(
    required_columns: list,
    optional_columns: list,
    table_name: str,
    method: str,
    cursor: Optional['Cursor'] = None,
    spark: Optional['SparkSession'] = None,
) -> bool:
    """Validate that required and optional columns exist in the Delta table."""
    try:
        # Reusing the run_query from the original script to fetch columns
        result = run_query(
            f'SHOW COLUMNS IN {table_name}',
            method,
            cursor,
            spark,
        )

        # Get the actual column names
        assert result
        actual_columns = [row.asDict()['col_name'] for row in result]

        missing_required = set(required_columns) - set(actual_columns)
        allowed_columns = set(required_columns + optional_columns)
        extra_columns = set(actual_columns) - allowed_columns

        if missing_required:
            logger.error(f'Missing required columns: {missing_required}')
            return False
        if extra_columns:
            logger.warning(f'Extra columns found: {extra_columns}')
            return False

        logger.info(
            f'Table {table_name} contains the required and optional columns.',
        )
        return True
    except Exception as e:
        logger.error(f'Error validating columns in table {table_name}: {e}')
        return False


def main(
    delta_table_name: str,
    http_path: Optional[str],
    cluster_id: Optional[str],
    use_serverless: bool,
    output_path: str,
    batch_size: int,
    processes: int,
):
    _check_imports()
    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient()
    DATABRICKS_HOST = w.config.host
    DATABRICKS_TOKEN = w.config.token

    logger.info(
        f'Validating columns in table {delta_table_name} and cluster info...',
    )
    dtypes = {
        'query_text': 'str',
        'positive_passage': 'str',
        'negative_passages': 'str',
    }
    required_columns = ['query_text', 'positive_passage']
    optional_columns = ['negative_passages']
    method, dbsql, sparkSession = validate_and_get_cluster_info(
        cluster_id=cluster_id,
        databricks_host=DATABRICKS_HOST,
        databricks_token=DATABRICKS_TOKEN,
        http_path=http_path,
        use_serverless=use_serverless,
    )
    logger.info(f'Validated cluster info')
    if not validate_columns_in_table(
        required_columns=required_columns,
        optional_columns=optional_columns,
        table_name=delta_table_name,
        method=method,
        cursor=dbsql.cursor() if dbsql else None,
        spark=sparkSession,
    ):
        logger.error('Column validation failed. Exiting.')
        raise ValueError('Column validation failed.')
    logger.info(f'Validated columns in table {delta_table_name}')

    compression = 'zstd:7'
    hashes = ['sha1']
    limit = '10mb'

    def convert_x(x: dict) -> dict:

        return {
            'query_text':
                x['query_text'],
            'positive_passage':
                x['positive_passage'],
            'negative_passages':
                json.dumps(x['negative_passages'])
                if 'negative_passages' in x else '[]',
        }

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info(f'Created temporary directory at {temp_dir}')
        json_output_path = os.path.join(temp_dir, 'output.jsonl')
        try:
            fetch_DT(
                delta_table_name=delta_table_name,
                json_output_folder=temp_dir,
                http_path=http_path,
                cluster_id=cluster_id,
                use_serverless=use_serverless,
                json_output_filename='output.jsonl',
                batch_size=batch_size,
                processes=processes,
                DATABRICKS_HOST=DATABRICKS_HOST,
                DATABRICKS_TOKEN=DATABRICKS_TOKEN,
            )
        except Exception as e:
            logger.error(f'Error fetching data: {e}')
            raise e
        with MDSWriter(
            out=output_path,
            columns=dtypes,
            compression=compression,
            hashes=hashes,
            size_limit=limit,
        ) as out:
            try:
                with open(json_output_path, 'r') as f:
                    for line in f:
                        out.write(convert_x(json.loads(line)))
            except FileNotFoundError as e:
                logger.error(f'JSON output file not found: {e}')
                raise e

    logger.info(f'Wrote to MDS at {output_path}')


if __name__ == '__main__':
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
    args = parser.parse_args()
    main(
        delta_table_name=args.delta_table_name,
        http_path=args.http_path,
        cluster_id=args.cluster_id,
        use_serverless=args.use_serverless,
        output_path=args.output_path,
        batch_size=args.batch_size,
        processes=args.processes,
    )
