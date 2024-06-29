# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import re
import time
import urllib.parse
from argparse import ArgumentParser, Namespace
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor
from typing import Iterable, List, Optional, Tuple, Union
from uuid import uuid4

import google.protobuf.any_pb2 as any_pb2
import lz4.frame
import pandas as pd
import pyarrow as pa
import requests
from composer.utils import retry
from databricks import sql
from databricks.connect import DatabricksSession
from databricks.sdk import WorkspaceClient
from databricks.sql.client import Connection as Connection
from databricks.sql.client import Cursor as Cursor
from packaging import version
from pyspark.sql import SparkSession
from pyspark.sql.connect.client.core import SparkConnectClient
from pyspark.sql.connect.dataframe import DataFrame
from pyspark.sql.dataframe import DataFrame as SparkDataFrame
from pyspark.sql.types import Row

from llmfoundry.utils import (
    maybe_create_mosaicml_logger,
    no_override_excepthook,
)
from llmfoundry.utils.exceptions import (
    ClusterDoesNotExistError,
    FailedToConnectToDatabricksError,
    FailedToCreateSQLConnectionError,
)

MINIMUM_DB_CONNECT_DBR_VERSION = '14.3'
MINIMUM_SQ_CONNECT_DBR_VERSION = '12.2'

log = logging.getLogger(__name__)

def iterative_combine_jsons(json_directory: str, output_file: str) -> None:
    """Combine jsonl files in json_directory into one big jsonl file.

    This function does not work for nested subdirectories.

    Args:
        json_directory(str): directory containing the JSONL files
        output_file(str): path to the output combined JSONL file
    """
    json_files = [f for f in os.listdir(json_directory) if f.endswith('.jsonl')]
    with open(output_file, 'w') as outfile:
        for file_name in json_files:
            with open(os.path.join(json_directory, file_name), 'r') as infile:
                for line in infile:
                    outfile.write(line)
    log.info('JSON files have been combined into a JSONL file.')


def run_query(
    query: str,
    method: str,
    cursor: Optional[Cursor] = None,
    spark: Optional[SparkSession] = None,
    collect: bool = True,
) -> Optional[Union[List[Row], DataFrame, SparkDataFrame]]:
    """Run SQL query via databricks-connect or databricks-sql.

    Args:
        query (str): sql query
        method (str): select from dbsql and dbconnect
        cursor (Optional[Cursor]): connection.cursor
        spark (Optional[SparkSession]): spark session
        collect (bool): whether to get the underlying data from spark dataframe
    """
    if method == 'dbsql':
        if cursor is None:
            raise ValueError(f'cursor cannot be None if using method dbsql')
        cursor.execute(query)
        if collect:
            return cursor.fetchall()
    elif method == 'dbconnect':
        if spark == None:
            raise ValueError(f'sparkSession is required for dbconnect')
        df = spark.sql(query)
        if collect:
            return df.collect()
        return df
    else:
        raise ValueError(f'Unrecognized method: {method}')


def get_args(signed: List, json_output_folder: str, columns: List) -> Iterable:
    for i, r in enumerate(signed):
        yield (i, r.url, json_output_folder, columns)


def download(
    ipart: int,
    url: str,
    json_output_folder: str,
    columns: Optional[List] = None,
    resp_format: str = 'arrow',
    compressed: bool = False,
) -> None:
    """Thread download presigned url and save to jsonl locally.

    Args:
        ipart (int): presigned url id
        url (str): presigned url
        json_output_folder (str): directory to save the ipart_th segment of dataframe
        columns (list): schema to save to json
        resp_format (str): whether to use arrow or json when collect
        compressed (bool): if data is compressed before downloading. Need decompress if compressed=True.
    """
    resp = requests.get(url)
    if resp.status_code == 200:
        if resp_format == 'json':
            data = resp.json()
            pd.DataFrame(data, columns=columns).to_json(
                os.path.join(
                    json_output_folder,
                    'part_' + str(ipart) + '.jsonl',
                ),
                orient='records',
                lines=True,
            )
            return

        # When resp_format is arrow:
        if compressed:
            # The data is lz4 compressed arrow format.
            # Decompress the data
            decompressed_data = lz4.frame.decompress(resp.content)
            # Convert the decompressed data into a PyArrow table
            reader = pa.ipc.open_stream(decompressed_data)
        else:
            reader = pa.ipc.open_stream(resp.content)
        table = reader.read_all()

        # Convert the PyArrow table into a pandas DataFrame
        df = table.to_pandas()
        df.to_json(
            os.path.join(json_output_folder, 'part_' + str(ipart) + '.jsonl'),
            orient='records',
            lines=True,
            force_ascii=False,
        )


def download_starargs(args: Tuple) -> None:
    return download(*args)


def fetch_data(
    method: str,
    cursor: Optional[Cursor],
    sparkSession: Optional[SparkSession],
    start: int,
    end: int,
    order_by: str,
    tablename: str,
    columns_str: str,
    json_output_folder: str,
) -> None:
    """Fetches a specified range of rows from a given table to a json file.

    This function executes a SQL query to retrieve a range of rows, determined by 'start' and 'end' indexes,
    from a specified table and column set. The fetched data is then exported as a JSON file.

    Args:
        method (str): The method to use for fetching data, either 'dbconnect' or 'dbsql'.
        cursor (Optional[Cursor]): The cursor object for executing queries in 'dbsql' method.
        sparkSession (Optional[SparkSession]): The Spark session object for executing queries in 'dbconnect' method.
        start (int): The starting index for row fetching.
        end (int): The ending index for row fetching.
        order_by (str): The column name to use for ordering the rows.
        tablename (str): The name of the table from which to fetch the data.
        columns_str (str): The string representation of the columns to select from the table.
        json_output_folder (str): The file path where the resulting JSON file will be saved.

    Returns:
        None: The function doesn't return any value, but writes the result to a JSONL file.
    """
    query = f"""
    WITH NumberedRows AS (
        SELECT
            *,
            ROW_NUMBER() OVER (ORDER BY {order_by}) AS rn
        FROM
            {tablename}
    )
    SELECT {columns_str}
    FROM NumberedRows
    WHERE rn BETWEEN {start+1} AND {end}"""

    if method == 'dbconnect':
        spark_df = run_query(query, method, cursor, sparkSession, collect=False)
        if spark_df is None:
            raise RuntimeError(
                f'Expect spark dataframe with {query} but got None',
            )
        pdf = spark_df.toPandas()  # pyright: ignore
    else:  #  method == 'dbsql':
        ans = run_query(query, method, cursor, sparkSession, collect=True)
        if ans is None:
            raise RuntimeError(f'Got empty results with {query}')
        records = [r.asDict() for r in ans]  # pyright: ignore
        pdf = pd.DataFrame.from_dict(records)

    pdf.to_json(
        os.path.join(json_output_folder, f'part_{start+1}_{end}.jsonl'),
        orient='records',
        lines=True,
    )


@retry(Exception, num_attempts=5, initial_backoff=1.0, max_jitter=0.5)
def get_total_rows(
    tablename: str,
    method: str,
    cursor: Optional[Cursor],
    sparkSession: Optional[SparkSession],
):
    ans = run_query(
        f'SELECT COUNT(*) FROM {tablename}',
        method,
        cursor,
        sparkSession,
    )
    nrows = [row.asDict() for row in ans][0].popitem()[1]  # pyright: ignore
    log.info(f'total_rows = {nrows}')
    return nrows


@retry(Exception, num_attempts=5, initial_backoff=1.0, max_jitter=0.5)
def get_columns_info(
    tablename: str,
    method: str,
    cursor: Optional[Cursor],
    sparkSession: Optional[SparkSession],
):
    ans = run_query(
        f'SHOW COLUMNS IN {tablename}',
        method,
        cursor,
        sparkSession,
    )
    columns = [row.asDict().popitem()[1] for row in ans]  # pyright: ignore
    order_by = columns[0]
    columns_str = ','.join(columns)
    log.info(f'order by column {order_by}')
    return columns, order_by, columns_str


def fetch(
    method: str,
    tablename: str,
    json_output_folder: str,
    batch_size: int = 1 << 30,
    processes: int = 1,
    sparkSession: Optional[SparkSession] = None,
    dbsql: Optional[Connection] = None,
) -> None:
    """Fetch UC delta table with databricks-connect as JSONL.

    Args:
        method (str): dbconnect or dbsql
        tablename (str): catalog.scheme.tablename on UC
        json_output_folder (str): path to write the result json file to
        batch_size (int): number of rows that dbsql fetches each time to avoid OOM
        processes (int): max number of processes to use to parallelize the fetch
        sparkSession (pyspark.sql.sparksession): spark session
        dbsql (databricks.sql.connect): dbsql session
    """
    cursor = dbsql.cursor() if dbsql is not None else None
    try:
        nrows = get_total_rows(
            tablename,
            method,
            cursor,
            sparkSession,
        )
    except Exception as e:
        raise RuntimeError(
            f'Error in get rows from {tablename}. Restart sparkSession and try again',
        ) from e

    try:
        columns, order_by, columns_str = get_columns_info(
            tablename,
            method,
            cursor,
            sparkSession,
        )
    except Exception as e:
        raise RuntimeError(
            f'Error in get columns from {tablename}. Restart sparkSession and try again',
        ) from e

    if method == 'dbconnect' and sparkSession is not None:
        log.info(f'{processes=}')
        df = sparkSession.table(tablename)

        # Running the query and collecting the data as arrow or json.
        query = df._plan.to_proto(df._session.client)  # pyright: ignore
        schema, cloudfetch_results = df._session.client.experimental_to_cloudfetch(query, "arrow", compression=False)  # pyright: ignore
        signed = [result.url for result in cloudfetch_results]
        log.info(f'len(signed) = {len(signed)}')

        args = get_args(signed, json_output_folder, columns)

        # Stopping the SparkSession to avoid spilling connection state into the subprocesses.
        sparkSession.stop()

        with ProcessPoolExecutor(max_workers=processes) as executor:
            list(executor.map(download_starargs, args))

    elif method == 'dbsql' and cursor is not None:
        for start in range(0, nrows, batch_size):
            log.warning(f'batch {start}')
            end = min(start + batch_size, nrows)
            fetch_data(
                method,
                cursor,
                sparkSession,
                start,
                end,
                order_by,
                tablename,
                columns_str,
                json_output_folder,
            )

    if cursor is not None:
        cursor.close()


def validate_and_get_cluster_info(
    cluster_id: str,
    databricks_host: str,
    databricks_token: str,
    http_path: Optional[str],
    use_serverless: bool = False,
) -> tuple:
    """Validate and get cluster info for running the Delta to JSONL conversion.

    Args:
        cluster_id (str): cluster id to validate and fetch additional info for
        databricks_host (str): databricks host name
        databricks_token (str): databricks auth token
        http_path (Optional[str]): http path to use for sql connect
        use_serverless (bool): whether to use serverless or not
    """
    method = 'dbsql'
    dbsql = None
    sparkSession = None

    if use_serverless:
        method = 'dbconnect'
    else:
        w = WorkspaceClient()
        res = w.clusters.get(cluster_id=cluster_id)
        if res is None:
            raise ClusterDoesNotExistError(cluster_id)

        assert res.spark_version is not None
        stripped_runtime = re.sub(
            r'[a-zA-Z]',
            '',
            res.spark_version.split('-scala')
            [0].replace(  # type: ignore
                'x-snapshot', '',
            ),
        )
        runtime_version = re.sub(r'[.-]*$', '', stripped_runtime)
        if version.parse(
            runtime_version,
        ) < version.parse(MINIMUM_SQ_CONNECT_DBR_VERSION):
            raise ValueError(
                f'The minium DBR version required is {MINIMUM_SQ_CONNECT_DBR_VERSION} but got {version.parse(runtime_version)}',
            )

        if http_path is None and version.parse(
            runtime_version,
        ) >= version.parse(MINIMUM_DB_CONNECT_DBR_VERSION):
            method = 'dbconnect'

    if method == 'dbconnect':
        try:
            if use_serverless:
                session_id = str(uuid4())
                sparkSession = DatabricksSession.builder.host(
                    databricks_host,
                ).token(
                    databricks_token,
                ).header('x-databricks-session-id', session_id).getOrCreate()

            else:
                sparkSession = DatabricksSession.builder.remote(
                    host=databricks_host,
                    token=databricks_token,
                    cluster_id=cluster_id,
                ).getOrCreate()

        except Exception as e:
            raise FailedToConnectToDatabricksError() from e
    else:
        try:
            dbsql = sql.connect(
                server_hostname=re.compile(r'^https?://').sub(
                    '', databricks_host).strip(
                    ),  # sqlconnect hangs if hostname starts with https
                http_path=http_path,
                access_token=databricks_token,
            )
        except Exception as e:
            raise FailedToCreateSQLConnectionError() from e
    return method, dbsql, sparkSession


def fetch_DT(args: Namespace) -> None:
    """Fetch UC Delta Table to local as jsonl."""
    log.info(f'Start .... Convert delta to json')

    obj = urllib.parse.urlparse(args.json_output_folder)
    if obj.scheme != '':
        raise ValueError(
            'Check the json_output_folder and verify it is a local path!',
        )

    if os.path.exists(args.json_output_folder):
        if not os.path.isdir(args.json_output_folder) or os.listdir(
            args.json_output_folder,
        ):
            raise RuntimeError(
                f'Output folder {args.json_output_folder} already exists and is not empty. Please remove it and retry.',
            )

    os.makedirs(args.json_output_folder, exist_ok=True)

    if not args.json_output_filename.endswith('.jsonl'):
        raise ValueError('json_output_filename needs to be a jsonl file')

    log.info(f'Directory {args.json_output_folder} created.')

    method, dbsql, sparkSession = validate_and_get_cluster_info(
        cluster_id=args.cluster_id,
        databricks_host=args.DATABRICKS_HOST,
        databricks_token=args.DATABRICKS_TOKEN,
        http_path=args.http_path,
        use_serverless=args.use_serverless,
    )

    fetch(
        method,
        args.delta_table_name,
        args.json_output_folder,
        args.batch_size,
        args.processes,
        sparkSession,
        dbsql,
    )

    if dbsql is not None:
        dbsql.close()

    # combine downloaded jsonl into one big jsonl for IFT
    iterative_combine_jsons(
        args.json_output_folder,
        os.path.join(args.json_output_folder, args.json_output_filename),
    )


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
        f'cluster id has runtime newer than {MINIMUM_DB_CONNECT_DBR_VERSION} and access mode of either assigned or shared can use databricks-connect.',
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
    mosaicml_logger = maybe_create_mosaicml_logger()

    try:
        w = WorkspaceClient()
        args.DATABRICKS_HOST = w.config.host
        args.DATABRICKS_TOKEN = w.config.token

        tik = time.time()
        fetch_DT(args)
        log.info(f'Elapsed time {time.time() - tik}')

    except Exception as e:
        if mosaicml_logger is not None and no_override_excepthook():
            mosaicml_logger.log_exception(e)
        raise e
