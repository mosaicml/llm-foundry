# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import re
import time
import urllib.parse
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor
from typing import Iterable, List, Optional, Tuple, Union
from uuid import uuid4

import google.protobuf.any_pb2 as any_pb2
import lz4.frame
import pandas as pd
import pyarrow as pa
import pyspark.sql.connect.proto as pb2
import pyspark.sql.connect.proto.cloud_pb2 as cloud_pb2
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
from pyspark.sql.connect.client.reattach import \
    ExecutePlanResponseReattachableIterator
from pyspark.sql.connect.dataframe import DataFrame
from pyspark.sql.dataframe import DataFrame as SparkDataFrame
from pyspark.sql.types import Row

from llmfoundry.utils.exceptions import (
    ClusterDoesNotExistError,
    FailedToConnectToDatabricksError,
    FailedToCreateSQLConnectionError,
)

MINIMUM_DB_CONNECT_DBR_VERSION = '14.1'
MINIMUM_SQ_CONNECT_DBR_VERSION = '12.2'

TABLENAME_PATTERN = re.compile(r'(\S+)\.(\S+)\.(\S+)')

log = logging.getLogger(__name__)

Result = namedtuple(
    'Result',
    [
        'url',
        'row_count',
        'compressed_size',
        'uncompressed_size',
    ],
)  # pyright: ignore

# ``collect_as_cf`` is an addon new feature monkey patch on top of the DB Connect package.
# It allows the client to fetch the results in different formats from the server.
# To be able to use the code make sure this module is not overriden by DB Connect classes.


def to_cf(self: SparkConnectClient,
          plan: pb2.Plan,
          type: str = 'json') -> Tuple[List[Result], int, bool]:
    """Executes the query plans and return as presigned URLS for cloud fetch.

    It can handle the current output formats that are supported by the server.
    In contrast to the regular API methods of the client, this method does not
    return the schema and drops all other responses.

    Args:
       plan (pb2.Plan): The plan object to be executed by spark.
       type (str): The output format of the result, supported formats are 'json', 'csv', and 'arrow'.

    Returns:
       Tuple[List[Result], int, bool]: A tuple containing:
           - A list of Result namedtuples, each containing a URL, row count, compressed size,
             and uncompressed size of the part of the result.
           - Total row count of all parts of the result.
           - A boolean indicating whether the result has been truncated.
    """
    req = self._execute_plan_request_with_metadata()
    req.plan.CopyFrom(plan)

    # Add the request options
    if type == 'json':
        format = cloud_pb2.ResultOptions.CloudOptions.FORMAT_JSON
    elif type == 'csv':
        format = cloud_pb2.ResultOptions.CloudOptions.FORMAT_CSV
    elif type == 'arrow':
        format = cloud_pb2.ResultOptions.CloudOptions.FORMAT_ARROW
    else:
        raise ValueError(
            f'Only formats json, csv, and arrow are supported. Got invalid type {type}',
        )

    ro = cloud_pb2.ResultOptions(
        type=cloud_pb2.ResultOptions.TYPE_CLOUD,
        cloudOptions=cloud_pb2.ResultOptions.CloudOptions(
            format=format,
            useCompression=False,
        ),
    )
    cloud_option = any_pb2.Any()
    cloud_option.Pack(ro)
    req.request_options.append(
        pb2.ExecutePlanRequest.RequestOption(extension=cloud_option),
    )

    # Create the iterator
    iterator = ExecutePlanResponseReattachableIterator(
        req,
        self._stub,
        self._retry_policy,
        self._builder.metadata(),
    )
    # Iterate over the response
    result = []
    row_count = 0
    is_overflow = False

    for response in iterator:
        if response.HasField('extension') and response.extension.Is(
            cloud_pb2.CloudResultBatch.DESCRIPTOR,
        ):
            batch = cloud_pb2.CloudResultBatch()
            if not response.extension.Is(cloud_pb2.CloudResultBatch.DESCRIPTOR):
                raise ValueError(
                    'Response extension is not of type CloudResultBatch.',
                )
            response.extension.Unpack(batch)
            result += [
                Result(
                    b.url,
                    b.row_count,
                    b.compressed_size,
                    b.uncompressed_size,
                ) for b in batch.results
            ]
            row_count += sum(result.row_count for result in batch.results)
            is_overflow |= batch.truncated
    return result, row_count, is_overflow


SparkConnectClient.to_cf = to_cf  # pyright: ignore


def collect_as_cf(self: DataFrame,
                  type: str = 'json') -> Tuple[List[Result], int, bool]:
    """Collects DataFrame execution plan as presigned URLs.

    This method is a wrapper around the `to_cf` method of SparkConnectClient. It takes the
    execution plan of the current DataFrame, converts it to a protocol buffer format, and then
    uses the `to_cf` method to execute the plan and fetch results as presigned URLs.

    Args:
        type (str): The output format of the result, supported formats are 'json', 'csv', and 'arrow'.

    Returns:
        Tuple[List[Result], int, bool]: A tuple containing:
            - A list of Result namedtuples, each containing a URL, row count, compressed size,
              and uncompressed size of the part of the result.
            - Total row count of all parts of the result.
            - A boolean indicating whether the result is truncated or overflowed.
    """
    query = self._plan.to_proto(self._session.client)  # pyright: ignore
    return self._session.client.to_cf(query, type)  # pyright: ignore


DataFrame.collect_cf = collect_as_cf  # pyright: ignore


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


def format_tablename(table_name: str) -> str:
    """Escape catalog, schema and table names with backticks.

    This needs to be done when running SQL queries/setting spark sessions to prevent invalid identifier errors.

    Args:
        table_name (str): catalog.scheme.tablename on UC
    """
    match = re.match(TABLENAME_PATTERN, table_name)

    if match is None:
        return table_name

    formatted_identifiers = []
    for i in range(1, 4):
        identifier = f'`{match.group(i)}`'
        formatted_identifiers.append(identifier)

    return '.'.join(formatted_identifiers)


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
        signed, _, _ = df.collect_cf('arrow')  # pyright: ignore
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
    cluster_id: Optional[str],
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
        if not cluster_id:
            raise ValueError(
                'cluster_id is not set, however use_serverless is False',
            )
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
                if not cluster_id:
                    raise ValueError('cluster_id is needed for dbconnect.',)
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


def fetch_DT(
    delta_table_name: str,
    json_output_folder: str,
    http_path: Optional[str],
    cluster_id: Optional[str],
    use_serverless: bool,
    DATABRICKS_HOST: str,
    DATABRICKS_TOKEN: str,
    batch_size: int = 1 << 30,
    processes: int = os.cpu_count(), # type: ignore
    json_output_filename: str = 'train-00000-of-00001.jsonl',
) -> None:
    """Fetch UC Delta Table to local as jsonl."""
    log.info(f'Start .... Convert delta to json')

    obj = urllib.parse.urlparse(json_output_folder)
    if obj.scheme != '':
        raise ValueError(
            'Check the json_output_folder and verify it is a local path!',
        )

    if os.path.exists(json_output_folder):
        if not os.path.isdir(json_output_folder) or os.listdir(
            json_output_folder,
        ):
            raise RuntimeError(
                f'Output folder {json_output_folder} already exists and is not empty. Please remove it and retry.',
            )

    os.makedirs(json_output_folder, exist_ok=True)

    if not json_output_filename.endswith('.jsonl'):
        raise ValueError('json_output_filename needs to be a jsonl file')

    log.info(f'Directory {json_output_folder} created.')

    # validate_and_get_cluster_info allows cluster_id to be None if use_serverless is True
    method, dbsql, sparkSession = validate_and_get_cluster_info(
        cluster_id=cluster_id,
        databricks_host=DATABRICKS_HOST,
        databricks_token=DATABRICKS_TOKEN,
        http_path=http_path,
        use_serverless=use_serverless,
    )

    formatted_delta_table_name = format_tablename(delta_table_name)

    fetch(
        method,
        formatted_delta_table_name,
        json_output_folder,
        batch_size,
        processes,
        sparkSession,
        dbsql,
    )

    if dbsql is not None:
        dbsql.close()

    # combine downloaded jsonl into one big jsonl for IFT
    iterative_combine_jsons(
        json_output_folder,
        os.path.join(json_output_folder, json_output_filename),
    )


def convert_delta_to_json_from_args(
    delta_table_name: str,
    json_output_folder: str,
    http_path: Optional[str],
    cluster_id: Optional[str],
    use_serverless: bool,
    batch_size: int,
    processes: int,
    json_output_filename: str,
) -> None:
    """A wrapper for `convert_dataset_json` that parses arguments.

    Args:
        delta_table_name (str): UC table <catalog>.<schema>.<table name>
        json_output_folder (str): Local path to save the converted json
        http_path (Optional[str]): If set, dbsql method is used
        batch_size (int): Row chunks to transmit a time to avoid OOM
        processes (int): Number of processes allowed to use
        cluster_id (Optional[str]): Cluster ID with runtime newer than 14.1.0 and access mode of either assigned or shared can use databricks-connect.
        use_serverless (bool): Use serverless or not. Make sure the workspace is entitled with serverless
        json_output_filename (str): The name of the combined final jsonl that combines all partitioned jsonl
    """
    w = WorkspaceClient()
    DATABRICKS_HOST = w.config.host
    DATABRICKS_TOKEN = w.config.token

    tik = time.time()
    fetch_DT(
        delta_table_name=delta_table_name,
        json_output_folder=json_output_folder,
        http_path=http_path,
        batch_size=batch_size,
        processes=processes,
        cluster_id=cluster_id,
        use_serverless=use_serverless,
        json_output_filename=json_output_filename,
        DATABRICKS_HOST=DATABRICKS_HOST,
        DATABRICKS_TOKEN=DATABRICKS_TOKEN,
    )
    log.info(f'Elapsed time {time.time() - tik}')
