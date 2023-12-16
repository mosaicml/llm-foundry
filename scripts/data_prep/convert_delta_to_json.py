# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import time
import urllib.parse
from argparse import ArgumentParser, Namespace
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor
from typing import Iterable, List, Optional, Tuple, Union
from uuid import uuid4

# PB2 stuff
import google.protobuf.any_pb2 as any_pb2
import lz4.frame
import pandas as pd
import pyarrow as pa
import pyspark.sql.connect.proto as pb2
import pyspark.sql.connect.proto.cloud_pb2 as cloud_pb2
import requests
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

MINIMUM_DBR_VERSION = '14.0.0'

log = logging.getLogger(__name__)

Result = namedtuple(
    'Result', ['url', 'row_count', 'compressed_size', 'uncompressed_size'
              ])  # pyright: ignore

cf_collect_type = 'arrow'  # optionally change to json if arrow fails

# This is a monkey patch on top of the DB Connect package that allows
# the client to fetch the results in different formats from the server. To be
# able to use the code make sure this module is not overriden by DB Connect classes.


def to_cf(self: SparkConnectClient,
          plan: pb2.Plan,
          type: str = 'json') -> Tuple[List[Result], int, bool]:
    """Executes plan object return as cloud fetch presigned URLS.

    It can handle the current outptu formats that are supported by the server.
    In contrast to the regular API methods of the client, this method does not
    return the schema and drops all other responses.
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
        raise Exception('Invalid type')

    ro = cloud_pb2.ResultOptions(
        type=cloud_pb2.ResultOptions.TYPE_CLOUD,
        cloudOptions=cloud_pb2.ResultOptions.CloudOptions(
            format=format,
            useCompression=False,
        ))
    cloud_option = any_pb2.Any()
    cloud_option.Pack(ro)
    req.request_options.append(
        pb2.ExecutePlanRequest.RequestOption(extension=cloud_option))

    # Create the iterator
    iterator = ExecutePlanResponseReattachableIterator(req, self._stub,
                                                       self._retry_policy,
                                                       self._builder.metadata())
    # Iterate over the response

    result = []
    row_count = 0
    is_overflow = False

    for response in iterator:
        if response.HasField('extension') and response.extension.Is(
                cloud_pb2.CloudResultBatch.DESCRIPTOR):
            batch = cloud_pb2.CloudResultBatch()
            assert response.extension.Is(cloud_pb2.CloudResultBatch.DESCRIPTOR)
            response.extension.Unpack(batch)
            result += [
                Result(b.url, b.row_count, b.compressed_size,
                       b.uncompressed_size) for b in batch.results
            ]
            row_count += sum(result.row_count for result in batch.results)
            is_overflow |= batch.truncated
    return result, row_count, is_overflow


SparkConnectClient.to_cf = to_cf  # pyright: ignore


def collect_as_cf(self: DataFrame,
                  type: str = 'json') -> Tuple[List[Result], int, bool]:
    query = self._plan.to_proto(self._session.client)  # pyright: ignore
    return self._session.client.to_cf(query, type)  # pyright: ignore


DataFrame.collect_cf = collect_as_cf  # pyright: ignore


def iterative_combine_jsons(json_directory: str, output_file: str) -> None:
    """Combine jsonl files in json_directory into one big jsonl file.

    Args:
        json_directory(str): directory containing the JSON files
        output_file(str): output JSONL file
    """
    json_files = [f for f in os.listdir(json_directory) if f.endswith('.jsonl')]
    with open(output_file, 'w') as outfile:
        for file_name in json_files:
            with open(os.path.join(json_directory, file_name), 'r') as infile:
                for line in infile:
                    outfile.write(line)
    print('JSON files have been combined into a JSONL file.')


def run_query(
    q: str,
    method: str,
    cursor: Optional[Cursor] = None,
    spark: Optional[SparkSession] = None,
    collect: bool = True
) -> Optional[Union[List[Row], DataFrame, SparkDataFrame]]:
    """Run SQL query via databricks-connect or databricks-sql.

    Args:
        q (str): sql query
        method (str): select from dbsql and dbconnect
        cursor (Cursor): connection.cursor
        spark (SparkSession): spark session
        collect (bool): whether to get the underlying data from spark dataframe
    """
    assert method in ['dbsql', 'dbconnect'], f'Unrecognized method: {method}'
    if method == 'dbsql':
        if cursor is None:
            raise ValueError(f'cursor cannot be None if using method dbsql')
        cursor.execute(q)
        if collect:
            return cursor.fetchall()

    elif method == 'dbconnect':
        if spark == None:
            raise ValueError(f'sparkSession is required for dbconnect')
        df = spark.sql(q)
        if collect:
            return df.collect()
        return df

    return None


def get_args(signed: List, json_output_path: str, columns: List,
             cf_collect_type: str) -> Iterable:
    for i, r in enumerate(signed):
        yield (i, r.url, json_output_path, columns, cf_collect_type)


def download(ipart: int,
             url: str,
             json_output_path: str,
             columns: Optional[List] = None,
             resp_format: str = 'arrow',
             compressed: bool = False) -> None:
    """Thread download presigned url and save to jsonl locally.

    Args:
        ipart (int): presigned url id
        url (str): presigned url
        json_output_path (str): directory to save the ipart_th segment of dataframe
        columns (list): schema to save to json
        resp_format (str): whether to use arrow or json when collect
        compressed (bool): if data is compressed before downloading. Need decompress if compressed=True.
    """
    resp = requests.get(url)
    if resp.status_code == 200:
        if resp_format == 'json':
            data = resp.json()
            pd.DataFrame(data, columns=columns).to_json(os.path.join(
                json_output_path, 'part_' + str(ipart) + '.jsonl'),
                                                        orient='records',
                                                        lines=True)
            return

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
        df.to_json(os.path.join(json_output_path,
                                'part_' + str(ipart) + '.jsonl'),
                   orient='records',
                   lines=True,
                   force_ascii=False)


def download_starargs(args: Tuple) -> None:
    return download(*args)


def fetch_data(method: str, cursor: Optional[Cursor],
               sparkSession: Optional[SparkSession], s: int, e: int,
               order_by: str, tablename: str, columns_str: str,
               json_output_path: str) -> None:
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
    WHERE rn BETWEEN {s+1} AND {e}"""

    if method == 'dbconnect':
        spark_df = run_query(query, method, cursor, sparkSession, collect=False)
        if not spark_df:
            raise RuntimeError(
                f'Expect spark dataframe with {query} but got None')
        pdf = spark_df.toPandas()  # pyright: ignore
    else:  #  method == 'dbsql':
        ans = run_query(query, method, cursor, sparkSession, collect=True)
        if not ans:
            raise RuntimeError(f'Got empty results with {query}')
        records = [r.asDict() for r in ans]  # pyright: ignore
        pdf = pd.DataFrame.from_dict(records)

    pdf.to_json(os.path.join(json_output_path, f'part_{s+1}_{e}.jsonl'))


def fetch(
    method: str,
    tablename: str,
    json_output_path: str,
    batch_size: int = 1 << 30,
    processes: int = 1,
    sparkSession: Optional[SparkSession] = None,
    dbsql: Optional[Connection] = None,
) -> None:
    """Fetch UC delta table with databricks-connnect as JSONL.

    Args:
        method (str): dbconnect or dbsql
        tablename (str): catalog.scheme.tablename on UC
        json_output_path (str): path to write the result json file to
        batch_size (int): number of rows that dbsql fetches each time to avoid OOM
        processes (int): max number of processes to use to parallelize the fetch
        sparkSession (pyspark.sql.sparksession): spark session
        dbsql (databricks.sql.connect): dbsql session
    """
    cursor = dbsql.cursor() if dbsql is not None else None

    try:
        ans = run_query(f'SELECT COUNT(*) FROM {tablename}', method, cursor,
                        sparkSession)
        nrows = [row.asDict() for row in ans][0].popitem()[1]  # pyright: ignore
        log.info(f'total_rows = {nrows}')
    except Exception as e:
        raise RuntimeError(
            f'Error in get total rows from {tablename}. Restart sparkSession and try again'
        ) from e

    try:
        ans = run_query(f'SHOW COLUMNS IN {tablename}', method, cursor,
                        sparkSession)
        columns = [row.asDict().popitem()[1] for row in ans]  # pyright: ignore
        order_by = columns[0]
        columns_str = ','.join(columns)
        log.info(f'order by column {order_by}')
    except Exception as e:
        raise RuntimeError(
            f'Error in get columns from {tablename}. Restart sparkSession and try again'
        ) from e

    if method == 'dbconnect' and sparkSession:
        print('processes = ', processes)
        df = sparkSession.table(
            tablename)  # "main.tpcds_sf100_delta.store_sales")

        # Running the query and collecting the data as arrow or json.
        signed, _, _ = df.collect_cf('arrow')  # pyright: ignore
        print(f'len(signed) = {len(signed)}')

        args = get_args(signed, json_output_path, columns, cf_collect_type)

        # Stopping the SparkSession to avoid spilling connection state into the subprocesses.
        sparkSession.stop()

        with ProcessPoolExecutor(max_workers=processes) as executor:
            list(executor.map(download_starargs, args))

    elif method == 'dbsql' and cursor:
        for start in range(0, nrows, batch_size):
            end = min(start + batch_size, nrows)
            fetch_data(method, cursor, sparkSession, start, end, order_by,
                       tablename, columns_str, json_output_path)

    if cursor is not None:
        cursor.close()


def fetch_DT(args: Namespace) -> None:
    """Fetch UC Delta Table to local as jsonl."""
    log.info(f'Start .... Convert delta to json')

    obj = urllib.parse.urlparse(args.json_output_path)
    if obj.scheme != '':
        raise ValueError(
            f'Check the json_output_path and verify it is a local path!')

    if os.path.exists(args.json_output_path):
        if not os.path.isdir(args.json_output_path) or os.listdir(
                args.json_output_path):
            raise RuntimeError(
                f'A file or a folder {args.json_output_path} already exists and is not empty. Remove it and retry!'
            )

    os.makedirs(args.json_output_path, exist_ok=True)

    log.info(f'Directory {args.json_output_path} created.')

    method = ''
    dbsql = None
    sparkSession = None

    if hasattr(args, 'http_path') and args.http_path:
        method = 'dbsql'
        try:
            dbsql = sql.connect(
                server_hostname=args.DATABRICKS_HOST,
                http_path=args.http_path,
                access_token=args.DATABRICKS_TOKEN,
            )
        except Exception as e:
            raise RuntimeError(
                'Failed to create sql connection to db workspace. Check {server_hostname} and {http_path} and access token!'
            ) from e
    else:
        method = 'dbconnect'
        if not args.cluster_id:
            session_id = str(uuid4())
            sparkSession = DatabricksSession.builder.host(
                args.DATABRICKS_HOST).token(args.DATABRICKS_TOKEN).header(
                    'x-databricks-session-id', session_id).getOrCreate()
        else:
            # IMPORTANT: make sure cluster has runtime newer than 14.1.0, the databricks-connect client version.
            compute_id = args.cluster_id  # "1115-130834-ms4m0yv" - valid 14.1.0
            w = WorkspaceClient()
            res = w.clusters.get(
                cluster_id=compute_id)  # '0704-124501-tsc2fxq' - invalid 12.2.x
            runtime_version = res.spark_version.split('-scala')[0].replace(
                'x-snapshot', '0').replace('x', '0')
            assert version.parse(runtime_version) >= version.parse(
                MINIMUM_DBR_VERSION
            ), f'You need at least {MINIMUM_DBR_VERSION} to use Databricks-connect to read delta table for FT API but got {res.spark_version}'
            sparkSession = DatabricksSession.builder.remote(
                host=args.DATABRICKS_HOST,
                token=args.DATABRICKS_TOKEN,
                cluster_id=compute_id).getOrCreate()

    fetch(method, args.delta_table_name, args.json_output_path, args.batch_size,
          args.processes, sparkSession, dbsql)

    if dbsql is not None:
        dbsql.close()

    iterative_combine_jsons(
        args.json_output_path,
        os.path.join(args.json_output_path, 'combined.jsonl'))


if __name__ == '__main__':
    parser = ArgumentParser(
        description=
        'Download delta table from UC and convert to json to save local')
    parser.add_argument('--delta_table_name',
                        required=True,
                        type=str,
                        help='UC table <catalog>.<schema>.<table name>')
    parser.add_argument('--json_output_path',
                        required=True,
                        type=str,
                        help='Local path to save the converted json')
    parser.add_argument('--DATABRICKS_HOST',
                        required=False,
                        type=str,
                        help='DATABRICKS_HOST')
    parser.add_argument('--DATABRICKS_TOKEN',
                        required=False,
                        type=str,
                        help='DATABRICKS_TOKEN')
    parser.add_argument('--http_path',
                        required=False,
                        type=str,
                        help='http_path is set then dbsql method is used')
    parser.add_argument('--batch_size',
                        required=False,
                        type=int,
                        default=1 << 30,
                        help='row chunks to transmit a time to avoid OOM')
    parser.add_argument('--processes',
                        required=False,
                        type=int,
                        default=1,
                        help='number of processes allowed to use')
    parser.add_argument(
        '--cluster_id',
        required=True,
        type=str,
        default=None,
        help=
        'Use serverless if not present. IMPORTANT! make sure cluster has runtime newer than 14.1.0, the databricks-connect client version'
    )
    parser.add_argument('--debug', type=bool, required=False, default=False)
    args = parser.parse_args()

    tik = time.time()
    fetch_DT(args)
    print('Elapsed time', time.time() - tik)
