# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import os
import time
import random

import urllib.parse
import pandas as pd
from databricks import sql
from typing import Any, Optional, List, Tuple
from databricks.connect import DatabricksSession
from uuid import uuid4
from pyspark.sql.types import Row
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool
import subprocess

import patch # Monkey Patching for SparkConnectClient
import requests
import pyarrow as pa
import lz4.frame


log = logging.getLogger(__name__)

def run_query(q:str, method:str, cursor=None, spark=None, collect=True) -> Optional[List[Row]]:
    if not q:
        return

    if method == 'dbsql':
        if cursor is None:
            raise ValueError(f"cursor cannot be None if using method dbsql")
        cursor.execute(q)
        if collect:
            return cursor.fetchall()

    if method == 'dbconnect':
        if spark == None:
            raise ValueError(f"sparkSession is required for dbconnect")
        df = spark.sql(q)
        if collect:
            return df.collect()
        return df

    return None


def get_args(signed, json_output_path):
    for i, r in enumerate(signed):
        yield (i, r.url, json_output_path)

def download_json(i, url, json_output_path):
    for attempt in range(3):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                pd.DataFrame.from_dict(data).to_json(os.path.join(json_output_path, 'part_'+str(i)+'.json'))
                break  # Break the loop if the request is successful
            else:
                print(f"Attempt {attempt + 1} failed with status code {response.status_code}")
        except requests.RequestException as e:
            print(f"An error occurred: {e}")

        time.sleep(random.randint(1, 5))

        if attempt == retries - 1:
            raise RuntimeError("Failed to download after 3 attempts")


def download_json_starargs(args: Tuple):
    return download_json(*args)


def download_arrow(i, url, json_output_path):
    resp = requests.get(url)
    if resp.status_code == 200:
        try:
            # Debug: Check the first few bytes of the response
            print("First 100 bytes of response:", resp.content[:100])

            # Decompress the data
            decompressed_data = lz4.frame.decompress(resp.content)

        except RuntimeError as e:
            print("Decompression error:", e)
            print("Response headers:", resp.headers)
            # Optionally, write the raw response to a file for analysis
            with open("raw_response_data.bin", "wb") as file:
                file.write(resp.content)
            raise

#def download_arrow(i, url, json_output_path):
#    resp = requests.get(url)
#    if resp.status_code == 200:
#        # The data is lz4 compressed arrow format.
#        # Decompress the data
#        decompressed_data = lz4.frame.decompress(resp.content)
#
#        # Convert the decompressed data into a PyArrow table
#        reader = pa.ipc.open_stream(decompressed_data)
#        table = reader.read_all()
#
#        # Convert the PyArrow table into a pandas DataFrame
#        df = table.to_pandas()
#        df.to_json(os.path.join(json_output_path, 'part_'+str(i)+'.json'))

def download_arrow_starargs(args: Tuple):
    return download_arrow(*args)

def fetch_data_starargs(args: Tuple):
    return fetch_data(*args)

def fetch_data(method, cursor, sparkSession, s, e, order_by, tablename, columns_str, json_output_path):
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
        pdf = run_query(query, method, cursor, sparkSession, collect=False).toPandas()
    elif method == 'dbsql':
        ans = run_query(query, method, cursor, sparkSession, collect=True)
        pdf = pd.DataFrame.from_dict([row.asDict() for row in ans])

    pdf.to_json(os.path.join(json_output_path,
                            f'part_{s+1}_{e}.json'))


def fetch(method,
          tablename: str,
          json_output_path: str,
          batch_size: int = 1 << 20,
          partitions = 1,
          sparkSession = None,
          dbsql = None,
          ):
    """Fetch UC delta table with databricks-connnect and convert them to json.
    In the case when table is very large, we fetch batch_size rows a time.
    Compared to fetch_DT_with_dbsql, this function does not need http_path.
    """
    cursor = dbsql.cursor() if dbsql is not None else None

    try:
        ans = run_query(f"SELECT COUNT(*) FROM {tablename}", method, cursor, sparkSession)
        total_rows = [row.asDict() for row in ans][0].popitem()[1]
        log.info(f'total_rows = {total_rows}')
    except Exception as e:
        raise RuntimeError(f"Error in get total rows from {tablename}. Restart sparkSession and try again") from e

    try:
        ans = run_query(f"SHOW COLUMNS IN {tablename}", method, cursor, sparkSession)
        columns = [row.asDict().popitem()[1] for row in ans]
        order_by = columns[0]
        columns_str = ','.join(columns)
        log.info(f'order by column {order_by}')
    except Exception as e:
        raise RuntimeError(f"Error in get columns from {tablename}. Restart sparkSession and try again") from e

    obj = urllib.parse.urlparse(json_output_path)

    if method == 'dbconnect':
        print('partitions = ', partitions)
        df = sparkSession.table(tablename) # "main.tpcds_sf100_delta.store_sales")

        # Running the query and collecting the data as arrow.
        signed, rows, overflow = df.collect_cf("json") # "arrow")
        print(f"len(signed) = {len(signed)}")
        args = get_args(signed, json_output_path)
        # Stopping the SparkSession to avoid spilling connection state into the subprocesses.
        sparkSession.stop()
        #with ProcessPoolExecutor(max_workers=partitions) as executor:
            #list(executor.map(download_arrow_starargs, args))
            #list(executor.map(download_json_starargs, args))
        with Pool(partitions) as p:
            p.map(download_json_starargs, args)

    elif method == 'dbsql':
        ans = run_query(query, method, cursor, sparkSession, collect=True)
        pdf = pd.DataFrame.from_dict([row.asDict() for row in ans])
        for start in range(0, total_rows, batch_size):
            end = min(start + batch_size, total_rows)
            fetch_data(method, cursor, sparkSession, start, end, order_by, tablename, columns_str, json_output_path)

    if cursor is not None:
        cursor.close()



def fetch_DT(*args: Any, **kwargs: Any):
    r"""Fetch Delta Table from UC and save to local
    Based on the arguments, the call is redirected to either fetch_DT_with_dbconnect or fetch_DT_with_dbsql
    """
    args = args[0]
    log.info(f'Start .... Convert delta to json')

    obj = urllib.parse.urlparse(args.json_output_path)
    if obj.scheme != '':
        raise ValueError(f"We don't support writing to remote yet in this script!")

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
        session_id = str(uuid4())
        #sparkSession = DatabricksSession.builder.host(args.DATABRICKS_HOST).token(args.DATABRICKS_TOKEN)._cluster_id("0704-124501-tsc2fxq").getOrCreate()
        sparkSession = DatabricksSession.builder.host(args.DATABRICKS_HOST).token(args.DATABRICKS_TOKEN).header("x-databricks-session-id", session_id).getOrCreate()
        #sparkSession = DatabricksSession.builder.remote(host =args.DATABRICKS_HOST, token =args.DATABRICKS_TOKEN, cluster_id ="0704-124501-tsc2fxq").getOrCreate()

    fetch(method, args.delta_table_name, args.json_output_path, args.batch_size, args.partitions, sparkSession, dbsql)

    if dbsql is not None:
        dbsql.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=
        'Download delta table from UC and convert to json to save local')
    parser.add_argument(
        '--delta_table_name',
        required=True,
        type=str,
        help='UC table of format <catalog>.<schema>.<table name>')
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
                        help=
                        'http_path from either dedicated cluster or serverless sql warehouse')
    parser.add_argument('--batch_size',
                        required=False,
                        type=int,
                        default=1<<20,
                        help=
                        'chunk of rows to transmit a time')
    parser.add_argument('--partitions',
                        required=False,
                        type=int,
                        default=1,
                        help=
                        'number of partitions allowed to use')
    parser.add_argument('--debug', type=bool, required=False, default=False)
    args = parser.parse_args()

    tik = time.time()
    fetch_DT(args)
    print("Elapsed time", time.time() - tik)

