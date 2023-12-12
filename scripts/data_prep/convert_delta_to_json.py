# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import os
import json
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

def iterative_combine_jsons(json_directory, output_file):
    """Combine json files in json_directory into one big jsonl file
    Args:
        json_directory(str): directory containing the JSON files
        output_file(str): output JSONL file
    """
    json_files = [f for f in os.listdir(json_directory) if f.endswith('.json')]

    def read_json(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for json_file in json_files:
            full_path = os.path.join(json_directory, json_file)
            json_obj = read_json(full_path)
            json.dump(json_obj, outfile, ensure_ascii=False)
            outfile.write('\n')  # Write a newline character after each JSON object

    print('JSON files have been combined into a JSONL file.')


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

def download_json(i, url, json_output_path, max_retry=3):
    for attempt in range(max_retry):
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
            raise RuntimeError(f"Failed to download after {max_retry} attempts")

def download_json_starargs(args: Tuple):
    return download_json(*args)

def download_arrow(i, url, json_output_path):
    resp = requests.get(url)
    if resp.status_code == 200:
        # The data is lz4 compressed arrow format.
        # Decompress the data
        decompressed_data = lz4.frame.decompress(resp.content)

        # Convert the decompressed data into a PyArrow table
        reader = pa.ipc.open_stream(decompressed_data)
        table = reader.read_all()

        # Convert the PyArrow table into a pandas DataFrame
        df = table.to_pandas()
        df.to_json(os.path.join(json_output_path, 'part_'+str(i)+'.json'))

def download_arrow_starargs(args: Tuple):
    return download_arrow(*args)

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
    """Fetch UC delta table with databricks-connnect and convert them to a number of json files.
       In the case when table is very large, we fetch batch_size rows a time.
       Args:
           method (str): dbconnect or dbsql
           tablename (str): catalog.scheme.tablename on UC
           batch_size (int): the number of rows that each time to fetch
           processes (int): max number of processes to use to parallelize the fetch
           sparkSession (pyspark.sql.sparksession): spark session
           dbsql (databricks.sql.connect): dbsql session
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
        signed, rows, overflow = df.collect_cf("arrow")
        print(f"len(signed) = {len(signed)}")

        args = get_args(signed, json_output_path)

        # Stopping the SparkSession to avoid spilling connection state into the subprocesses.
        sparkSession.stop()

        with ProcessPoolExecutor(max_workers=partitions) as executor:
            list(executor.map(download_arrow_starargs, args))
        #with Pool(partitions) as p:
        #    p.map(download_json_starargs, args)

    elif method == 'dbsql':
        ans = run_query(query, method, cursor, sparkSession, collect=True)
        pdf = pd.DataFrame.from_dict([row.asDict() for row in ans])
        for start in range(0, total_rows, batch_size):
            end = min(start + batch_size, total_rows)
            fetch_data(method, cursor, sparkSession, start, end, order_by, tablename, columns_str, json_output_path)

    if cursor is not None:
        cursor.close()



def fetch_DT(*args: Any, **kwargs: Any):
    """Fetch UC Delta Table to local as json files and combined into one big jsonl
       By default, databricks-connect is used. Only when ``http_path`` is present in the argument, use dbsql.
    """
    args = args[0]
    log.info(f'Start .... Convert delta to json')

    obj = urllib.parse.urlparse(args.json_output_path)
    if obj.scheme != '':
        raise ValueError(f"Check the json_output_path and verify it is a local path!")

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
            sparkSession = DatabricksSession.builder.host(args.DATABRICKS_HOST).token(args.DATABRICKS_TOKEN).header("x-databricks-session-id", session_id).getOrCreate()
        else:
            # IMPORTANT: make sure cluster has runtime newer than 14.1.0, the databricks-connect client version.
            compute_id = args.cluster_id # "1115-130834-ms4m0yv"
            sparkSession = DatabricksSession.builder.remote(host =args.DATABRICKS_HOST, token =args.DATABRICKS_TOKEN, cluster_id = compute_id).getOrCreate()

    fetch(method, args.delta_table_name, args.json_output_path, args.batch_size, args.partitions, sparkSession, dbsql)

    if dbsql is not None:
        dbsql.close()

    iterative_combine_jsons(args.json_output_path, os.path.join(args.json_output_path, 'combined.jsonl'))

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
    parser.add_argument('--cluster_id',
                        required=False,
                        type=str,
                        default=None,
                        help=
                        'Use serverless if not present. IMPORTANT! make sure cluster has runtime newer than 14.1.0, the databricks-connect client version')
    parser.add_argument('--debug', type=bool, required=False, default=False)
    args = parser.parse_args()

    tik = time.time()
    fetch_DT(args)
    print("Elapsed time", time.time() - tik)

