# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import os
import time

import pandas as pd
from databricks import sql

log = logging.getLogger(__name__)

def fetch_DT(*args: Any, **kwargs: Any):
    r"""Fetch Delta Table from UC and save to local

    This can be called as

    ```
        fetch_DT(server_hostname: str,
                 access_token: str,
                 tablename: str,
                 json_output_path: str,
                 batch_size: int = 1 << 20)
        or

        fetch_DT(server_hostname: str,
                 access_token: str,
                 http_path: str,
                 tablename: str,
                 json_output_path: str,
                 batch_size: int = 1 << 20)
    ```
    Based on the arguments, the call is redirected to either fetch_DT_with_dbconnect or fetch_DT_with_dbsql
    """
    if 'http_path' not in args and 'http_path' not in kwargs:
        return fetch_DT_with_dbconnect(*args, **kwargs)
    else:
        return fetch_DT_with_dbsql(*args, **kargs)

def fetch_DT_with_dbconnect(server_hostname: str,
                            access_token: str,
                            tablename: str,
                            json_output_path: str,
                            batch_size: int = 1 << 20):
    """Fetch UC delta table with databricks-connnect and convert them to json.
    In the case when table is very large, we fetch batch_size rows a time.
    Compared to fetch_DT_with_dbsql, this function does not need http_path.
    """
    from databricks.connect import DatabricksSession
    from uuid import uuid4

    session_id = str(uuid4())
    spark = DatabricksSession.builder.host("https://e2-dogfood.staging.cloud.databricks.com/").token("TOKEN").header("x-databricks-session-id", session_id).getOrCreate()

    try:
        ans = spark.sql(f"SELECT COUNT(*) FROM {tablename}").collect()
        total_rows = [row.asDict() for row in ans][0].popitem()[1]

        ans = spark.sql(f"SHOW COLUMNS IN {tablename}").collect()
        order_by = [row.asDict() for row in ans][0].popitem()[1]

        log.info(f'total_rows = {total_rows}')
        log.info(f'order by column {order_by}')
    except e:
        raise RuntimeError(f"Error in get total rows / columns from {tablename}. Restart sparksession and try again") from e

    for start in range(0, total_rows, batch_size):
        end = min(start + batch_size, total_rows)

        query = f"""
        WITH NumberedRows AS (
            SELECT
                *,
                ROW_NUMBER() OVER (ORDER BY {order_by}) AS rn
            FROM
                {tablename}
        )
        SELECT *
        FROM NumberedRows
        WHERE rn BETWEEN {start+1} AND {end}"""

        ans = spark.sql(query).collect()
        df = spark.createDataFrame(ans).collect()
        shard = os.path.join(json_output_path, f'shard_{start+1}_{end}.json')
        shard.write.format('json').mode('overwrite').option('header', 'true').save('/tmp/new')


def fetch_DT_with_dbsql(server_hostname: str,
                        access_token: str,
                        http_path: str,
                        tablename: str,
                        json_output_path: str,
                        batch_size: int = 1 << 20):
    """Fetch UC delta table locally as dataframes and convert them to json.
    In the case when table is very large, we fetch batch_size rows a time.
    """
    log.info(f'Start .... Convert delta to json')

    if os.path.exists(json_output_path):
        if not os.path.isdir(json_output_path) or os.listdir(
                json_output_path):
            raise RuntimeError(
                f'A file or a folder {json_output_path} already exists and is not empty. Remove it and retry!'
            )

    os.makedirs(json_output_path, exist_ok=True)

    log.info(f'Directory {json_output_path} created.')

    try:
        connection = sql.connect(
            server_hostname=server_hostname,
            http_path=http_path,
            access_token=access_token,
        )
    except Exception as e:
        raise RuntimeError(
            'Failed to create sql connection to db workspace. Check {server_hostname} and {http_path} and access token!'
        ) from e

    cursor = connection.cursor()
    cursor.execute(f'USE CATALOG main;')
    cursor.execute(f'USE SCHEMA streaming;')
    cursor.execute(f'SELECT COUNT(*) FROM {tablename}')
    ans = cursor.fetchall()

    total_rows = [row.asDict() for row in ans][0].popitem()[1]
    log.info(f'total_rows = {total_rows}')

    cursor.execute(f'SHOW COLUMNS IN {tablename}')
    ans = cursor.fetchall()

    # Get the first column to order by. can be any column
    order_by = [row.asDict() for row in ans][0].popitem()[1]
    log.info(f'order by column {order_by}')

    for start in range(0, total_rows, batch_size):
        end = min(start + batch_size, total_rows)

        query = f"""
        WITH NumberedRows AS (
            SELECT
                *,
                ROW_NUMBER() OVER (ORDER BY {order_by}) AS rn
            FROM
                {tablename}
        )
        SELECT *
        FROM NumberedRows
        WHERE rn BETWEEN {start+1} AND {end}"""

        cursor.execute(query)
        ans = cursor.fetchall()

        result = [row.asDict() for row in ans]
        df = pd.DataFrame.from_dict(result)
        df.to_json(os.path.join(json_output_path,
                                f'shard_{start+1}_{end}.json'))

    cursor.close()
    connection.close()

    print(f'Convert delta to json is done. check {json_output_path}.')
    log.info(f'Convert delta to json is done. check {json_output_path}.')

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
    parser.add_argument(
        '--http_path',
        required=True,
        type=str,
        help=
        'http_path from either dedicated cluster or serverless sql warehouse')
    parser.add_argument('--debug', type=bool, required=False, default=False)
    args = parser.parse_args()

    #server_hostname = args.DATABRICKS_HOST if args.DATABRICKS_HOST is not None else os.getenv(
    #    'DATABRICKS_HOST')
    #access_token = args.DATABRICKS_TOKEN if args.DATABRICKS_TOKEN is not None else os.getenv(
    #    'DATABRICKS_TOKEN')
    #http_path = args.http_path
    #tablename = args.delta_table_name
    #json_output_path = args.json_output_path

    tik = time.time()
    print("start timer", tik)

    fetch_DT(*args)

    print("end timer", time.time() - tik)

