import os
import logging
import requests
import json
import boto3
from botocore.config import Config
import argparse
import tempfile
from urllib.parse import urlparse
from pyspark.sql import SparkSession
#from deltalake import DeltaTable
import time
import pandas as pd
from databricks import sql

log = logging.getLogger(__name__)

def delta_to_json(spark, delta_table_path, json_output_path):
    log.info('Convert table {delta_table_path} to json and saving to {json_output_path}')
    obj = urlparse(delta_table_path)
    scheme, bucket, path = obj.scheme, obj.netloc, obj.path
    if scheme == '' and bucket == '' and path == '':
        raise FileNotFoundError(
            f'Check data availability! local index {url[0]} is not accessible.' +
            f'remote index {url[1]} does not have a valid url format')

    if scheme == '': # local
        #delta_df = read_table_from_local(path)
        delta_df = read_table_from_uc(delta_table_path)
    #elif scheme == 'dbfs': # uc table format: dbfs:/Volumes/<catalog>/<schema>/<volume-name>/path/to/folder
    #    if path.startswith('/Volumes'):
    #        delta_df = read_table_from_uc(delta_table_path)
    else:
        log.warning(f"Support of scheme {scheme} has not implemented!")
        raise NotImplementedError

    delta_df.write.mode("overwrite").format("json").save(json_output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download delta table from UC and convert to json to save local")
    parser.add_argument("--delta_table_name", required=True, type=str, help="UC table of format <catalog>.<schema>.<table name>")
    parser.add_argument("--json_output_path", required=True, type=str, help="Local path to save the converted json")
    parser.add_argument("--debug", type=bool, required=False, default=False)
    args = parser.parse_args()

    # Note: delta-io has been renamed to delta-spark after 3.0.0
    #       For spark < 3.5.0, update the configuration from
    #       https://docs.delta.io/latest/quick-start.html
    spark = SparkSession.builder \
        .appName("Delta to JSON Test") \
        .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.0.0") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .getOrCreate()

    if 0: # args.debug == True:
        # Test Local
        sample_data = [("Alice", 1), ("Bob", 2)]
        columns = ["Name", "Id"]
        spark_df = spark.createDataFrame(sample_data, schema=columns)

        #with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = '/tmp/test_delta_to_json'
        print('temp_dir = ', temp_dir)
        delta_table_path = os.path.join(temp_dir, "delta_table")
        spark_df.write.format("delta").save(delta_table_path)
        json_output_path = os.path.join(temp_dir, "json_output")

        delta_to_json(spark, delta_table_path, json_output_path)


    #connection =  sql.connect(
    #        server_hostname=os.getenv("DATABRICKS_HOST"),
    #        http_path="sql/protocolv1/o/7395834863327820/1116-234530-6seh113n", # from compute.JDBC
    #        access_token=os.getenv("DATABRICKS_TOKEN")
    #    )
    connection =  sql.connect(
            server_hostname="e2-dogfood.staging.cloud.databricks.com",
            http_path="/sql/1.0/warehouses/7e083095329f3ca5",
            access_token="dapi18a0a6fa53b5fb1afbf1101c93eee31f"
        )
    cursor = connection.cursor()
    cursor.execute(f"USE CATALOG main;")
    cursor.execute(f"USE SCHEMA streaming;")
    cursor.execute(f"SELECT * FROM dummy_table")
    ans = cursor.fetchall()
    connection.commit()

    result = [ row.asDict() for row in ans ]
    df = pd.DataFrame.from_dict(result)
    df.to_json(args.json_output_path)
