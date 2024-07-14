# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

# copyright 2022 mosaicml llm foundry authors
# spdx-license-identifier: apache-2.0

import unittest
from argparse import Namespace
from typing import Any
from unittest.mock import MagicMock, mock_open, patch

from llmfoundry.data_prep.convert_delta_to_json import (
    download,
    fetch_DT,
    format_tablename,
    iterative_combine_jsons,
    run_query,
)


class TestConvertDeltaToJsonl(unittest.TestCase):

    @patch('llmfoundry.data_prep.convert_delta_to_json.sql.connect')
    @patch('llmfoundry.data_prep.convert_delta_to_json.os.makedirs')
    @patch('llmfoundry.data_prep.convert_delta_to_json.iterative_combine_jsons')
    @patch('llmfoundry.data_prep.convert_delta_to_json.fetch')
    @patch('llmfoundry.data_prep.convert_delta_to_json.WorkspaceClient')
    def test_stream_delta_to_json(
        self,
        mock_workspace_client: Any,
        mock_fetch: Any,
        mock_combine_jsons: Any,
        mock_makedirs: Any,
        mock_sql_connect: Any,
    ):
        delta_table_name = 'test_table'
        json_output_folder = '/path/to/jsonl'
        DATABRICKS_HOST = 'test_host'
        DATABRICKS_TOKEN = 'test_token'
        http_path = 'test_path'
        batch_size = 1000
        cluster_id = '1234'
        use_serverless = False
        json_output_filename = 'combined.jsonl'

        mock_cluster_get = MagicMock()
        mock_cluster_get.return_value = MagicMock(
            spark_version='14.1.0-scala2.12',
        )
        mock_workspace_client.return_value.clusters.get = mock_cluster_get

        fetch_DT(
            delta_table_name=delta_table_name,
            json_output_folder=json_output_folder,
            http_path=http_path,
            cluster_id=cluster_id,
            DATABRICKS_HOST=DATABRICKS_HOST,
            DATABRICKS_TOKEN=DATABRICKS_TOKEN,
            use_serverless=use_serverless,
            batch_size=batch_size,
            json_output_filename=json_output_filename,
            processes=1,
        )
        mock_sql_connect.assert_called_once_with(
            server_hostname='test_host',
            http_path='test_path',
            access_token='test_token',
        )
        mock_makedirs.assert_called_once_with('/path/to/jsonl', exist_ok=True)
        mock_fetch.assert_called_once()
        mock_combine_jsons.assert_called_once_with(
            '/path/to/jsonl',
            '/path/to/jsonl/combined.jsonl',
        )

    @patch('llmfoundry.data_prep.convert_delta_to_json.os.listdir')
    @patch(
        'builtins.open',
        new_callable=mock_open,
        read_data='{"key": "value"}',
    )
    def test_iterative_combine_jsons(self, mock_file: Any, mock_listdir: Any):
        mock_listdir.return_value = ['file1.jsonl', 'file2.jsonl']
        json_directory = '/fake/dir'
        output_file = '/fake/output.jsonl'

        iterative_combine_jsons(json_directory, output_file)

        mock_listdir.assert_called_once_with(json_directory)
        mock_file.assert_called()
        """
        Diagnostic print
        for call_args in mock_file().write.call_args_list:
            print(call_args)
        --------------------
        call('{')
        call('"key"')
        call(': ')
        call('"value"')
        call('}')
        call('\n')
        call('{')
        call('"key"')
        call(': ')
        call('"value"')
        call('}')
        call('\n')
        --------------------
        """
        self.assertEqual(mock_file().write.call_count, 2)

    @patch('llmfoundry.data_prep.convert_delta_to_json.SparkSession')
    def test_run_query_dbconnect(self, mock_spark: Any):
        method = 'dbconnect'
        mock_cursor = None
        mock_spark.sql.return_value.collect.return_value = 'result'

        result = run_query(
            'SELECT * FROM table',
            method,
            cursor=mock_cursor,
            spark=mock_spark,
        )

        mock_spark.sql.assert_called_once_with('SELECT * FROM table')
        self.assertEqual(result, 'result')

    @patch('llmfoundry.data_prep.convert_delta_to_json.Cursor')
    def test_run_query_dbsql(self, mock_cursor: Any):
        method = 'dbsql'
        mock_cursor.fetchall.return_value = 'result'
        mock_spark = None

        result = run_query(
            'SELECT * FROM table',
            method,
            cursor=mock_cursor,
            spark=mock_spark,
        )

        mock_cursor.execute.assert_called_once_with('SELECT * FROM table')
        self.assertEqual(result, 'result')

    @patch('llmfoundry.data_prep.convert_delta_to_json.requests.get')
    @patch('llmfoundry.data_prep.convert_delta_to_json.pd.DataFrame.to_json')
    @patch(
        'llmfoundry.data_prep.convert_delta_to_json.os.path.join',
        return_value='/fake/path/part_1.jsonl',
    )
    @patch(
        'llmfoundry.data_prep.convert_delta_to_json.time.sleep',
    )  # Mock sleep to speed up the test
    def test_download_success(
        self,
        mock_sleep: Any,
        mock_join: Any,
        mock_to_json: Any,
        mock_get: Any,
    ):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [['val1.1', 'val1.2'],
                                           ['val2.1', 'val2.2']]
        mock_get.return_value = mock_response

        download(
            1,
            'http://fakeurl.com/data',
            '/fake/path',
            ['A', 'B'],
            resp_format='json',
        )

        mock_get.assert_called_with('http://fakeurl.com/data')
        mock_join.assert_called_with('/fake/path', 'part_1.jsonl')
        mock_to_json.assert_called_with(
            '/fake/path/part_1.jsonl',
            orient='records',
            lines=True,
        )

        mock_get.assert_called_once_with('http://fakeurl.com/data')

    @patch('llmfoundry.data_prep.convert_delta_to_json.sql.connect')
    @patch('llmfoundry.data_prep.convert_delta_to_json.DatabricksSession')
    @patch('llmfoundry.data_prep.convert_delta_to_json.WorkspaceClient')
    @patch('llmfoundry.data_prep.convert_delta_to_json.os.makedirs')
    @patch('llmfoundry.data_prep.convert_delta_to_json.iterative_combine_jsons')
    @patch('llmfoundry.data_prep.convert_delta_to_json.fetch')
    def test_dbconnect_called(
        self,
        mock_fetch: Any,
        mock_combine_jsons: Any,
        mock_makedirs: Any,
        mock_workspace_client: Any,
        mock_databricks_session: Any,
        mock_sql_connect: Any,
    ):
        delta_table_name = 'test_table'
        json_output_folder = '/path/to/jsonl'
        # Execute function with http_path=None (should use dbconnect)
        http_path = None
        cluster_id = '1234'
        DATABRICKS_HOST = 'host'
        DATABRICKS_TOKEN = 'token'
        use_serverless = False

        mock_cluster_response = Namespace(spark_version='14.1.0-scala2.12')
        mock_workspace_client.return_value.clusters.get.return_value = mock_cluster_response

        mock_remote = MagicMock()
        mock_remote.getOrCreate.return_value = MagicMock(
        )  # Mock return value for getOrCreate
        mock_databricks_session.builder.remote.return_value = mock_remote

        fetch_DT(
            delta_table_name=delta_table_name,
            json_output_folder=json_output_folder,
            http_path=http_path,
            cluster_id=cluster_id,
            DATABRICKS_HOST=DATABRICKS_HOST,
            DATABRICKS_TOKEN=DATABRICKS_TOKEN,
            use_serverless=use_serverless,
            batch_size=1,
            processes=1,
            json_output_filename='tmp.jsonl',
        )
        mock_databricks_session.builder.remote.assert_called_once_with(
            host=DATABRICKS_HOST,
            token=DATABRICKS_TOKEN,
            cluster_id=cluster_id,
        )

    @patch('llmfoundry.data_prep.convert_delta_to_json.sql.connect')
    @patch('llmfoundry.data_prep.convert_delta_to_json.DatabricksSession')
    @patch('llmfoundry.data_prep.convert_delta_to_json.WorkspaceClient')
    @patch('llmfoundry.data_prep.convert_delta_to_json.os.makedirs')
    @patch('llmfoundry.data_prep.convert_delta_to_json.iterative_combine_jsons')
    @patch('llmfoundry.data_prep.convert_delta_to_json.fetch')
    def test_sqlconnect_called_dbr13(
        self,
        mock_fetch: Any,
        mock_combine_jsons: Any,
        mock_makedirs: Any,
        mock_workspace_client: Any,
        mock_databricks_session: Any,
        mock_sql_connect: Any,
    ):
        delta_table_name = 'test_table'
        json_output_folder = '/path/to/jsonl'
        # Execute function with http_path=None (should use dbconnect)
        http_path = 'test_path'
        cluster_id = '1234'
        DATABRICKS_HOST = 'host'
        DATABRICKS_TOKEN = 'token'
        use_serverless = False

        mock_cluster_response = Namespace(spark_version='13.0.0-scala2.12')
        mock_workspace_client.return_value.clusters.get.return_value = mock_cluster_response

        fetch_DT(
            delta_table_name=delta_table_name,
            json_output_folder=json_output_folder,
            http_path=http_path,
            cluster_id=cluster_id,
            DATABRICKS_HOST=DATABRICKS_HOST,
            DATABRICKS_TOKEN=DATABRICKS_TOKEN,
            use_serverless=use_serverless,
            batch_size=1,
            processes=1,
            json_output_filename='tmp.jsonl',
        )
        mock_sql_connect.assert_called_once_with(
            server_hostname=DATABRICKS_HOST,
            http_path=http_path,
            access_token=DATABRICKS_TOKEN,
        )

    @patch('llmfoundry.data_prep.convert_delta_to_json.sql.connect')
    @patch('llmfoundry.data_prep.convert_delta_to_json.DatabricksSession')
    @patch('llmfoundry.data_prep.convert_delta_to_json.WorkspaceClient')
    @patch('llmfoundry.data_prep.convert_delta_to_json.os.makedirs')
    @patch('llmfoundry.data_prep.convert_delta_to_json.iterative_combine_jsons')
    @patch('llmfoundry.data_prep.convert_delta_to_json.fetch')
    def test_sqlconnect_called_dbr14(
        self,
        mock_fetch: Any,
        mock_combine_jsons: Any,
        mock_makedirs: Any,
        mock_workspace_client: Any,
        mock_databricks_session: Any,
        mock_sql_connect: Any,
    ):
        delta_table_name = 'test_table'
        json_output_folder = '/path/to/jsonl'
        # Execute function with http_path=None (should use dbconnect)
        http_path = 'test_path'
        cluster_id = '1234'
        DATABRICKS_HOST = 'host'
        DATABRICKS_TOKEN = 'token'
        use_serverless = False

        mock_cluster_response = Namespace(spark_version='14.2.0-scala2.12')
        mock_workspace_client.return_value.clusters.get.return_value = mock_cluster_response

        fetch_DT(
            delta_table_name=delta_table_name,
            json_output_folder=json_output_folder,
            http_path=http_path,
            cluster_id=cluster_id,
            DATABRICKS_HOST=DATABRICKS_HOST,
            DATABRICKS_TOKEN=DATABRICKS_TOKEN,
            use_serverless=use_serverless,
            batch_size=1,
            processes=1,
            json_output_filename='tmp.jsonl',
        )
        mock_sql_connect.assert_called_once_with(
            server_hostname=DATABRICKS_HOST,
            http_path=http_path,
            access_token=DATABRICKS_TOKEN,
        )

    @patch('llmfoundry.data_prep.convert_delta_to_json.sql.connect')
    @patch('llmfoundry.data_prep.convert_delta_to_json.DatabricksSession')
    @patch('llmfoundry.data_prep.convert_delta_to_json.WorkspaceClient')
    @patch('llmfoundry.data_prep.convert_delta_to_json.os.makedirs')
    @patch('llmfoundry.data_prep.convert_delta_to_json.iterative_combine_jsons')
    @patch('llmfoundry.data_prep.convert_delta_to_json.fetch')
    def test_sqlconnect_called_https(
        self,
        mock_fetch: Any,
        mock_combine_jsons: Any,
        mock_makedirs: Any,
        mock_workspace_client: Any,
        mock_databricks_session: Any,
        mock_sql_connect: Any,
    ):
        delta_table_name = 'test_table'
        json_output_folder = '/path/to/jsonl'
        # Execute function with http_path=None (should use dbconnect)
        http_path = 'test_path'
        cluster_id = '1234'
        DATABRICKS_HOST = 'https://test-host'
        DATABRICKS_TOKEN = 'token'
        use_serverless = False

        mock_cluster_response = Namespace(spark_version='14.2.0-scala2.12')
        mock_workspace_client.return_value.clusters.get.return_value = mock_cluster_response

        fetch_DT(
            delta_table_name=delta_table_name,
            json_output_folder=json_output_folder,
            http_path=http_path,
            cluster_id=cluster_id,
            DATABRICKS_HOST=DATABRICKS_HOST,
            DATABRICKS_TOKEN=DATABRICKS_TOKEN,
            use_serverless=use_serverless,
            batch_size=1,
            processes=1,
            json_output_filename='tmp.jsonl',
        )
        mock_sql_connect.assert_called_once_with(
            server_hostname='test-host',
            http_path=http_path,
            access_token=DATABRICKS_TOKEN,
        )

    @patch('llmfoundry.data_prep.convert_delta_to_json.sql.connect')
    @patch('llmfoundry.data_prep.convert_delta_to_json.DatabricksSession')
    @patch('llmfoundry.data_prep.convert_delta_to_json.WorkspaceClient')
    @patch('llmfoundry.data_prep.convert_delta_to_json.os.makedirs')
    @patch('llmfoundry.data_prep.convert_delta_to_json.iterative_combine_jsons')
    @patch('llmfoundry.data_prep.convert_delta_to_json.fetch')
    def test_serverless(
        self,
        mock_fetch: Any,
        mock_combine_jsons: Any,
        mock_makedirs: Any,
        mock_workspace_client: Any,
        mock_databricks_session: Any,
        mock_sql_connect: Any,
    ):
        delta_table_name = 'test_table'
        json_output_folder = '/path/to/jsonl'
        # Execute function with http_path=None (should use dbconnect)
        http_path = 'test_path'
        cluster_id = '1234'
        DATABRICKS_HOST = 'https://test-host'
        DATABRICKS_TOKEN = 'token'
        use_serverless = True

        mock_cluster_response = Namespace(spark_version='14.2.0-scala2.12')
        mock_workspace_client.return_value.clusters.get.return_value = mock_cluster_response

        fetch_DT(
            delta_table_name=delta_table_name,
            json_output_folder=json_output_folder,
            http_path=http_path,
            cluster_id=cluster_id,
            DATABRICKS_HOST=DATABRICKS_HOST,
            DATABRICKS_TOKEN=DATABRICKS_TOKEN,
            use_serverless=use_serverless,
            batch_size=1,
            processes=1,
            json_output_filename='tmp.jsonl',
        )
        assert not mock_sql_connect.called
        assert not mock_databricks_session.builder.remote.called

    def test_format_tablename(self):
        self.assertEqual(
            format_tablename('test_catalog.hyphenated-schema.test_table'),
            '`test_catalog`.`hyphenated-schema`.`test_table`',
        )
        self.assertEqual(
            format_tablename('catalog.schema.table'),
            '`catalog`.`schema`.`table`',
        )
        self.assertEqual(
            format_tablename('hyphenated-catalog.schema.test_table'),
            '`hyphenated-catalog`.`schema`.`test_table`',
        )
