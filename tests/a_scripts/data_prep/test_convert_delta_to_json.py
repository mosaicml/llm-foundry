# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import shutil
import sys
import tempfile
import unittest
from argparse import Namespace
from contextlib import contextmanager
from tempfile import NamedTemporaryFile, mkdtemp
from typing import Any
from unittest.mock import MagicMock, mock_open, patch

import grpc
from databricks.sql.exc import ServerOperationError
from pyspark.errors import AnalysisException
from pyspark.errors.exceptions.connect import SparkConnectGrpcException

from llmfoundry.command_utils.data_prep.convert_delta_to_json import (
    FaultyDataPrepCluster,
    InsufficientPermissionsError,
    _validate_written_file,
    download,
    fetch,
    fetch_DT,
    format_tablename,
    iterative_combine_jsons,
    run_query,
)
from llmfoundry.utils.exceptions import (
    DeltaTableNotFoundError,
    MalformedUCTableError,
    StoragePermissionError,
    TableDownloadError,
)


def _mock_write_jsonl(filename: str):
    """Writes a mock .jsonl file to filename."""

    def _inner(*_: Any, **__: Any):
        base, ___ = os.path.split(filename)
        os.makedirs(base, exist_ok=True)
        with open(filename, 'w') as f:
            f.write(json.dumps({'prompt': 'prompt', 'response': 'response'}))

        assert os.path.exists(filename)

    return _inner


@contextmanager
def UncreatedNamedTemporaryFile(suffix: str):
    """Makes a temp folder for a named temporary file."""
    tempdir = None  # pyright
    try:
        tempdir = mkdtemp()
        tempfile = NamedTemporaryFile(dir=tempdir, suffix=suffix)
        tempfile.__enter__()
        os.remove(tempfile.name)
        yield tempfile

    finally:
        tempfile.__exit__(None, None, None)
        if tempdir is not None:
            shutil.rmtree(tempdir)


class TestConvertDeltaToJsonl(unittest.TestCase):

    def test_run_query_dbconnect_insufficient_permissions(self):
        error_message = (
            '[INSUFFICIENT_PERMISSIONS] Insufficient privileges: User does not have USE SCHEMA '
            "on Schema 'main.oogabooga'. SQLSTATE: 42501"
        )

        class MockAnalysisException(Exception):

            def __init__(self, message: str):
                self.message = message

            def __str__(self):
                return self.message

        with patch.dict('sys.modules', {'pyspark.errors': MagicMock()}):
            sys.modules[
                'pyspark.errors'
            ].AnalysisException = MockAnalysisException  # type: ignore

            mock_spark = MagicMock()
            mock_spark.sql.side_effect = MockAnalysisException(error_message)

            with self.assertRaises(InsufficientPermissionsError) as context:
                fetch(
                    method='dbconnect',
                    tablename='main.oogabooga',
                    json_output_folder='/fake/path',
                    batch_size=1,
                    processes=1,
                    sparkSession=mock_spark,
                    dbsql=None,
                )

            self.assertEqual(
                str(context.exception),
                error_message,
            )
            mock_spark.sql.assert_called()

    @patch(
        'databricks.sql.connect',
    )
    @patch(
        'llmfoundry.command_utils.data_prep.convert_delta_to_json.os.makedirs',
    )
    @patch(
        'llmfoundry.command_utils.data_prep.convert_delta_to_json.iterative_combine_jsons',
    )
    @patch('llmfoundry.command_utils.data_prep.convert_delta_to_json.fetch')
    @patch(
        'databricks.sdk.WorkspaceClient',
    )
    def test_stream_delta_to_json(
        self,
        mock_workspace_client: Any,
        mock_fetch: Any,
        mock_combine_jsons: Any,
        mock_makedirs: Any,
        mock_sql_connect: Any,
    ):
        delta_table_name = 'test_table'
        DATABRICKS_HOST = 'test_host'
        DATABRICKS_TOKEN = 'test_token'
        http_path = 'test_path'
        batch_size = 1000
        cluster_id = '1234'
        use_serverless = False

        mock_cluster_get = MagicMock()
        mock_cluster_get.return_value = MagicMock(
            spark_version='14.1.0-scala2.12',
        )
        mock_workspace_client.return_value.clusters.get = mock_cluster_get

        with UncreatedNamedTemporaryFile(
            suffix='.jsonl',
        ) as tf:
            mock_combine_jsons.side_effect = _mock_write_jsonl(tf.name)
            json_output_folder, json_output_filename = os.path.split(tf.name)
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
            )
        mock_sql_connect.assert_called_once_with(
            server_hostname='test_host',
            http_path='test_path',
            access_token='test_token',
        )
        mock_makedirs.assert_called()
        mock_fetch.assert_called_once()
        mock_combine_jsons.assert_called_once()

    @patch(
        'llmfoundry.command_utils.data_prep.convert_delta_to_json.os.listdir',
    )
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
        # Diagnostic print
        # for call_args in mock_file().write.call_args_list:
        #     print(call_args)
        # --------------------
        # call('{')
        # call('"key"')
        # call(': ')
        # call('"value"')
        # call('}')
        # call('\n')
        # call('{')
        # call('"key"')
        # call(': ')
        # call('"value"')
        # call('}')
        # call('\n')
        # --------------------

        self.assertEqual(mock_file().write.call_count, 2)

    @patch(
        'pyspark.sql.SparkSession',
    )
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

    @patch(
        'databricks.sql.client.Cursor',
    )
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

    @patch(
        'llmfoundry.command_utils.data_prep.convert_delta_to_json.requests.get',
    )
    @patch(
        'llmfoundry.command_utils.data_prep.convert_delta_to_json.pd.DataFrame.to_json',
    )
    @patch(
        'llmfoundry.command_utils.data_prep.convert_delta_to_json.os.path.join',
        return_value='/fake/path/part_1.jsonl',
    )
    @patch(
        'llmfoundry.command_utils.data_prep.convert_delta_to_json.time.sleep',
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

    @patch(
        'databricks.sql.connect',
    )
    @patch(
        'databricks.connect.DatabricksSession',
    )
    @patch(
        'databricks.sdk.WorkspaceClient',
    )
    @patch(
        'llmfoundry.command_utils.data_prep.convert_delta_to_json.os.makedirs',
    )
    @patch(
        'llmfoundry.command_utils.data_prep.convert_delta_to_json.iterative_combine_jsons',
    )
    @patch('llmfoundry.command_utils.data_prep.convert_delta_to_json.fetch')
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
        # Execute function with http_path=None (should use dbconnect)
        http_path = None
        cluster_id = '1234'
        DATABRICKS_HOST = 'host'
        DATABRICKS_TOKEN = 'token'
        use_serverless = False

        mock_cluster_response = Namespace(
            spark_version='14.1.0-scala2.12',
            data_security_mode='SINGLE_USER',
        )
        mock_workspace_client.return_value.clusters.get.return_value = mock_cluster_response

        mock_remote = MagicMock()
        mock_remote.getOrCreate.return_value = MagicMock(
        )  # Mock return value for getOrCreate
        mock_databricks_session.builder.remote.return_value = mock_remote

        with UncreatedNamedTemporaryFile(
            suffix='.jsonl',
        ) as tf:
            mock_combine_jsons.side_effect = _mock_write_jsonl(tf.name)
            json_output_folder, json_output_filename = os.path.split(tf.name)
            fetch_DT(
                delta_table_name=delta_table_name,
                json_output_folder=json_output_folder,
                http_path=http_path,
                cluster_id=cluster_id,
                DATABRICKS_HOST=DATABRICKS_HOST,
                DATABRICKS_TOKEN=DATABRICKS_TOKEN,
                use_serverless=use_serverless,
                json_output_filename=json_output_filename,
            )
        mock_databricks_session.builder.remote.assert_called_once_with(
            host=DATABRICKS_HOST,
            token=DATABRICKS_TOKEN,
            cluster_id=cluster_id,
        )
        mock_combine_jsons.assert_called_once()

    @patch(
        'databricks.sql.connect',
    )
    @patch(
        'databricks.connect.DatabricksSession',
    )
    @patch(
        'databricks.sdk.WorkspaceClient',
    )
    @patch(
        'llmfoundry.command_utils.data_prep.convert_delta_to_json.os.makedirs',
    )
    @patch(
        'llmfoundry.command_utils.data_prep.convert_delta_to_json.iterative_combine_jsons',
    )
    @patch('llmfoundry.command_utils.data_prep.convert_delta_to_json.fetch')
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
        # Execute function with http_path=None (should use dbconnect)
        http_path = 'test_path'
        cluster_id = '1234'
        DATABRICKS_HOST = 'host'
        DATABRICKS_TOKEN = 'token'
        use_serverless = False

        mock_cluster_response = Namespace(
            spark_version='13.0.0-scala2.12',
            data_security_mode='SINGLE_USER',
        )
        mock_workspace_client.return_value.clusters.get.return_value = mock_cluster_response

        with UncreatedNamedTemporaryFile(
            suffix='.jsonl',
        ) as tf:
            mock_combine_jsons.side_effect = _mock_write_jsonl(tf.name)
            json_output_folder, json_output_filename = os.path.split(tf.name)
            fetch_DT(
                delta_table_name=delta_table_name,
                json_output_folder=json_output_folder,
                http_path=http_path,
                cluster_id=cluster_id,
                DATABRICKS_HOST=DATABRICKS_HOST,
                DATABRICKS_TOKEN=DATABRICKS_TOKEN,
                use_serverless=use_serverless,
                json_output_filename=json_output_filename,
            )

        mock_sql_connect.assert_called_once_with(
            server_hostname=DATABRICKS_HOST,
            http_path=http_path,
            access_token=DATABRICKS_TOKEN,
        )
        mock_combine_jsons.assert_called_once()

    @patch(
        'databricks.sql.connect',
    )
    @patch(
        'databricks.connect.DatabricksSession',
    )
    @patch(
        'databricks.sdk.WorkspaceClient',
    )
    @patch(
        'llmfoundry.command_utils.data_prep.convert_delta_to_json.os.makedirs',
    )
    @patch(
        'llmfoundry.command_utils.data_prep.convert_delta_to_json.iterative_combine_jsons',
    )
    @patch('llmfoundry.command_utils.data_prep.convert_delta_to_json.fetch')
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
        # Execute function with http_path=None (should use dbconnect)
        http_path = 'test_path'
        cluster_id = '1234'
        DATABRICKS_HOST = 'host'
        DATABRICKS_TOKEN = 'token'
        use_serverless = False

        mock_cluster_response = Namespace(
            spark_version='14.2.0-scala2.12',
            data_security_mode='SINGLE_USER',
        )
        mock_workspace_client.return_value.clusters.get.return_value = mock_cluster_response

        with UncreatedNamedTemporaryFile(
            suffix='.jsonl',
        ) as tf:
            mock_combine_jsons.side_effect = _mock_write_jsonl(tf.name)
            json_output_folder, json_output_filename = os.path.split(tf.name)
            fetch_DT(
                delta_table_name=delta_table_name,
                json_output_folder=json_output_folder,
                http_path=http_path,
                cluster_id=cluster_id,
                DATABRICKS_HOST=DATABRICKS_HOST,
                DATABRICKS_TOKEN=DATABRICKS_TOKEN,
                use_serverless=use_serverless,
                json_output_filename=json_output_filename,
            )

        mock_sql_connect.assert_called_once_with(
            server_hostname=DATABRICKS_HOST,
            http_path=http_path,
            access_token=DATABRICKS_TOKEN,
        )
        mock_combine_jsons.assert_called_once()

    @patch(
        'databricks.sql.connect',
    )
    @patch(
        'databricks.connect.DatabricksSession',
    )
    @patch(
        'databricks.sdk.WorkspaceClient',
    )
    @patch(
        'llmfoundry.command_utils.data_prep.convert_delta_to_json.os.makedirs',
    )
    @patch(
        'llmfoundry.command_utils.data_prep.convert_delta_to_json.iterative_combine_jsons',
    )
    @patch('llmfoundry.command_utils.data_prep.convert_delta_to_json.fetch')
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
        # Execute function with http_path=None (should use dbconnect)
        http_path = 'test_path'
        cluster_id = '1234'
        DATABRICKS_HOST = 'https://test-host'
        DATABRICKS_TOKEN = 'token'
        use_serverless = False

        mock_cluster_response = Namespace(
            spark_version='14.2.0-scala2.12',
            data_security_mode='SINGLE_USER',
        )
        mock_workspace_client.return_value.clusters.get.return_value = mock_cluster_response

        with UncreatedNamedTemporaryFile(
            suffix='.jsonl',
        ) as tf:
            mock_combine_jsons.side_effect = _mock_write_jsonl(tf.name)
            json_output_folder, json_output_filename = os.path.split(tf.name)
            fetch_DT(
                delta_table_name=delta_table_name,
                json_output_folder=json_output_folder,
                http_path=http_path,
                cluster_id=cluster_id,
                DATABRICKS_HOST=DATABRICKS_HOST,
                DATABRICKS_TOKEN=DATABRICKS_TOKEN,
                use_serverless=use_serverless,
                json_output_filename=json_output_filename,
            )
        mock_sql_connect.assert_called_once_with(
            server_hostname='test-host',
            http_path=http_path,
            access_token=DATABRICKS_TOKEN,
        )
        mock_combine_jsons.assert_called_once()

    @patch(
        'databricks.sql.connect',
    )
    @patch(
        'databricks.connect.DatabricksSession',
    )
    @patch(
        'databricks.sdk.WorkspaceClient',
    )
    @patch(
        'llmfoundry.command_utils.data_prep.convert_delta_to_json.os.makedirs',
    )
    @patch(
        'llmfoundry.command_utils.data_prep.convert_delta_to_json.iterative_combine_jsons',
    )
    @patch('llmfoundry.command_utils.data_prep.convert_delta_to_json.fetch')
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
        # Execute function with http_path=None (should use dbconnect)
        http_path = 'test_path'
        cluster_id = '1234'
        DATABRICKS_HOST = 'https://test-host'
        DATABRICKS_TOKEN = 'token'
        use_serverless = True

        mock_cluster_response = Namespace(spark_version='14.2.0-scala2.12')
        mock_workspace_client.return_value.clusters.get.return_value = mock_cluster_response

        with UncreatedNamedTemporaryFile(
            suffix='.jsonl',
        ) as tf:
            mock_combine_jsons.side_effect = _mock_write_jsonl(tf.name)
            json_output_folder, json_output_filename = os.path.split(tf.name)
            fetch_DT(
                delta_table_name=delta_table_name,
                json_output_folder=json_output_folder,
                http_path=http_path,
                cluster_id=cluster_id,
                DATABRICKS_HOST=DATABRICKS_HOST,
                DATABRICKS_TOKEN=DATABRICKS_TOKEN,
                use_serverless=use_serverless,
                json_output_filename=json_output_filename,
            )

        assert not mock_sql_connect.called
        assert not mock_databricks_session.builder.remote.called
        mock_combine_jsons.assert_called_once()

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

    @patch(
        'llmfoundry.command_utils.data_prep.convert_delta_to_json.get_total_rows',
    )
    @patch(
        'llmfoundry.command_utils.data_prep.convert_delta_to_json.validate_and_get_cluster_info',
    )
    def test_fetch_DT_catches_grpc_errors(
        self,
        mock_validate_cluster_info: MagicMock,
        mock_get_total_rows: MagicMock,
    ):
        # Arrange
        # Mock the validate_and_get_cluster_info to return test values
        mock_validate_cluster_info.return_value = ('dbconnect', None, None)

        grpc_lib_error = grpc.RpcError()
        grpc_lib_error.code = lambda: grpc.StatusCode.INTERNAL
        grpc_lib_error.details = lambda: 'Job aborted due to stage failure: Task failed due to an error.'

        error_contexts = [
            (
                SparkConnectGrpcException('Cannot start cluster etc...'),
                FaultyDataPrepCluster,
                [
                    'The data preparation cluster you provided is terminated. Please retry with a cluster that is healthy and alive.',
                ],
            ),
            (
                SparkConnectGrpcException('cluster ... is not usable'),
                FaultyDataPrepCluster,
                [
                    'The data preparation cluster you provided is not usable. Please retry with a cluster that is healthy and alive.',
                ],
            ),
            (
                SparkConnectGrpcException(
                    'do not have permission to attach to cluster etc...',
                ),
                FaultyDataPrepCluster,
                [
                    'You do not have permission to attach to the data preparation cluster you provided.',
                ],
            ),
            (
                grpc_lib_error,
                FaultyDataPrepCluster,
                [
                    'Faulty data prep cluster, please try swapping data prep cluster: ',
                    'Job aborted due to stage failure',
                ],
            ),
        ]

        for (
            err_to_throw,
            err_to_catch,
            texts_to_check_in_error,
        ) in error_contexts:
            # Configure the fetch function to raise the SparkConnectGrpcException
            mock_get_total_rows.side_effect = err_to_throw

            # Test inputs
            delta_table_name = 'test_table'
            json_output_folder = '/tmp/to/jsonl'
            http_path = None
            cluster_id = None
            use_serverless = False
            DATABRICKS_HOST = 'https://test-host'
            DATABRICKS_TOKEN = 'test-token'

            # Act & Assert
            with self.assertRaises(err_to_catch) as context:
                fetch_DT(
                    delta_table_name=delta_table_name,
                    json_output_folder=json_output_folder,
                    http_path=http_path,
                    cluster_id=cluster_id,
                    use_serverless=use_serverless,
                    DATABRICKS_HOST=DATABRICKS_HOST,
                    DATABRICKS_TOKEN=DATABRICKS_TOKEN,
                )

            # Verify that the FaultyDataPrepCluster contains the expected message
            for text in texts_to_check_in_error:
                self.assertIn(text, str(context.exception))

        # Verify that fetch was called
        mock_get_total_rows.assert_called()

    @patch(
        'llmfoundry.command_utils.data_prep.convert_delta_to_json.get_total_rows',
    )
    def test_fetch_nonexistent_table_error(
        self,
        mock_gtr: MagicMock,
    ):
        # Create a spark.AnalysisException with specific details
        analysis_exception = AnalysisException(
            message=
            "[DELTA_TABLE_NOT_FOUND] Delta table `volume_name`.`table_name` doesn't exist",
        )

        # Configure the fetch function to raise the AnalysisException
        mock_gtr.side_effect = analysis_exception

        # Test inputs
        method = 'dbsql'
        delta_table_name = 'test_table'
        json_output_folder = '/tmp/to/jsonl'

        # Act & Assert
        with self.assertRaises(DeltaTableNotFoundError) as context:
            fetch(
                method=method,
                tablename=delta_table_name,
                json_output_folder=json_output_folder,
            )

        # Verify that the DeltaTableNotFoundError contains the expected message
        self.assertIn(
            'Please double check your delta table name',
            str(context.exception),
        )

        # Verify that get_total_rows was called
        mock_gtr.assert_called_once()

    def test_fetch_DT_catches_bad_download(self):
        with NamedTemporaryFile() as tf:
            file_name = tf.name
            file_folder, file_name = os.path.split(file_name)
            with self.assertRaises(StoragePermissionError):
                _validate_written_file(
                    file_folder,
                    file_name,
                    'test_delta_table',
                )

    @patch(
        'llmfoundry.command_utils.data_prep.convert_delta_to_json.get_total_rows',
    )
    def test_fetch_malformed_table_error(
        self,
        mock_gtr: MagicMock,
    ):
        # Create a spark.AnalysisException with specific details
        server_exception = ServerOperationError(
            '[UNRESOLVED_COLUMN.WITH_SUGGESTION] yada yada',
        )

        # Configure the fetch function to raise the AnalysisException
        mock_gtr.side_effect = server_exception

        # Test inputs
        method = 'dbsql'
        delta_table_name = 'test_table'
        json_output_folder = '/tmp/to/jsonl'

        # Act & Assert
        with self.assertRaises(MalformedUCTableError):
            fetch(
                method=method,
                tablename=delta_table_name,
                json_output_folder=json_output_folder,
            )

        # Verify that get_total_rows was called
        mock_gtr.assert_called_once()

    @patch(
        'llmfoundry.command_utils.data_prep.convert_delta_to_json.fetch',
    )
    @patch(
        'llmfoundry.command_utils.data_prep.convert_delta_to_json.validate_and_get_cluster_info',
    )
    def test_non_shared_single_user_cluster_error(
        self,
        mock_validate_cluster_info: MagicMock,
        mock_fetch: MagicMock,
    ):
        mock_validate_cluster_info.return_value = ('dbconnect', None, None)

        exception_message = 'Cluster is not Shared or Single User Cluster'
        spark_exception = SparkConnectGrpcException(exception_message)

        mock_fetch.side_effect = spark_exception

        # Define test inputs
        delta_table_name = 'test_table'
        json_output_folder = '/tmp/to/jsonl'
        http_path = None
        cluster_id = 'test-cluster-id'
        use_serverless = False
        DATABRICKS_HOST = 'https://test-host'
        DATABRICKS_TOKEN = 'test-token'

        # Act & Assert
        with self.assertRaises(FaultyDataPrepCluster) as context:
            fetch_DT(
                delta_table_name=delta_table_name,
                json_output_folder=json_output_folder,
                http_path=http_path,
                cluster_id=cluster_id,
                use_serverless=use_serverless,
                DATABRICKS_HOST=DATABRICKS_HOST,
                DATABRICKS_TOKEN=DATABRICKS_TOKEN,
            )

        self.assertIn(
            f'The cluster you have provided: {cluster_id} does not have data governance enabled. Please use a cluster with a data security mode other than NONE.',
            str(context.exception),
        )

        mock_fetch.assert_called()

    @patch(
        'llmfoundry.command_utils.data_prep.convert_delta_to_json.get_args',
        return_value=[(None, None, None, None)],
    )
    @patch(
        'llmfoundry.command_utils.data_prep.convert_delta_to_json.get_total_rows',
    )
    @patch(
        'llmfoundry.command_utils.data_prep.convert_delta_to_json.get_columns_info',
        return_value=(None, None, None),
    )
    @patch(
        'llmfoundry.command_utils.data_prep.convert_delta_to_json.ProcessPoolExecutor',
    )
    @patch(
        'llmfoundry.command_utils.data_prep.convert_delta_to_json.validate_and_get_cluster_info',
    )
    def test_general_table_download_error(
        self,
        mock_validate_cluster_info: MagicMock,
        mock_pp_executor: MagicMock,
        mock_get_columns_info: MagicMock,
        mock_get_total_rows: MagicMock,
        mock_get_args: MagicMock,
    ):
        mock_session = MagicMock()
        mock_table = MagicMock()
        mock_session.table.return_value = mock_table
        mock_validate_cluster_info.return_value = (
            'dbconnect',
            None,
            mock_session,
        )
        mock_table.collect_cf.return_value = (MagicMock(), None, None)

        exception_message = 'Overflow occurred in npy_datetimestruct_to_datetime'
        overflow_exception = OverflowError(exception_message)

        mock_pp_executor.side_effect = overflow_exception

        # Define test inputs
        delta_table_name = 'test_table'
        json_output_folder = tempfile.mkdtemp()
        http_path = None
        cluster_id = 'test-cluster-id'
        use_serverless = False
        DATABRICKS_HOST = 'https://test-host'
        DATABRICKS_TOKEN = 'test-token'

        # Act & Assert
        with self.assertRaises(TableDownloadError) as context:
            fetch_DT(
                delta_table_name=delta_table_name,
                json_output_folder=json_output_folder,
                http_path=http_path,
                cluster_id=cluster_id,
                use_serverless=use_serverless,
                DATABRICKS_HOST=DATABRICKS_HOST,
                DATABRICKS_TOKEN=DATABRICKS_TOKEN,
                processes=1,
            )

        self.assertIn(
            f'Error downloading table {delta_table_name}: {exception_message}',
            str(context.exception),
        )

        mock_pp_executor.assert_called()
