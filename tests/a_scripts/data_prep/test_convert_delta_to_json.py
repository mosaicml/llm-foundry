# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

# copyright 2022 mosaicml llm foundry authors
# spdx-license-identifier: apache-2.0

import unittest
from typing import Any
from unittest.mock import MagicMock, mock_open, patch

from scripts.data_prep.convert_delta_to_json import (download_json, fetch_DT,
                                                     iterative_combine_jsons,
                                                     run_query)


class TestConverDeltaToJsonl(unittest.TestCase):

    @patch('scripts.data_prep.convert_delta_to_json.sql.connect')
    @patch('scripts.data_prep.convert_delta_to_json.os.makedirs')
    @patch('scripts.data_prep.convert_delta_to_json.iterative_combine_jsons')
    @patch('scripts.data_prep.convert_delta_to_json.fetch')
    def test_stream_delta_to_json(self, mock_fetch: Any,
                                  mock_combine_jsons: Any, mock_makedirs: Any,
                                  mock_sql_connect: Any):

        args = MagicMock()
        args.delta_table_name = 'test_table'
        args.json_output_path = '/path/to/json'
        args.DATABRICKS_HOST = 'test_host'
        args.DATABRICKS_TOKEN = 'test_token'
        args.http_path = 'test_path'
        args.batch_size = 1000
        args.partitions = 1
        args.cluster_id = None
        args.debug = False

        fetch_DT(args)
        mock_sql_connect.assert_called_once_with(server_hostname='test_host',
                                                 http_path='test_path',
                                                 access_token='test_token')
        mock_makedirs.assert_called_once_with('/path/to/json', exist_ok=True)
        mock_fetch.assert_called_once()
        mock_combine_jsons.assert_called_once_with(
            '/path/to/json', '/path/to/json/combined.jsonl')

    @patch('scripts.data_prep.convert_delta_to_json.os.listdir')
    @patch('builtins.open',
           new_callable=mock_open,
           read_data='{"key": "value"}')
    def test_iterative_combine_jsons(self, mock_file: Any, mock_listdir: Any):
        mock_listdir.return_value = ['file1.json', 'file2.json']
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
        self.assertEqual(mock_file().write.call_count, 12)

    @patch('scripts.data_prep.convert_delta_to_json.SparkSession')
    def test_run_query_dbconnect(self, mock_spark: Any):
        method = 'dbconnect'
        mock_cursor = None
        mock_spark.sql.return_value.collect.return_value = 'result'

        result = run_query('SELECT * FROM table',
                           method,
                           cursor=mock_cursor,
                           spark=mock_spark)

        mock_spark.sql.assert_called_once_with('SELECT * FROM table')
        self.assertEqual(result, 'result')

    @patch('scripts.data_prep.convert_delta_to_json.Cursor')
    def test_run_query_dbsql(self, mock_cursor: Any):
        method = 'dbsql'
        mock_cursor.fetchall.return_value = 'result'
        mock_spark = None

        result = run_query('SELECT * FROM table',
                           method,
                           cursor=mock_cursor,
                           spark=mock_spark)

        mock_cursor.execute.assert_called_once_with('SELECT * FROM table')
        self.assertEqual(result, 'result')

    @patch('scripts.data_prep.convert_delta_to_json.requests.get')
    @patch('scripts.data_prep.convert_delta_to_json.pd.DataFrame.from_dict')
    @patch('scripts.data_prep.convert_delta_to_json.os.path.join',
           return_value='/fake/path/part_1.json')
    @patch('scripts.data_prep.convert_delta_to_json.time.sleep'
          )  # Mock sleep to speed up the test
    def test_download_json_success(self, mock_sleep: Any, mock_join: Any,
                                   mock_from_dict: Any, mock_get: Any):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'data': 'test'}
        mock_get.return_value = mock_response

        download_json(1, 'http://fakeurl.com/data', '/fake/path')

        mock_get.assert_called_once_with('http://fakeurl.com/data')
        mock_from_dict.assert_called_once_with({'data': 'test'})


if __name__ == '__main__':
    unittest.main()
