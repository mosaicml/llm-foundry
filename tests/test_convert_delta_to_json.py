# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

# copyright 2022 mosaicml llm foundry authors
# spdx-license-identifier: apache-2.0

import os
import sys
from typing import Any

# Add repo root to path so we can import scripts and test it
repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(repo_dir)

import unittest
from unittest.mock import MagicMock, patch

from scripts.data_prep.convert_delta_to_json import stream_delta_to_json


class TestStreamDeltaToJson(unittest.TestCase):

    @patch('scripts.data_prep.convert_delta_to_json.sql.connect')
    @patch('scripts.data_prep.convert_delta_to_json.pd.DataFrame.to_json')
    def test_stream_delta_to_json(self, mock_to_json: Any, mock_connect: Any):
        mock_args = MagicMock()
        mock_args.DATABRICKS_HOST = 'test_host'
        mock_args.DATABRICKS_TOKEN = 'test_token'
        mock_args.http_path = 'test_http_path'
        mock_args.delta_table_name = 'test_table'
        mock_args.json_output_path = 'test_output_path'

        # Mock database connection and cursor
        mock_cursor = MagicMock()
        mock_connection = MagicMock()
        mock_connection.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_connection

        # Mock fetchall response
        count_response = MagicMock()
        count_response.asDict.return_value = {'COUNT(*)': 3}
        column_response_item = MagicMock()
        column_response_item.asDict.return_value = {
            'COLUMN_NAME': 'name'
        }  # Assuming SHOW COLUMNS query returns this format
        data_response_item = MagicMock()
        data_response_item.asDict.return_value = {
            'name': 'test',
            'id': 1
        }  # Assuming SELECT query returns this format
        mock_cursor.fetchall.side_effect = [[count_response],
                                            [column_response_item],
                                            [data_response_item]]

        stream_delta_to_json(mock_args)

        mock_connect.assert_called_once_with(server_hostname='test_host',
                                             http_path='test_http_path',
                                             access_token='test_token')
        mock_to_json.assert_called()
        mock_cursor.close.assert_called()
        mock_connection.close.assert_called()


if __name__ == '__main__':
    unittest.main()
