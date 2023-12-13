# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

# copyright 2022 mosaicml llm foundry authors
# spdx-license-identifier: apache-2.0

import unittest
from argparse import Namespace
from typing import Any
from unittest.mock import MagicMock, patch

from scripts.data_prep.convert_delta_to_json import fetch_DT


class TestStreamDeltaToJson():

    @patch('scripts.data_prep.convert_delta_to_json.sql.connect')
    @patch('scripts.data_prep.convert_delta_to_json.pd.DataFrame.to_json')
    def test_stream_delta_to_json(self, mock_to_json: Any, mock_connect: Any):

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

        args = Namespace(DATABRICKS_HOST='test_host',
                         DATABRICKS_TOKEN='test_token',
                         http_path='test_http_path',
                         tablename='test_table',
                         json_output_path='test_output_path',
                         cluster_id='test_cluster_id')

        fetch_DT(args)
        mock_connect.assert_called_once_with(server_hostname='test_host',
                                             http_path='test_http_path',
                                             access_token='test_token')
        mock_to_json.assert_called()
        mock_cursor.close.assert_called()
        mock_connection.close.assert_called()

if __name__ == '__main__':
    unittest.main()
