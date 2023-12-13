# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

# copyright 2022 mosaicml llm foundry authors
# spdx-license-identifier: apache-2.0

import unittest
from argparse import Namespace
from typing import Any
from unittest.mock import MagicMock, patch

from scripts.data_prep.convert_delta_to_json import fetch_DT

@patch('scripts.data_prep.convert_delta_to_json.sql.connect')
@patch('scripts.data_prep.convert_delta_to_json.os.makedirs')
@patch('scripts.data_prep.convert_delta_to_json.iterative_combine_jsons')
@patch('scripts.data_prep.convert_delta_to_json.fetch')
def test_stream_delta_to_json(self, mock_fetch: Any, mock_combine_jsons: Any, mock_makedirs: Any, mock_sql_connect: Any):

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
    mock_sql_connect.assert_called_once_with(server_hostname='test_host', http_path='test_path', access_token='test_token')
    mock_makedirs.assert_called_once_with('/path/to/json', exist_ok=True)
    mock_fetch.assert_called_once()
    mock_combine_jsons.assert_called_once_with('/path/to/json', '/path/to/json/combined.jsonl')


if __name__ == '__main__':
    unittest.main()
