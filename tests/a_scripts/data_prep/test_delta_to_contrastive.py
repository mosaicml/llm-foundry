# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

# tests/a_scripts/contrastive/test_delta_to_contrastive.py

import json
import unittest
from typing import Any
from unittest.mock import MagicMock, mock_open, patch


class TestValidateColumnsInTable(unittest.TestCase):
    """Unit tests for the validate_columns_in_table function."""

    @patch(
        'llmfoundry.command_utils.data_prep.convert_delta_to_contrastive_mds.run_query',
        autospec=True,
    )
    def test_validate_columns_success(self, mock_run_query: MagicMock) -> None:
        # Import inside the test after patching
        from llmfoundry.command_utils.data_prep.convert_delta_to_contrastive_mds import \
            validate_columns_in_table

        # Mock the run_query to return all required and optional columns
        mock_run_query.return_value = [
            MagicMock(
                asDict=MagicMock(return_value={'col_name': 'query_text'}),
            ),
            MagicMock(
                asDict=MagicMock(return_value={'col_name': 'positive_passage'}),
            ),
            MagicMock(
                asDict=MagicMock(
                    return_value={'col_name': 'negative_passages'},
                ),
            ),
        ]

        required_columns = ['query_text', 'positive_passage']
        optional_columns = ['negative_passages']
        table_name = 'test_table'
        method = 'dbconnect'

        result: bool = validate_columns_in_table(
            required_columns=required_columns,
            optional_columns=optional_columns,
            table_name=table_name,
            method=method,
            cursor=None,
            spark=None,
        )

        self.assertTrue(result)
        mock_run_query.assert_called_once_with(
            f'SHOW COLUMNS IN {table_name}',
            method,
            None,
            None,
        )

    @patch(
        'llmfoundry.command_utils.data_prep.convert_delta_to_contrastive_mds.run_query',
        autospec=True,
    )
    def test_validate_columns_missing_required(
        self,
        mock_run_query: MagicMock,
    ) -> None:
        from llmfoundry.command_utils.data_prep.convert_delta_to_contrastive_mds import \
            validate_columns_in_table

        # Mock the run_query to return missing required columns
        mock_run_query.return_value = [
            MagicMock(
                asDict=MagicMock(return_value={'col_name': 'query_text'}),
            ),
        ]

        required_columns = ['query_text', 'positive_passage']
        optional_columns = ['negative_passages']
        table_name = 'test_table'
        method = 'dbconnect'

        with self.assertLogs(
            'llmfoundry.command_utils.data_prep.convert_delta_to_contrastive_mds',
            level='ERROR',
        ) as log:
            result: bool = validate_columns_in_table(
                required_columns=required_columns,
                optional_columns=optional_columns,
                table_name=table_name,
                method=method,
                cursor=None,
                spark=None,
            )

        self.assertFalse(result)
        self.assertIn('Missing required columns', log.output[0])

    @patch(
        'llmfoundry.command_utils.data_prep.convert_delta_to_contrastive_mds.run_query',
        autospec=True,
    )
    def test_validate_columns_extra_columns(
        self,
        mock_run_query: MagicMock,
    ) -> None:
        from llmfoundry.command_utils.data_prep.convert_delta_to_contrastive_mds import \
            validate_columns_in_table

        # Mock the run_query to return extra columns
        mock_run_query.return_value = [
            MagicMock(
                asDict=MagicMock(return_value={'col_name': 'query_text'}),
            ),
            MagicMock(
                asDict=MagicMock(return_value={'col_name': 'positive_passage'}),
            ),
            MagicMock(
                asDict=MagicMock(return_value={'col_name': 'extra_column'}),
            ),
        ]

        required_columns = ['query_text', 'positive_passage']
        optional_columns = ['negative_passages']
        table_name = 'test_table'
        method = 'dbconnect'

        with self.assertLogs(
            'llmfoundry.command_utils.data_prep.convert_delta_to_contrastive_mds',
            level='WARNING',
        ) as log:
            result: bool = validate_columns_in_table(
                required_columns=required_columns,
                optional_columns=optional_columns,
                table_name=table_name,
                method=method,
                cursor=None,
                spark=None,
            )

        self.assertFalse(result)
        self.assertIn('Extra columns found', log.output[0])

    @patch(
        'llmfoundry.command_utils.data_prep.convert_delta_to_contrastive_mds.run_query',
        autospec=True,
    )
    def test_validate_columns_exception(
        self,
        mock_run_query: MagicMock,
    ) -> None:
        from llmfoundry.command_utils.data_prep.convert_delta_to_contrastive_mds import \
            validate_columns_in_table

        # Mock run_query to raise an exception
        mock_run_query.side_effect = Exception('Test Exception')

        required_columns = ['query_text', 'positive_passage']
        optional_columns = ['negative_passages']
        table_name = 'test_table'
        method = 'dbconnect'

        with self.assertLogs(
            'llmfoundry.command_utils.data_prep.convert_delta_to_contrastive_mds',
            level='ERROR',
        ) as log:
            result: bool = validate_columns_in_table(
                required_columns=required_columns,
                optional_columns=optional_columns,
                table_name=table_name,
                method=method,
                cursor=None,
                spark=None,
            )

        self.assertFalse(result)
        self.assertIn('Error validating columns in table', log.output[0])


class TestMainFunction(unittest.TestCase):
    """Unit tests for the main function."""

    @patch('databricks.sdk.WorkspaceClient', autospec=True)
    @patch(
        'llmfoundry.command_utils.data_prep.convert_delta_to_contrastive_mds.validate_columns_in_table',
        autospec=True,
    )
    @patch(
        'llmfoundry.command_utils.data_prep.convert_delta_to_contrastive_mds.validate_and_get_cluster_info',
        autospec=True,
    )
    @patch(
        'llmfoundry.command_utils.data_prep.convert_delta_to_contrastive_mds.fetch_DT',
        autospec=True,
    )
    @patch(
        'llmfoundry.command_utils.data_prep.convert_delta_to_contrastive_mds.MDSWriter',
        autospec=True,
    )
    def test_main_success(
        self,
        mock_mds_writer: MagicMock,
        mock_fetch_DT: MagicMock,
        mock_validate_cluster_info: MagicMock,
        mock_validate_columns: MagicMock,
        mock_workspace_client_class: MagicMock,
    ) -> None:
        with patch(
            'databricks.sdk.WorkspaceClient',
            mock_workspace_client_class,
        ):
            from llmfoundry.command_utils.data_prep.convert_delta_to_contrastive_mds import \
                convert_delta_to_contrastive_mds

            # Setup mocks
            mock_workspace_client_instance = MagicMock()
            mock_workspace_client_class.return_value = mock_workspace_client_instance

            mock_validate_cluster_info.return_value = (
                'dbconnect',
                MagicMock(),
                MagicMock(),
            )
            mock_validate_columns.return_value = True

            mock_mds_instance: MagicMock = MagicMock()
            mock_mds_writer.return_value.__enter__.return_value = mock_mds_instance

            args: dict[str, Any] = {
                'delta_table_name': 'test_table',
                'http_path': 'http://test_path',
                'cluster_id': 'cluster123',
                'use_serverless': False,
                'output_path': '/output/path',
                'batch_size': 1000,
                'processes': 4,
            }

            with patch('tempfile.TemporaryDirectory') as mock_temp_dir:
                mock_temp_dir.return_value.__enter__.return_value = '/tmp/mock_dir'

                # **Update the mock_open data to include negative_passages as a list**
                with patch(
                    'builtins.open',
                    mock_open(
                        read_data=
                        '{"query_text": "sample", "positive_passage": "passage", "negative_passages": []}\n',
                    ),
                ):
                    convert_delta_to_contrastive_mds(**args)

                    # Assertions
                    mock_workspace_client_class.assert_called_once()
                    mock_validate_cluster_info.assert_called_once_with(
                        cluster_id='cluster123',
                        databricks_host=mock_workspace_client_instance.config.
                        host,
                        databricks_token=mock_workspace_client_instance.config.
                        token,
                        http_path='http://test_path',
                        use_serverless=False,
                    )
                    mock_validate_columns.assert_called_once_with(
                        required_columns=['query_text', 'positive_passage'],
                        optional_columns=['negative_passages'],
                        table_name='test_table',
                        method='dbconnect',
                        cursor=mock_validate_cluster_info.return_value[1].
                        cursor(),
                        spark=mock_validate_cluster_info.return_value[2],
                    )
                    mock_fetch_DT.assert_called_once_with(
                        delta_table_name='test_table',
                        json_output_folder='/tmp/mock_dir',
                        http_path='http://test_path',
                        cluster_id='cluster123',
                        use_serverless=False,
                        json_output_filename='output.jsonl',
                        batch_size=1000,
                        processes=4,
                        DATABRICKS_HOST=mock_workspace_client_instance.config.
                        host,
                        DATABRICKS_TOKEN=mock_workspace_client_instance.config.
                        token,
                    )
                    mock_mds_writer.assert_called_once_with(
                        out='/output/path',
                        columns={
                            'query_text': 'str',
                            'positive_passage': 'str',
                            'negative_passages': 'str',
                        },
                        compression='zstd:7',
                        hashes=['sha1'],
                        size_limit='10mb',
                    )
                    mock_mds_instance.write.assert_called_once_with({
                        'query_text': 'sample',
                        'positive_passage': 'passage',
                        'negative_passages': json.dumps([]),
                    })

    @patch('databricks.sdk.WorkspaceClient', autospec=True)
    @patch(
        'llmfoundry.command_utils.data_prep.convert_delta_to_contrastive_mds.validate_columns_in_table',
        autospec=True,
    )
    @patch(
        'llmfoundry.command_utils.data_prep.convert_delta_to_contrastive_mds.validate_and_get_cluster_info',
        autospec=True,
    )
    @patch(
        'llmfoundry.command_utils.data_prep.convert_delta_to_contrastive_mds.fetch_DT',
        side_effect=Exception('Fetch DT Error'),
        autospec=True,
    )
    @patch(
        'llmfoundry.command_utils.data_prep.convert_delta_to_contrastive_mds.MDSWriter',
        autospec=True,
    )
    def test_main_fetch_DT_exception(
        self,
        mock_mds_writer: MagicMock,
        mock_fetch_DT: MagicMock,
        mock_validate_cluster_info: MagicMock,
        mock_validate_columns: MagicMock,
        mock_workspace_client_class: MagicMock,
    ) -> None:
        """Test that main raises an exception when fetch_DT fails."""
        with patch(
            'databricks.sdk.WorkspaceClient',
            mock_workspace_client_class,
        ):
            from llmfoundry.command_utils.data_prep.convert_delta_to_contrastive_mds import \
                convert_delta_to_contrastive_mds

            # Setup mocks
            mock_workspace_client_instance = MagicMock()
            mock_workspace_client_class.return_value = mock_workspace_client_instance

            mock_validate_cluster_info.return_value = (
                'dbconnect',
                MagicMock(),
                MagicMock(),
            )
            mock_validate_columns.return_value = True

            args: dict[str, Any] = {
                'delta_table_name': 'test_table',
                'http_path': 'http://test_path',
                'cluster_id': 'cluster123',
                'use_serverless': False,
                'output_path': '/output/path',
                'batch_size': 1000,
                'processes': 4,
            }

            with self.assertLogs(
                'llmfoundry.command_utils.data_prep.convert_delta_to_contrastive_mds',
                level='ERROR',
            ) as log:
                with self.assertRaises(Exception) as cm:
                    convert_delta_to_contrastive_mds(**args)

            self.assertIn('Error fetching data: Fetch DT Error', log.output[0])
            self.assertEqual(str(cm.exception), 'Fetch DT Error')

    @patch('databricks.sdk.WorkspaceClient', autospec=True)
    @patch(
        'llmfoundry.command_utils.data_prep.convert_delta_to_contrastive_mds.validate_columns_in_table',
        return_value=True,
        autospec=True,
    )
    @patch(
        'llmfoundry.command_utils.data_prep.convert_delta_to_contrastive_mds.validate_and_get_cluster_info',
        return_value=('dbconnect', MagicMock(), MagicMock()),
        autospec=True,
    )
    @patch(
        'llmfoundry.command_utils.data_prep.convert_delta_to_contrastive_mds.fetch_DT',
        autospec=True,
    )
    @patch(
        'llmfoundry.command_utils.data_prep.convert_delta_to_contrastive_mds.MDSWriter',
        autospec=True,
    )
    def test_main_temporary_directory_handling(
        self,
        mock_mds_writer: MagicMock,
        mock_fetch_DT: MagicMock,
        mock_validate_cluster_info: MagicMock,
        mock_validate_columns: MagicMock,
        mock_workspace_client_class: MagicMock,
    ) -> None:
        with patch(
            'databricks.sdk.WorkspaceClient',
            mock_workspace_client_class,
        ):
            from llmfoundry.command_utils.data_prep.convert_delta_to_contrastive_mds import \
                convert_delta_to_contrastive_mds

            # Setup mocks
            mock_workspace_client_instance = MagicMock()
            mock_workspace_client_class.return_value = mock_workspace_client_instance

            mock_validate_cluster_info.return_value = (
                'dbconnect',
                MagicMock(),
                MagicMock(),
            )
            mock_validate_columns.return_value = True

            mock_mds_instance: MagicMock = MagicMock()
            mock_mds_writer.return_value.__enter__.return_value = mock_mds_instance

            args: dict[str, Any] = {
                'delta_table_name': 'test_table',
                'http_path': 'http://test_path',
                'cluster_id': 'cluster123',
                'use_serverless': False,
                'output_path': '/output/path',
                'batch_size': 1000,
                'processes': 4,
            }

            with patch('tempfile.TemporaryDirectory') as mock_temp_dir:
                mock_temp_dir.return_value.__enter__.return_value = '/tmp/mock_dir'
                # **Update the mock_open data to include negative_passages as a list**
                with patch(
                    'builtins.open',
                    mock_open(
                        read_data=
                        '{"query_text": "sample", "positive_passage": "passage", "negative_passages": []}\n',
                    ),
                ):
                    convert_delta_to_contrastive_mds(**args)
                    mock_temp_dir.assert_called_once()
                    mock_fetch_DT.assert_called_once_with(
                        delta_table_name='test_table',
                        json_output_folder='/tmp/mock_dir',
                        http_path='http://test_path',
                        cluster_id='cluster123',
                        use_serverless=False,
                        json_output_filename='output.jsonl',
                        batch_size=1000,
                        processes=4,
                        DATABRICKS_HOST=mock_workspace_client_instance.config.
                        host,
                        DATABRICKS_TOKEN=mock_workspace_client_instance.config.
                        token,
                    )
                    mock_mds_writer.assert_called_once_with(
                        out='/output/path',
                        columns={
                            'query_text': 'str',
                            'positive_passage': 'str',
                            'negative_passages': 'str',
                        },
                        compression='zstd:7',
                        hashes=['sha1'],
                        size_limit='10mb',
                    )
                    mock_mds_instance.write.assert_called_once_with({
                        'query_text': 'sample',
                        'positive_passage': 'passage',
                        'negative_passages': json.dumps([]),
                    })
