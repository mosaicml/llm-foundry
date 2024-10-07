import unittest
from unittest.mock import patch, MagicMock
from argparse import Namespace
import os
import pandas as pd
from collections import defaultdict
from llmfoundry.utils import token_counts_with_collate, create_om_cfg, convert_text_to_mds, parse_args
from llmfoundry.data.finetuning.tasks import _validate_chat_formatted_example, _get_example_type
import datasets

class TestTrainingWorkflow(unittest.TestCase):

    @patch('llmfoundry.utils.is_hf_dataset_path')
    @patch('llmfoundry.utils.is_uc_delta_table')
    @patch('datasets.load_dataset')
    def test_data_loading(self, mock_load_dataset, mock_is_uc_delta_table, mock_is_hf_dataset_path):
        mock_is_hf_dataset_path.return_value = True
        mock_load_dataset.return_value = MagicMock()
        FT_API_args = Namespace(model='mosaicml/mpt-7b', train_data_path='mosaicml/dolly_hhrlhf/train', task_type='INSTRUCTION_FINETUNE', training_duration=3, context_length=2048)

        if mock_is_hf_dataset_path(FT_API_args.train_data_path):
            dataset_id, split = '/'.join(FT_API_args.train_data_path.split('/')[:2]), FT_API_args.train_data_path.split('/')[-1]
            raw_dataset = datasets.load_dataset(dataset_id, split=split)
        else:
            self.fail("HF dataset path mock did not return True")

        self.assertIsNotNone(raw_dataset)

    @patch('pandas.DataFrame.to_json')
    @patch('pandas.read_json')
    @patch('llmfoundry.utils.is_hf_dataset_path', return_value=False)
    @patch('llmfoundry.utils.is_uc_delta_table', return_value=True)
    def test_delta_table_data_loading(self, mock_is_uc_delta_table, mock_is_hf_dataset_path, mock_read_json, mock_to_json):
        mock_df = pd.DataFrame({'example': [1, 2, 3]})
        mock_read_json.return_value = mock_df
        FT_API_args = Namespace(model='mosaicml/mpt-7b', train_data_path='catalog.schema.table', task_type='INSTRUCTION_FINETUNE', training_duration=3, context_length=2048)

        df = mock_df
        df.to_json('dummy_path', orient='records', lines=True)
        raw_dataset = datasets.Dataset.from_pandas(df)

        self.assertIsNotNone(raw_dataset)
        mock_to_json.assert_called_once()

    def test_data_quality_checks(self):
        raw_dataset = [{'prompt': 'test prompt', 'response': 'test response'}]
        format_errors = defaultdict(int)

        for example in raw_dataset:
            try:
                example_format = _get_example_type(example)
            except ValueError:
                format_errors["unknown example type"] += 1
                continue

            if example_format == 'chat':
                try:
                    _validate_chat_formatted_example(example)
                except Exception:
                    format_errors['chat_format_error'] += 1

            elif example_format == 'prompt_response':
                try:
                    _ = example
                except Exception:
                    format_errors['prompt_response_format_error'] += 1

        self.assertEqual(len(format_errors), 0)

    @patch('llmfoundry.utils.token_counts_with_collate')
    def test_token_estimation(self, mock_token_counts_with_collate):
        mock_token_counts_with_collate.return_value = {'ntokens': [1000, 2000, 3000]}
        FT_API_args = Namespace(model='mosaicml/mpt-7b', task_type='INSTRUCTION_FINETUNE', training_duration=3)

        n_epochs = FT_API_args.training_duration if FT_API_args.training_duration is not None else 1
        batch_tokens = token_counts_with_collate(FT_API_args)
        n_billing_tokens_in_dataset = sum(batch_tokens['ntokens'])

        self.assertEqual(n_billing_tokens_in_dataset, 6000)

    @patch('llmfoundry.utils.create_om_cfg')
    @patch('llmfoundry.utils.convert_text_to_mds')
    def test_continued_pretrain(self, mock_convert_text_to_mds, mock_create_om_cfg):
        FT_API_args = Namespace(model='mosaicml/mpt-7b', train_data_path='/tmp/ABT', task_type='CONTINUED_PRETRAIN', training_duration=3, context_length=8)
        temporary_mds_output_path = '/tmp/mds_data_11Jan24_5'

        cfg, tokenizer = MagicMock(), MagicMock()
        mock_create_om_cfg.return_value = (cfg, tokenizer)

        n_samples = mock_convert_text_to_mds.return_value = 10
        n_billing_tokens_in_dataset = n_samples * FT_API_args.context_length

        self.assertEqual(n_billing_tokens_in_dataset, 80)
        mock_convert_text_to_mds.assert_called_once()

if __name__ == '__main__':
    unittest.main()

