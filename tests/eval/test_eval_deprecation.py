# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import unittest
import warnings

from llmfoundry.command_utils.eval import evaluate_model
from llmfoundry.utils.warnings import VersionedDeprecationWarning


class TestEvaluateModelDeprecation(unittest.TestCase):

    def setUp(self):
        self.common_args = { # type: ignore
            'tokenizer': {
                'name': 'test_tokenizer',
            },
            'model': {
                'name': 'test_model',
            },
            'model_name': 'test',
            'dist_timeout': 60,
            'run_name': 'test_run',
            'seed': 42,
            'icl_tasks': [],
            'max_seq_len': 512,
            'device_eval_batch_size': 1,
            'eval_gauntlet_config': None,
            'eval_loader_config': None,
            'loggers': [],
            'python_log_level': None,
            'precision': 'fp32',
            'eval_gauntlet_df': None,
            'eval_subset_num_batches': 1,
            'icl_subset_num_batches': None,
            'callback_configs': None,
            'metadata': None,
            'logged_config': {},
        }

    def test_no_deprecation_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            import composer.utils.parallelism
            deprecated_fsdp_args = list(
                composer.utils.parallelism.FSDPConfig.__annotations__.keys(),
            )
            print(deprecated_fsdp_args)

            try:
                parallelism_config = {'fsdp': {'verbose': True}}
                evaluate_model(
                    **self.common_args,
                    parallelism_config=parallelism_config,
                )
            except ValueError as ve:
                if 'parallelism_config cannot contain deprecated fsdp_config arguments.' in str(
                    ve,
                ):
                    self.fail(
                        'Raised ValueError about deprecated fsdp_config arguments',
                    )
                elif 'Both fsdp_config and parallelism_config cannot be provided at the same time.' in str(
                    ve,
                ):
                    self.fail(
                        'Raised ValueError about both configs being provided',
                    )
            except Exception:
                pass

            deprecation_warnings = [
                warning for warning in w
                if isinstance(warning.message, VersionedDeprecationWarning)
            ]
            if deprecation_warnings:
                self.fail('VersionedDeprecationWarning was raised')

    def test_deprecation_warning_with_deprecated_arg(self):
        # Use assertRaises to catch the expected ValueError
        with self.assertRaises(ValueError) as context:
            # Directly call evaluate_model; do not use try-except here
            evaluate_model(
                **self.common_args,
                parallelism_config={'activation_checkpointing': True},
            )

        # Assert that the correct error message is in the exception
        self.assertIn(
            'parallelism_config cannot contain deprecated fsdp_config arguments.',
            str(context.exception),
        )

    def test_deprecation_warning_with_fsdp_config(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')

            try:
                evaluate_model(
                    **self.common_args,
                    parallelism_config=None,
                    fsdp_config={'verbose': True},
                )
            except Exception:
                pass

            self.assertTrue(
                any(
                    issubclass(warning.category, VersionedDeprecationWarning)
                    for warning in w
                ),
            )

    def test_error_with_both_fsdp_and_parallelism_config(self):
        with self.assertRaises(ValueError) as context:
            evaluate_model(
                **self.common_args,
                parallelism_config={'some_arg': True},
                fsdp_config={'some_arg': True},
            )

        self.assertIn(
            'Both fsdp_config and parallelism_config cannot be provided at the same time.',
            str(context.exception),
        )
