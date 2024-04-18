# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
import copy
import json
import os
import warnings

import omegaconf
import pytest
from omegaconf import DictConfig
from omegaconf import OmegaConf as om

from scripts.train.train import main  # noqa: E402


def make_fake_index_file(path: str) -> None:
    """Create a fake index file in the path."""
    fake_index = {
        'shards': [{
            'column_encodings': ['bytes'],
            'column_names': ['tokens'],
            'column_sizes': [None],
            'compression': 'zstd',
            'format': 'mds',
            'hashes': [],
            'raw_data': {
                'basename': 'shard.00000.mds',
                'bytes': 5376759,
                'hashes': {},
            },
            'samples': 328,
            'size_limit': 67108864,
            'version': 2,
            'zip_data': {
                'basename': 'shard.00000.mds.zstd',
                'bytes': 564224,
                'hashes': {},
            }
        }],
        'version': 2
    }
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(fake_index, f)


class TestTrainingYAMLInputs:
    """Validate and tests error handling for the input YAML file."""

    @pytest.fixture
    def cfg(self, foundry_dir: str) -> DictConfig:
        """Create YAML cfg fixture for testing purposes."""
        conf_path: str = os.path.join(
            foundry_dir, 'scripts/train/yamls/pretrain/testing.yaml')
        with open(conf_path, 'r', encoding='utf-8') as config:
            test_cfg = om.load(config)
        assert isinstance(test_cfg, DictConfig)
        return test_cfg

    def test_misspelled_mandatory_params_fail(self, cfg: DictConfig) -> None:
        """Check that mandatory misspelled inputs fail to train."""
        cfg.trai_loader = cfg.pop('train_loader')
        with pytest.raises((omegaconf.errors.MissingMandatoryValue, TypeError)):
            main(cfg)

    def test_missing_mandatory_parameters_fail(self, cfg: DictConfig) -> None:
        """Check that missing mandatory parameters fail to train."""
        mandatory_params = [
            'train_loader',
            'model',
            'tokenizer',
            'optimizer',
            'scheduler',
            'max_duration',
            'eval_interval',
            'max_seq_len',
        ]
        for param in mandatory_params:
            orig_param = cfg.pop(param)
            with pytest.raises(
                (omegaconf.errors.MissingMandatoryValue, NameError,
                 omegaconf.errors.InterpolationKeyError)):
                main(cfg)
            cfg[param] = orig_param

    def test_optional_misspelled_params_raise_warning(self,
                                                      cfg: DictConfig) -> None:
        """Check that warnings are raised for optional misspelled parameters."""
        optional_params = [
            'save_weights_only',
            'save_filename',
            'run_name',
            'progress_bar',
            'python_log_level',
            'eval_first',
            'autoresume',
            'save_folder',
            'fsdp_config',
            'lora_config',
            'eval_loader',
            'icl_tasks_config',
        ]
        old_cfg = copy.deepcopy(cfg)
        for param in optional_params:
            orig_value = cfg.pop(param, None)
            updated_param = param + '-misspelling'
            cfg[updated_param] = orig_value
            with warnings.catch_warnings(record=True) as warning_list:
                try:
                    main(cfg)
                except:
                    pass
                assert any(f'Unused parameter {updated_param} found in cfg.' in
                           str(warning.message) for warning in warning_list)
            # restore configs.
            cfg = copy.deepcopy(old_cfg)

    def test_extra_params_in_optimizer_cfg_errors(self,
                                                  cfg: DictConfig) -> None:
        data_local = './my-copy-c4-opt1'
        make_fake_index_file(f'{data_local}/train/index.json')
        make_fake_index_file(f'{data_local}/val/index.json')
        cfg.train_loader.dataset.local = data_local
        cfg.eval_loader.dataset.local = data_local
        cfg.optimizer.beta2 = 'extra-parameter'
        with pytest.raises(TypeError):
            main(cfg)

    def test_invalid_name_in_optimizer_cfg_errors(self,
                                                  cfg: DictConfig) -> None:
        data_local = './my-copy-c4-opt2'
        make_fake_index_file(f'{data_local}/train/index.json')
        make_fake_index_file(f'{data_local}/val/index.json')
        cfg.optimizer.name = 'invalid-optimizer'
        cfg.train_loader.dataset.local = data_local
        cfg.eval_loader.dataset.local = data_local
        with pytest.raises(ValueError) as exception_info:
            main(cfg)
        assert str(exception_info.value).startswith(
            "Cant't find 'invalid-optimizer' in registry llmfoundry -> optimizers."
        )

    def test_extra_params_in_scheduler_cfg_errors(self,
                                                  cfg: DictConfig) -> None:
        cfg.scheduler.t_warmup_extra = 'extra-parameter'
        with pytest.raises(TypeError):
            main(cfg)

    def test_invalid_name_in_scheduler_cfg_errors(self,
                                                  cfg: DictConfig) -> None:
        cfg.scheduler.name = 'invalid-scheduler'
        with pytest.raises(ValueError) as exception_info:
            main(cfg)
        assert str(exception_info.value).startswith(
            "Cant't find 'invalid-scheduler' in registry llmfoundry -> schedulers."
        )

    def test_no_label_multiple_eval_datasets(self, cfg: DictConfig) -> None:
        data_local = './my-copy-c4-multi-eval'
        make_fake_index_file(f'{data_local}/train/index.json')
        make_fake_index_file(f'{data_local}/val/index.json')
        cfg.train_loader.dataset.local = data_local
        # Set up multiple eval datasets
        first_eval_loader = cfg.eval_loader
        first_eval_loader.dataset.local = data_local
        second_eval_loader = copy.deepcopy(first_eval_loader)
        # Set the first eval dataloader to have no label
        first_eval_loader.label = None
        second_eval_loader.label = 'eval_1'
        cfg.eval_loader = om.create([first_eval_loader, second_eval_loader])
        with pytest.raises(ValueError) as exception_info:
            main(cfg)
        assert str(
            exception_info.value
        ) == 'When specifying multiple evaluation datasets, each one must include the \
                            `label` attribute.'
