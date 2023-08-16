# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
import copy
import os
import sys
import warnings

import omegaconf
import pytest
from omegaconf import DictConfig
from omegaconf import OmegaConf as om

# Add repo root to path so we can import scripts and test it
repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(repo_dir)

from scripts.train.train import main  # noqa: E402


class TestTrainingYAMLInputs:
    """Validate and tests error handling for the input YAML file."""

    @pytest.fixture
    def cfg(self) -> DictConfig:
        """Create YAML cfg fixture for testing purposes."""
        conf_path: str = os.path.join(
            repo_dir, 'scripts/train/yamls/pretrain/testing.yaml')
        with open(conf_path, 'r', encoding='utf-8') as config:
            test_cfg = om.load(config)
        assert isinstance(test_cfg, DictConfig)
        return test_cfg

    def test_mispelled_mandatory_params_fail(self, cfg: DictConfig) -> None:
        """Check that mandatory mispelled inputs fail to train."""
        cfg.trai_loader = cfg.pop('train_loader')
        with pytest.raises(omegaconf.errors.ConfigAttributeError):
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
            'precision',
            'max_seq_len',
        ]
        for param in mandatory_params:
            orig_param = cfg.pop(param)
            with pytest.raises(
                (omegaconf.errors.ConfigAttributeError, NameError)):
                main(cfg)
            cfg[param] = orig_param

    def test_optional_mispelled_params_raise_warning(self,
                                                     cfg: DictConfig) -> None:
        """Check that warnings are raised for optional mispelled parameters."""
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
            'eval_loader_config',
            'icl_tasks_config',
        ]
        old_cfg = copy.deepcopy(cfg)
        for param in optional_params:
            orig_value = cfg.pop(param, None)
            updated_param = param + '-mispelling'
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
