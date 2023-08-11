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

from scripts.eval.eval import main  # noqa: E402


class TestEvalYAMLInputs:
    """Validate and tests error handling for the input YAML file."""

    @pytest.fixture
    def cfg(self) -> DictConfig:
        """Create YAML cfg fixture for testing purposes."""
        conf_path: str = os.path.join(repo_dir,
                                      'scripts/eval/yamls/hf_eval.yaml')
        with open(conf_path, 'r', encoding='utf-8') as config:
            test_cfg = om.load(config)
        assert isinstance(test_cfg, DictConfig)
        return test_cfg

    def test_mispelled_mandatory_params_fail(self, cfg: DictConfig) -> None:
        """Check that mandatory mispelled inputs fail to train."""
        mandatory_params = [
            'max_seq_len',
            'device_eval_batch_size',
            'precision',
            'model_configs',
        ]
        mandatory_configs = ['models', 'icl_tasks']
        for p in mandatory_params + mandatory_configs:
            with pytest.raises((omegaconf.errors.ConfigKeyError,
                                omegaconf.errors.InterpolationKeyError)):
                cfg[p + '-mispelled'] = cfg.pop(p)
                main(cfg)
                cfg[p] = cfg.pop(p + '-mispelled')

    def test_optional_mispelled_params_raise_warning(self,
                                                     cfg: DictConfig) -> None:
        """Check that warnings are raised for optional mispelled parameters."""
        optional_params = [
            'seed',
            'dist_timeout',
            'run_name',
            'num_retries',
            'loggers',
            'model_gauntlet',
            'fsdp_config',
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
