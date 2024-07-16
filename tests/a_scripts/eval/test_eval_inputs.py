# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
import copy
import os

import omegaconf
import pytest
from omegaconf import DictConfig
from omegaconf import OmegaConf as om

from llmfoundry.command_utils import evaluate  # noqa: E402


class TestHuggingFaceEvalYAMLInputs:
    """Validate and tests error handling for the input YAML file."""

    @pytest.fixture
    def cfg(self, foundry_dir: str) -> DictConfig:
        """Create YAML cfg fixture for testing purposes."""
        conf_path: str = os.path.join(
            foundry_dir,
            'scripts/eval/yamls/hf_eval.yaml',
        )
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
            with pytest.raises((
                omegaconf.errors.ConfigKeyError,
                omegaconf.errors.InterpolationKeyError,
                omegaconf.errors.MissingMandatoryValue,
                TypeError,
                ValueError,
            )):
                cfg[p + '-mispelled'] = cfg.pop(p)
                evaluate(cfg)
                cfg[p] = cfg.pop(p + '-mispelled')

    def test_optional_mispelled_params_raise_error(
        self,
        cfg: DictConfig,
    ) -> None:
        """Check that warnings are raised for optional mispelled parameters."""
        optional_params = [
            'seed',
            'dist_timeout',
            'run_name',
            'num_retries',
            'loggers',
            'eval_gauntlet',
            'fsdp_config',
            'eval_loader',
        ]
        old_cfg = copy.deepcopy(cfg)
        for param in optional_params:
            orig_value = cfg.pop(param, None)
            updated_param = param + '-mispelling'
            cfg[updated_param] = orig_value
            with pytest.raises(ValueError):
                evaluate(cfg)
            # restore configs.
            cfg = copy.deepcopy(old_cfg)


class TestMPTEvalYAMLInputs:

    @pytest.fixture
    def cfg(self, foundry_dir: str) -> DictConfig:
        """Create YAML cfg fixture for testing purposes."""
        conf_path: str = os.path.join(
            foundry_dir,
            'scripts/eval/yamls/mpt_eval.yaml',
        )
        with open(conf_path, 'r', encoding='utf-8') as config:
            test_cfg = om.load(config)

        test_cfg.icl_tasks[0].dataset_uri = os.path.join(
            foundry_dir,
            'scripts',
            test_cfg.icl_tasks[0].dataset_uri,
        )

        # make tests use cpu initialized transformer models only
        test_cfg.models[0].model.init_device = 'cpu'
        test_cfg.models[0].model.attn_config.attn_impl = 'torch'
        test_cfg.models[0].model.loss_fn = 'torch_crossentropy'
        test_cfg.precision = 'fp32'
        assert isinstance(test_cfg, DictConfig)
        return test_cfg

    def test_empty_load_path_raises_error(self, cfg: DictConfig) -> None:
        """Check that empty load paths for mpt models raise an error."""
        error_string = 'MPT causal LMs require a load_path to the checkpoint for model evaluation.' \
            + ' Please check your yaml and the model_cfg to ensure that load_path is set.'
        cfg.models[0].load_path = None
        with pytest.raises(ValueError, match=error_string):
            evaluate(cfg)
