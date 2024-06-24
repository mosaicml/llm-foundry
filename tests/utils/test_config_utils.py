# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from llmfoundry.utils.config_utils import update_config_with_batch_size_info


def test_update_config_with_batch_size_info():
    config = {}
    config = update_config_with_batch_size_info(config, 1, 2, 3)

    assert config['n_gpus'] == 1
    assert config['device_train_batch_size'] == 1
    assert config['device_train_microbatch_size'] == 2
    assert config['device_train_grad_accum'] == 3
    assert config['device_eval_batch_size'] == 2
