# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest

from llmfoundry.utils.config_utils import (
    process_init_device,
    update_config_with_batch_size_info,
)


def test_update_config_with_batch_size_info():
    config = {}
    config = update_config_with_batch_size_info(config, 1, 2, 3)

    assert config['n_gpus'] == 1
    assert config['device_train_batch_size'] == 1
    assert config['device_train_microbatch_size'] == 2
    assert config['device_train_grad_accum'] == 3
    assert config['device_eval_batch_size'] == 2


@pytest.mark.parametrize('shard_degree_specified', [True, False])
@pytest.mark.parametrize('replicate_degree_specified', [True, False])
@pytest.mark.parametrize('should_shard_only', [True, False])
def test_moe_fsdp_config_ffn_config(
    shard_degree_specified: bool,
    replicate_degree_specified: bool,
    should_shard_only: bool,
):
    model_cfg = {
        'moe_world_size': 4,
        'lbl_process_group': 'not_real',
        'fc_type': 'torch',
        'ffn_config': {
            'ffn_type': 'mb_moe',
        },
    }
    fsdp_config = {}
    if shard_degree_specified and replicate_degree_specified:
        if should_shard_only:
            fsdp_config['data_parallel_shard_degree'] = 8
            fsdp_config['data_parallel_replicate_degree'] = 1
        else:
            fsdp_config['data_parallel_shard_degree'] = 4
            fsdp_config['data_parallel_replicate_degree'] = 2
    elif shard_degree_specified:
        if should_shard_only:
            fsdp_config['data_parallel_shard_degree'] = 8
        else:
            fsdp_config['data_parallel_shard_degree'] = 4
    elif replicate_degree_specified:
        if should_shard_only:
            fsdp_config['data_parallel_replicate_degree'] = 1
        else:
            fsdp_config['data_parallel_replicate_degree'] = 2

    # Ensure the ffn_config's device_mesh is set correctly using the fsdp_config
    with patch(
        'composer.utils.dist.get_world_size',
        return_value=8,
    ), patch('catalogue.Registry.__contains__', return_value=True):
        _ = process_init_device(model_cfg, fsdp_config)

        if should_shard_only or (
            not shard_degree_specified and not replicate_degree_specified
        ):
            assert model_cfg['ffn_config']['device_mesh'] == [8]
        else:
            assert model_cfg['ffn_config']['device_mesh'] == [2, 4]
