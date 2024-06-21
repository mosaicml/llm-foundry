# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest
from torch import distributed

from llmfoundry.models.utils.config_moe_args import (
    config_megablocks_moe_args,
    get_megablocks_device_mesh,
)


@pytest.mark.gpu
@pytest.mark.world_size(2)
def test_config_megablocks_moe_args_pg():
    ffn_config_base: dict[str, Any] = {
        'moe_world_size': 1,
        'ffn_type': 'mb_moe',
        'fc_type': 'torch',
    }

    ffn_config_str = ffn_config_base.copy()
    ffn_config_str['lbl_process_group'] = 'global_group'

    ffn_config_pg = ffn_config_base.copy()
    ffn_config_pg['lbl_process_group'] = distributed.group.WORLD

    output_str = config_megablocks_moe_args(
        ffn_config=ffn_config_str,
        d_model=128,
        expansion_ratio=4,
        n_layers=2,
        get_device_mesh=get_megablocks_device_mesh,
    )
    output_pg = config_megablocks_moe_args(
        ffn_config=ffn_config_pg,
        d_model=128,
        expansion_ratio=4,
        n_layers=2,
        get_device_mesh=get_megablocks_device_mesh,
    )

    assert output_str['lbl_process_group'] == output_pg['lbl_process_group']
