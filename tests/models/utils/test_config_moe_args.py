# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

from llmfoundry.models.utils.config_moe_args import (
    config_megablocks_moe_args,
    get_megablocks_device_mesh,
)


@pytest.mark.gpu
def test_config_megablocks_moe_args_error():
    ffn_config_base: dict[str, Any] = {
        'moe_world_size': 1,
        'lbl_process_group': 'not_real',
        'ffn_type': 'mb_moe',
        'fc_type': 'torch',
    }

    with pytest.raises(ValueError):
        config_megablocks_moe_args(
            ffn_config=ffn_config_base,
            d_model=128,
            expansion_ratio=4,
            n_layers=2,
            get_device_mesh=get_megablocks_device_mesh,
        )
