# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from torch.distributed._tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
)

from llmfoundry.models.mpt.modeling_mpt import ComposerMPTCausalLM
from llmfoundry.utils.builders import build_tp_strategy


def test_ffn_tp_strategy_layer_plan():
    # Actual layer plan
    tp_config = {
        'strategy': 'ffn',
    }

    model_cfg = {
        'name': 'mpt_causal_lm',
        'd_model': 128,
        'n_heads': 4,
        'n_layers': 3,
        'expansion_ratio': 1,
        'max_seq_len': 16,
        'vocab_size': 50368,
        'attn_config': {
            'attn_impl': 'flash',
        },
    }
    model = ComposerMPTCausalLM(**model_cfg)
    layer_plan = build_tp_strategy(tp_config['strategy'], model)

    # Expected layer plan
    _expected_layer_plan = {
        'ffn':
            PrepareModuleInput(
                input_layouts=Shard(0),
                desired_input_layouts=Replicate(),
                use_local_output=True,
            ),
        'ffn.down_proj':
            RowwiseParallel(
                input_layouts=Shard(-1),
                output_layouts=Shard(0),
            ),
        'ffn.up_proj':
            ColwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(-1),
            ),
    }
    expected_layer_plan = {
        f'model.transformer.blocks.{layer_idx}.{name}': layer_plan
        for name, layer_plan in _expected_layer_plan.items()
        for layer_idx in range(model_cfg['n_layers'])
    }

    # Compare expected and actual layer plans
    for (n1, lp1), (n2, lp2) in zip(
        sorted(expected_layer_plan.items()),
        sorted(layer_plan.items()),
    ):
        assert n1 == n2
        assert type(lp1) == type(lp2)
        if isinstance(
            lp1,
            PrepareModuleInput,
        ) and isinstance(lp2, PrepareModuleInput):
            assert lp1.input_layouts == lp2.input_layouts
            assert lp1.desired_input_layouts == lp2.desired_input_layouts
            assert lp1.use_local_output == lp2.use_local_output
        elif (
            isinstance(lp1, ColwiseParallel) and
            isinstance(lp2, ColwiseParallel)
        ) or (
            isinstance(lp1, RowwiseParallel) and
            isinstance(lp2, RowwiseParallel)
        ):
            assert lp1.input_layouts == lp2.input_layouts
            assert lp1.output_layouts == lp2.output_layouts
            assert lp1.use_local_output == lp2.use_local_output
        else:
            raise ValueError(f'Layer plan of wrong type: {type(layer_plan)}')
