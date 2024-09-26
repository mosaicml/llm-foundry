# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from composer.models import ComposerModel
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
)
from torch.distributed.tensor.parallel.style import ParallelStyle


def ffn_tp_strategy(model: ComposerModel) -> dict[str, ParallelStyle]:
    TP_LAYERS = {'ffn', 'ffn.up_proj', 'ffn.down_proj'}

    # Validate that all TP_LAYERS are in model
    tp_layers_in_model = {
        layer for layer in TP_LAYERS for name, _ in model.named_modules()
        if layer in name
    }
    if tp_layers_in_model != TP_LAYERS:
        raise RuntimeError(
            f'The FFN tensor parallelism strategy requires `model` to have layers {TP_LAYERS}. But `model` is missing layers {TP_LAYERS - tp_layers_in_model}.',
        )

    # Generate layer plan
    layer_plan: dict[str, ParallelStyle] = {}
    for name, _ in model.named_modules():
        # Before the ffn layer starts, distribute the input data for proper TP use
        # Inputs are currently sharded across the batch dimension (dim 0) as is done in standard DDP
        # Inputs will be replicated across hidden dimension (dim 1) via allgather
        if name.split('.')[-1] == 'ffn':
            layer_plan[name] = PrepareModuleInput(
                input_layouts=Shard(0),
                desired_input_layouts=Replicate(),
                use_local_output=True,
            )
        # Shard the ffn.up_proj weight matrix across its columns
        # Inputs are already replicated across each TP group
        # Outputs will be sharded along the hidden dimension (dim 1) via allgather
        elif name.split('.')[-2:] == ['ffn', 'up_proj']:
            layer_plan[name] = ColwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(-1),
            )
        # Shard the ffn.down_proj weight matrix across its rows
        # Inputs are sharded along the hidden dimension (dim 1)
        # Outputs will be sharded along batch dimension (dim 0) via allreduce
        elif name.split('.')[-2:] == ['ffn', 'down_proj']:
            layer_plan[name] = RowwiseParallel(
                input_layouts=Shard(-1),
                output_layouts=Shard(0),
            )

    return layer_plan
