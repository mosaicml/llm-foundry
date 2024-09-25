
# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from composer.models import ComposerModel
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (ColwiseParallel,
                                               PrepareModuleInput,
                                               RowwiseParallel,)
from torch.distributed.tensor.parallel.style import ParallelStyle


def ffn_tp_strategy(model: ComposerModel) -> dict[str, ParallelStyle]:
    TP_LAYERS = {'up_proj', 'down_proj'}

    # validate that all TP_LAYERS are in model
    tp_layers_in_model = set([
        layer for layer in TP_LAYERS for name, _ in model.named_modules()
        if layer in name
    ])
    assert tp_layers_in_model == TP_LAYERS, f'The FFN tensor parallelism strategy requires `model` to have layers {TP_LAYERS}. But `model` is missing layers {TP_LAYERS - tp_layers_in_model}.'

    # generate layer plan
    layer_plan: dict[str, ParallelStyle] = {}
    for name, _ in model.named_modules():
        if name.split('.')[-2:] == ['ffn', 'up_proj']:
            layer_plan[name] = ColwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(-1),
            )
        elif name.split('.')[-2:] == ['ffn', 'down_proj']:
            layer_plan[name] = RowwiseParallel(
                input_layouts=Shard(-1),
                output_layouts=Shard(0),
            )
        elif name.split('.')[-1] == 'ffn':
            layer_plan[name] = PrepareModuleInput(
                input_layouts=Shard(0),
                desired_input_layouts=Replicate(),
                use_local_output=True,
            )

    return layer_plan
