from typing import Union, Dict, Optional

from composer.models import ComposerModel
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel
from torch.distributed.tensor.parallel.style import ParallelStyle
from torch.distributed._tensor import Replicate, Shard


def ffn_tp_strategy(model: ComposerModel) -> Dict[str, ParallelStyle]:

    TP_LAYERS = set(['up_proj', 'down_proj'])

    # validate that all TP_LAYERS are in model
    tp_layers_in_model = set([layer for layer in TP_LAYERS for name, _ in model.named_modules() if layer in name])
    assert tp_layers_in_model == TP_LAYERS, f'The FFN tensor parallelism strategy requires `model` to have layers {TP_LAYERS}. But `model` is missing layers {TP_LAYERS - tp_layers_in_model}.'

    # generate layer plan
    layer_plan: Dict[str, ParallelStyle] = {}
    for name, _ in model.named_modules():
        if 'up_proj' in name:
            layer_plan[name] = ColwiseParallel(
                input_layouts = Replicate(),
                output_layouts = Shard(-1),
            )
        elif 'down_proj' in name:
            layer_plan[name] = RowwiseParallel(
                input_layouts = Shard(-1),
                output_layouts = Replicate(),
            )

    return layer_plan
