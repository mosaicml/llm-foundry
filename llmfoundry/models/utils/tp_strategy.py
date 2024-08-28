from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel
from torch.distributed._tensor import Replicate, Shard

from llmfoundry.registry import tp_strategy

def ffn_tp_strategy(model):
    TP_LAYERS = set(['up_proj', 'down_proj'])

    # validate that all TP_LAYERS are in model
    tp_layers_in_model = set([name for name, _ in model.named_modules() if name in TP_LAYERS])
    assert tp_layers_in_model == TP_LAYERS, f'This tensor parallelism strategy requires `model` to have layers {TP_LAYERS}. But `model` is missing layers {TP_LAYERS - tp_layers_in_model}.'

    # generate layer plan
    layer_plan = {}
    for name, _ in model.named_modules():
        if 'up_proj' in name:
            layer_plan[name] = ColwiseParallel(input_layouts=Replicate(), output_layouts=Shard(-1))
        if 'down_proj' in name:
            layer_plan[name] = RowwiseParallel(input_layouts=Shard(-1), output_layouts=Shard(0))

    return layer_plan, model


tp_strategy.register('ffn', func=ffn_tp_strategy)
