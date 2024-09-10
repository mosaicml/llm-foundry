from typing import Union, Dict, Optional

from composer.models import ComposerModel
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel,  PrepareModuleInput
from torch.distributed.tensor.parallel.style import ParallelStyle
from torch.distributed._tensor import Replicate, Shard, Placement


# class SerializableColwiseParallel(ColwiseParallel):
#     @classmethod
#     def __struct_hook__(cls, *args, **kwargs):
#         return 'torch.distributed.tensor.parallel.ColwiseParallel'

#     def __reduce__(self):
#         return (SerializableColwiseParallel, ())


# class SerializableRowwiseParallel(RowwiseParallel):
#     @classmethod
#     def __struct_hook__(cls, *args, **kwargs):
#         return 'torch.distributed.tensor.parallel.RowwiseParallel'

#     def __reduce__(self):
#         return (SerializableRowwiseParallel, ())


class GatherColwiseParallel(ColwiseParallel):
    """ColwiseParallel layer that allgathers inputs and optionally reshards outputs."""
    def __init__(
        self,
        *,
        #input_layouts: Optional[Placement] = None,
        output_layouts: Optional[Placement] = None,
        use_local_output: bool = True
    ):
        super().__init__()
        # Inputs over the TP dimension are sharded by device batches.
        self.input_layouts = (Shard(0), )
        # All-gather inputs so that each GPU now has the same input activations.
        self.desired_input_layouts = (Replicate(), )
        self.output_layouts = (output_layouts or Shard(-1), )
        self.use_local_output = use_local_output


def retrieve_layer_plan(model):
    layer_plan = {}
    for name, _ in model.named_modules():
        split_name = name.split('.')
        # First block -- allgathers device batches from TP group devices. Residual stream activations
        # will be full, allgathered device batches from all TP group devices.
        if len(split_name) >= 2 and split_name[-2] == 'blocks' and split_name[-1] == '0':
            print(f"using PrepareModuleInput, (in=Shard(0), desired_in=Replicate) for module {name}")
            layer_plan[name] = PrepareModuleInput(
                input_layouts = Shard(0),
                desired_input_layouts = Replicate(),
                use_local_output = True,
            )
        # Wqkv -- inputs are all samples from TP group, but to keep KV cache unique to each device,
        # we need to reshard device batches back to TP group devices.
        elif 'Wqkv' in name:
            print(f"using ColwiseParallel, (in=Replicate, out=Shard(0)) for module {name}")
            layer_plan[name] = ColwiseParallel(
                input_layouts = Replicate(),
                output_layouts = Shard(0),
            )
        # Attn out_proj -- inputs should again be allgathered from TP group devices and remain allgathered.
        elif 'out_proj' in name:
            print(f"using GatherColwiseParallel, (out=Replicate) for module {name}")
            layer_plan[name] = GatherColwiseParallel(
                output_layouts = Replicate(),
            )
        # FFN up_proj -- inputs are already allgathered but should get sharded along the embedding dimension.
        if 'up_proj' in name or 'gate_proj' in name:
            print(f"using ColwiseParallel, [Replicate, Shard(-1)] for module {name}")
            layer_plan[name] = ColwiseParallel(
                input_layouts = Replicate(),
                output_layouts = Shard(-1),
            )
        # FFN down_proj -- inputs are sharded along the embedding dimension but should get allreduced.
        if 'down_proj' in name:
            print(f"using RowwiseParallel, [Shard(-1), Replicate] for module {name}")
            layer_plan[name] = RowwiseParallel(
                input_layouts = Shard(-1),
                output_layouts = Replicate(),
            )
        # LM head reshards device batches back to TP group devices.
        elif 'lm_head' in name:
            print(f"using ColwiseParallel, [Replicate, Shard(0)] for module {name}")
            layer_plan[name] = ColwiseParallel(
                input_layouts = Replicate(),
                output_layouts = Shard(0),
            )
    return layer_plan

def ffn_tp_strategy(model: ComposerModel) -> Dict[str, ParallelStyle]:

    return retrieve_layer_plan(model)
    TP_LAYERS = set(['up_proj', 'down_proj'])

    # validate that all TP_LAYERS are in model
    tp_layers_in_model = set([layer for layer in TP_LAYERS for name, _ in model.named_modules() if layer in name])
    assert tp_layers_in_model == TP_LAYERS, f'The FFN tensor parallelism strategy requires `model` to have layers {TP_LAYERS}. But `model` is missing layers {TP_LAYERS - tp_layers_in_model}.'

    # generate layer plan
    layer_plan: Dict[str, ParallelStyle] = {}
    for name, _ in model.named_modules():
        split_name = name.split('.')

        if len(split_name) >= 2 and split_name[-2] == 'blocks' and split_name[-1] == '0':
            layer_plan[name] = PrepareModuleInput(
                input_layouts = Shard(0),
                desired_input_layouts = Replicate(),
                use_local_output = True,
            )
        elif 'up_proj' in name:
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
