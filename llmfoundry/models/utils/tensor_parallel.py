# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

###
# The code in this file is copied from torch/distributed/tensor/parallel/style.py and
# torch/distributed/_tensor/redistribute.py. PairwiseSequenceParallel is not available in torch 2.0.0 release and
# this copying makes it available for torch 2.0.0. We will remove this code once we move to newer release of torch.
###
from typing import List, Optional

import torch
from .misc import is_torch_2_or_higher
# Each of these are in their own try/except block otherwise process_file in
# scripts/inference/convert_composer_to_hf.py doesn't like it due to the import
# shenanigans from the HF library. https://github.com/huggingface/transformers/pull/23725
try:
    import torch.distributed._tensor.redistribute as redist  # type: ignore
except:
    pass
try:
    from torch.distributed._tensor import DeviceMesh, DTensor  # type: ignore
except:
    pass
try:
    from torch.distributed._tensor.placement_types import (Placement, Replicate)
except:
    pass
try:
    from torch.distributed._tensor.redistribute import redistribute_dtensor  # type: ignore
except:
    pass
try:
    from torch.distributed.tensor.parallel import (ParallelStyle, make_input_replicate_1d, make_input_shard_1d, make_output_shard_1d)
except:
    pass
try:
    from torch.distributed.tensor.parallel._utils import (_prepare_input_validate, _prepare_output_validate)
except:
    pass


def backward(ctx, grad_output: 'dtensor.DTensor'):  # type: ignore[override]
    previous_placement = ctx.previous_placement
    previous_device_mesh = ctx.previous_device_mesh
    # When we run backward pass of redistribute (i.e. manual redistribute from
    # user code instead of torch_dispatch), we scan first and see if we need
    # to change the target placement for one special case:
    #   replicate -> partial.
    # In this case we keep the grad as replicate, this is because we don't
    # want to convert the replicated gradients back to partial, although
    # that's logically conform with the same layout, converting the gradients
    # back to partial is acutally useless as you would have to do reduce later
    # which would be more expensive than keeping it replicate! For this reason,
    # we keep the replicate grad here.
    # TODO: see if this make sense for all cases.
    target_placements: List[Placement] = []
    for current, target in zip(grad_output.placements, previous_placement):
        if not current.is_partial() and target.is_partial():
            # keep target placement to replicate instead of partial in this case
            target_placements.append(Replicate())
        else:
            target_placements.append(target)

    return (
        redistribute_dtensor(grad_output, previous_device_mesh,
                             target_placements),
        None,
        None,
    )


if is_torch_2_or_higher():
    # Monkey-patch Redistribute.backward to have a fix for PairwiseSequenceParallel
    # See https://github.com/pytorch/pytorch/pull/94369/files
    redist.Redistribute.backward = backward

    class PairwiseSequenceParallel(ParallelStyle):
        """PairwiseSequenceParallel concatenate colwise and rowwise styles as a.

        fixed pair together with sequence parallel like what Megatron-LM Sequence
        parallel (https://arxiv.org/pdf/2205.05198.pdf) is doing. We assume both
        input and output need to be sharded DTensors.

        .. warning::     PairwiseSequenceParallel only supports ``nn.Multihead
        Attention``,     ``nn.Transformer`` or even-number-layer MLP for now.
        """

        def __init__(self) -> None:
            super().__init__(make_input_reshard_replicate,
                             make_output_reshard_tensor)


    @_prepare_input_validate  # type: ignore[arg-type] # pyre-ignore[56]
    def make_input_reshard_replicate(
        input: torch.Tensor,
        device_mesh: DeviceMesh,
    ) -> DTensor:
        """To construct a Sharded DTensor from a tensor on different ranks and then.

        convert to a replicate DTensor.

        Args:
            input (:class:`torch.Tensor`): The input tensor on each rank which consists of a global DTensor
                sharded on dimension ``0`` over the 1-D :class:`DeviceMesh`
                and then the sharded DTensor is converted to a replicate DTensor.
            device_mesh (:class:`DeviceMesh`, optional): The 1-D device mesh where ``input`` will be sharded.
                If :class:`DeviceMesh` is not 1-D, an exception will be thrown.
                Default: ``None``
        Returns:
            A :class:`DTensor` sharded on dimension ``0`` over ``device_mesh``
                and then converted to replicate.
        """
        return make_input_replicate_1d(  # type: ignore[call-arg]
            make_input_shard_1d(input, device_mesh, dim=0),
            device_mesh  # type: ignore[call-arg]
        )


    @_prepare_output_validate  # type: ignore[arg-type] # pyre-ignore[56]
    def make_output_reshard_tensor(
        output: DTensor,
        device_mesh: Optional[DeviceMesh] = None,
    ) -> torch.Tensor:
        """Convert Output DTensor to a sharded DTensor and return the local tensor.

        Args:
            output (:class:`DTensor`): Output of module to be converted.
            device_mesh (:class:`DeviceMesh`, optional): Object needed to shard the output and it needs to be a 1D ``device_mesh``
                and we will throw exceptions if a non-1D ``device_mesh`` is passed in.
                If no ``device_mesh`` is passed in, we will reuse the one from output.
                Default: ``None``
        Return:
            A :class:`torch.Tensor` object converted from output DTensor.
        """
        return make_output_shard_1d(
            output, device_mesh).to_local()  # type: ignore[call-arg, attr-defined]
