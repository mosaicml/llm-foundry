# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Helper functions for computing parameter counts for MPT model.

Use if generic `sum(p.numel() for p in self.parameters())`
style computation does not account for MoE parameter sharding.
The helper functions in this file account for MoE parameter
sharding in the parameter count calculation. The functions below
calculate the total parameter count and the active parameter count.
Note: MPT has both n_total_params and n_active_params methods.
"""

from typing import Union

from torch import Tensor, nn
from torch.distributed._tensor import DTensor

from llmfoundry.layers_registry import ffns_with_megablocks


def module_n_params(module: nn.Module) -> int:
    """Gets the number of parameters in this module excluding child modules.

    Args:
        module (nn.Module): Module of which we get the number of parameters.

    Returns:
        An int for the number of parameters in this module.
    """
    n_params = 0
    for p in module.parameters(recurse=False):
        n_params += p.numel()
    return n_params


def _dtensor_safe_check_numel(tensor: Union[Tensor, DTensor]) -> int:
    if isinstance(tensor, DTensor):
        tensor = tensor._local_tensor
    return tensor.numel()


def megablocks_n_total_params(mpt_model) -> int:  # type: ignore
    """Calculates the number of parameters in a MegaBlocks enabled MPT model.

    MoE experts are sharded across workers. This function scans for MegaBlocks
    modules then multiplies expert params count by MoE world size.

    Args:
        mpt_model (ComposerMPTCausalLM): MPT model of which the number of
            parameters is calculated.

    Returns:
        An int for the total number of parameters in this MPT model.
    """
    import megablocks

    moe_world_size = mpt_model.config.ffn_config.get('moe_world_size')

    if mpt_model.config.ffn_config.get('moe_weight_parallelism', False):
        # If MegaBlocks shards experts, the total sharding world size
        # must be increased by the degree to which MegaBlocks shards the
        # experts.
        mb_args = mpt_model.model.transformer.mb_args
        moe_world_size *= mb_args.weight_parallel_group.size()

    n_total_params = 0
    for module in mpt_model.modules():
        if isinstance(
                module,
            (megablocks.layers.mlp.SparseMLP, megablocks.layers.mlp.MLP)):
            n_w1 = _dtensor_safe_check_numel(module.w1)
            n_total_params += n_w1 * moe_world_size
            n_w2 = _dtensor_safe_check_numel(module.w2)
            n_total_params += n_w2 * moe_world_size

            # GLU has an extra weight
            if hasattr(module, 'v1'):
                n_v1 = _dtensor_safe_check_numel(module.v1)
                n_total_params += n_v1 * moe_world_size
        else:
            n_total_params += module_n_params(module)

    return n_total_params


def megablocks_n_active_params(mpt_model) -> int:  # type: ignore
    """Calculates the number of active parameters in a MegaBlocks enabled MPT.

    This requires we calculate the number of elements per expert and
    multiply this by top k.

    Args:
        mpt_model (ComposerMPTCausalLM): MPT model of which the number of
            active parameters is calculated.

    Returns:
        An int for the active number of parameters in this MPT model.
    """
    import megablocks

    moe_num_experts = mpt_model.config.ffn_config.get('moe_num_experts', 1)
    moe_world_size = mpt_model.config.ffn_config.get('moe_world_size')

    local_experts = moe_num_experts / moe_world_size  # if local_experts is < 1, then the expert is sharded
    if mpt_model.config.ffn_config.get('moe_weight_parallelism', False):
        mb_args = mpt_model.model.transformer.mb_args
        local_experts /= mb_args.weight_parallel_group.size()

    moe_top_k = mpt_model.config.ffn_config.get('moe_top_k', 1)
    n_active_params = 0
    for module in mpt_model.modules():
        if isinstance(
                module,
            (megablocks.layers.mlp.SparseMLP, megablocks.layers.mlp.MLP)):
            n_w1 = _dtensor_safe_check_numel(module.w1)
            n_active_params += int(n_w1 / local_experts * moe_top_k)
            n_w2 = _dtensor_safe_check_numel(module.w2)
            n_active_params += int(n_w2 / local_experts * moe_top_k)

            # GLU has an extra weight
            if hasattr(module, 'v1'):
                n_v1 = _dtensor_safe_check_numel(module.v1)
                n_active_params += int(n_v1 / local_experts * moe_top_k)
        else:
            n_active_params += module_n_params(module)

    return n_active_params


def mpt_get_total_params(mpt_model) -> int:  # type: ignore
    """Calculates the total parameter count of an MPT model.

    Note: Must be called before model parameters are sharded by FSDP.

    Args:
        mpt_model (ComposerMPTCausalLM): MPT model of which the number of
            active parameters is calculated.

    Returns:
        An int for the total number of parameters in this MPT model.
    """
    if mpt_model.config.ffn_config['ffn_type'] in ffns_with_megablocks:
        return megablocks_n_total_params(mpt_model)
    else:
        return sum(p.numel() for p in mpt_model.parameters())


def mpt_get_active_params(mpt_model) -> int:  # type: ignore
    """Calculates the total parameter count of an MPT model.

    Note: Must be called before model parameters are sharded by FSDP.

    Args:
        mpt_model (ComposerMPTCausalLM): MPT model of which the number of
            active parameters is calculated.

    Returns:
        An int for the active number of parameters in this MPT model.
    """
    if mpt_model.config.ffn_config['ffn_type'] in ffns_with_megablocks:
        params = megablocks_n_active_params(mpt_model)
    else:
        params = sum(p.numel() for p in mpt_model.parameters())
    if not mpt_model.model.transformer.config.tie_word_embeddings:
        # Embedding layers are lookup tables, therefore are not counted in the FLOP computation
        params -= _dtensor_safe_check_numel(
            mpt_model.model.transformer.wte.weight)
    return params
