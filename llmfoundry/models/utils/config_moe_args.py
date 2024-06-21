# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Helper function to configure MPT with MoEs."""

import inspect
from typing import Callable, Optional, Union

import torch
from packaging import version
from torch import distributed
from torch.distributed._tensor import DeviceMesh

from llmfoundry.layers_registry import ffns_with_megablocks
from llmfoundry.models.layers.ffn import resolve_ffn_hidden_size

__all__ = [
    'config_moe_args',
]


def create_process_group_ranks(ranks: tuple[int, ...]):
    """Creates a new distributed group.

    Used in create_set_process_group and create_mod_process_group methods below.

    This function is an alternative to `distributed.new_group(ranks)`.

    Args:
        ranks (tuple[int, ...]): Tuple of ranks of group members.

    Returns:
        A handle of distributed group that can be given to collective calls.
    """
    ranks_gather_list = [None for _ in range(distributed.get_world_size())]
    distributed.all_gather_object(ranks_gather_list, ranks)
    ranks_per_subgroup = list(set(ranks_gather_list))
    group, _ = distributed.distributed_c10d.new_subgroups_by_enumeration(
        ranks_per_subgroup,
    )
    return group


def create_set_process_group(k: int):
    """Creates a new distributed group using sets of k GPUs.

    For example, if you have 16 GPUs and input k=4, the resulting process groups
    will have ranks:
        process group 0 ranks: [ 0,  1,  2,  3]
        process group 1 ranks: [ 4,  5,  6,  7]
        process group 2 ranks: [ 8,  9, 10, 11]
        process group 3 ranks: [12, 13, 14, 15]

    Args:
        k (int): Number of GPUs to use in set size.

    Returns:
        A handle of distributed group that can be given to collective calls.
    """
    world_size = distributed.get_world_size()
    if world_size % k != 0:
        raise RuntimeError(f'{world_size=} must be divisible by {k=}.')
    start = distributed.get_rank() // k * k
    ranks = tuple(range(start, start + k))
    return create_process_group_ranks(ranks)


def get_megablocks_device_mesh(
    device_mesh_cfg: Optional[tuple[int, ...]],
    moe_world_size: int,
    world_size: int,
) -> DeviceMesh:
    """Helper function to get the device mesh for MegaBlocks MoE.

    Args:
        device_mesh_cfg (Optional[tuple[int, ...]]): The device mesh configuration specification.
        moe_world_size (int): The MoE world size.
        world_size (int): The world size.

    Raises:
        ValueError: If the device mesh configuration is not valid.

    Returns:
        The device mesh for MegaBlocks MoE.
    """
    from torch.distributed._tensor.device_mesh import init_device_mesh

    if device_mesh_cfg is None or len(device_mesh_cfg) == 1:
        if device_mesh_cfg is not None:
            world_size = device_mesh_cfg[0]
        sharding_group_dim = world_size // moe_world_size
        device_mesh = init_device_mesh(
            'cuda',
            (sharding_group_dim, moe_world_size),
            mesh_dim_names=('weight_parallel', 'expert_parallel'),
        )
    else:
        raise ValueError(f'{device_mesh_cfg=} must be length 1')

    return device_mesh


def config_megablocks_moe_args(
    ffn_config: dict,
    d_model: int,
    expansion_ratio: Union[int, float],
    n_layers: int,
    get_device_mesh: Callable,
) -> dict:
    """Configures `ffn_config` for MegaBlocks MoE.

    We prepare all necessary arguments for `megablocks.layers.arguments.Arguments` so that process
    groups can be initialized and shared across all blocks in the network.

    Args:
        ffn_config (dict): FFN configuration before the MegaBlocks MoE is configured.
        d_model (int): Hidden size of the network.
        expansion_ratio (Union[int, float]): Expansion ratio in FFN.
        n_layers (int): Number of blocks used in the network.
        get_device_mesh (Callable): Function to get the device mesh. Takes in the device mesh config and the MoE world size.

    Returns:
        ffn_config (dict): FFN configuration with MegaBlocks MoE configured.
    """
    try:
        import megablocks
    except:
        raise RuntimeError(
            'Requirements for MegaBlocks not installed; see install instructions in `README.md`.',
        )

    ffn_config.setdefault('fp16', False)
    ffn_config.setdefault('bf16', False)
    ffn_config['num_layers'] = n_layers

    ffn_type = ffn_config.pop('ffn_type')
    fc_type = ffn_config.pop('fc_type')
    ffn_act_fn = ffn_config.pop('ffn_act_fn', None)

    # Config for MegaBlocks MoE world size and device mesh
    world_size = 1  # default
    moe_world_size = ffn_config.pop('moe_world_size')
    device_mesh = None
    device_mesh_cfg = ffn_config.pop('device_mesh', None)
    if moe_world_size > 1:
        if version.parse(
            torch.__version__.split('.dev')[0],
        ) < version.parse('2.2.0'):  # type: ignore
            raise RuntimeError(
                'MoE world size > 1 is not supported in torch version {torch.__version__}<2.2.',
            )

        world_size = distributed.get_world_size()
        if world_size < moe_world_size or world_size % moe_world_size:
            raise ValueError(
                f'Invalid world size configuration: {world_size=} and {moe_world_size=}',
            )

        device_mesh = get_device_mesh(
            device_mesh_cfg=device_mesh_cfg,
            moe_world_size=moe_world_size,
            world_size=world_size,
        )

        ffn_config['moe_expert_model_parallelism'] = True
        ffn_config['expert_parallel_group'] = device_mesh[
            'expert_parallel'].get_group(0)  # type: ignore

    lbl_process_group = ffn_config.get('lbl_process_group', None)
    if lbl_process_group is not None:
        if lbl_process_group == 'expert_group':
            lbl_process_group = ffn_config['expert_parallel_group']
        elif lbl_process_group == 'global_group':
            lbl_process_group = distributed.group.WORLD
        elif isinstance(lbl_process_group, int):
            if lbl_process_group > 1:
                lbl_process_group = create_set_process_group(lbl_process_group)
            else:
                lbl_process_group = None
        elif not isinstance(lbl_process_group, distributed.ProcessGroup):
            raise ValueError(
                f'Unknown {lbl_process_group=}. Options are: none | a process group | ``expert_group`` | ``global_group`` | <GROUP_SIZE>.',
            )
        ffn_config['lbl_process_group'] = lbl_process_group

    ffn_hidden_size = resolve_ffn_hidden_size(d_model, expansion_ratio)
    ffn_config.setdefault('ffn_hidden_size', ffn_hidden_size)

    args_to_keep_in_ffn_config = inspect.signature(
        megablocks.layers.arguments.Arguments,
    ).parameters

    ffn_config = {
        k: v for k, v in ffn_config.items() if k in args_to_keep_in_ffn_config
    }

    args = megablocks.layers.arguments.Arguments(
        hidden_size=d_model,
        **ffn_config,
    )
    ffn_config['args'] = args
    ffn_config['device_mesh'] = device_mesh
    ffn_config['moe_world_size'] = moe_world_size
    ffn_config['ffn_type'] = ffn_type
    ffn_config['fc_type'] = fc_type
    ffn_config['ffn_act_fn'] = ffn_act_fn

    return ffn_config


def config_moe_args(
    ffn_config: dict,
    d_model: int,
    expansion_ratio: Union[int, float],
    n_layers: int,
) -> dict:
    """Configures `ffn_config` for MoE.

    Args:
        ffn_config (dict): FFN configuration before the MoE is configured.
        d_model (int): Hidden size of the network.
        expansion_ratio (int, float): Expansion ratio in FFN.
        n_layers (int): Number of blocks used in the network.

    Returns:
        ffn_config (dict): FFN configuration with MoE configured.
    """
    if ffn_config['ffn_type'] in ffns_with_megablocks:
        return config_megablocks_moe_args(
            ffn_config=ffn_config,
            d_model=d_model,
            expansion_ratio=expansion_ratio,
            n_layers=n_layers,
            get_device_mesh=get_megablocks_device_mesh,
        )
    else:
        raise ValueError(f'Invalid ffn_type ({ffn_config["ffn_type"]}).')
