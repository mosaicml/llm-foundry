# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""MPT Blocks used for the MPT Model."""

import logging
from copy import deepcopy
from functools import partial
from typing import Any, Callable, List, Optional, Union

import torch
import torch.nn as nn
from torch.distributed import ProcessGroup
from torch.distributed._tensor import DeviceMesh, DTensor, Placement, Shard

from llmfoundry.layers_registry import (
    ffns,
    ffns_with_megablocks,
    ffns_with_norm,
)
from llmfoundry.models.layers.dmoe import dMoE
from llmfoundry.models.layers.layer_builders import build_fc

try:
    import transformer_engine.pytorch as te
    is_te_imported = True
except ModuleNotFoundError:
    is_te_imported = False

try:
    import megablocks
    is_megablocks_imported = True
except ModuleNotFoundError:
    is_megablocks_imported = False

log = logging.getLogger(__name__)

__all__ = [
    'MPTMLP',
    'MPTGLU',
    'build_mptglu',
    'build_mptmlp',
    'build_te_ln_mlp',
    'build_torch_dmoe',
    'build_mb_moe',
    'build_mb_dmoe',
]

_FFN_ACT_FN_DEFAULT = {
    'name': 'gelu',
    'approximate': 'none',
}


def resolve_ffn_act_fn(
    config: Optional[dict] = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Resolve the activation function for the feed-forward network.

    Args:
        config (Optional[dict]): The configuration dictionary for the activation function.
            The dict config must specify the 'name' of a torch.nn.functional activation
            function. All of other key values pairs are bound to the function as a partial.

    Returns:
        Callable[[torch.Tensor], torch.Tensor]: The activation function.
    """
    if config is None:
        config = _FFN_ACT_FN_DEFAULT
    config = deepcopy(config)
    name = config.pop('name')
    if not hasattr(torch.nn.functional, name):
        raise ValueError(f'Unrecognized activation function name ({name}).')
    act = getattr(torch.nn.functional, name)
    return partial(act, **config)


_DEFAULT_ACT_FN = resolve_ffn_act_fn(_FFN_ACT_FN_DEFAULT)


def resolve_ffn_hidden_size(
    d_model: int,
    expansion_ratio: Union[int, float],
    ffn_hidden_size: Optional[int] = None,
) -> int:
    """Resolve the hidden size of the feed-forward network.

    Args:
        d_model (int): The dimension of the input and output of the feed-forward network.
        expansion_ratio (Union[int, float]): The expansion ratio of the feed-forward network.
        ffn_hidden_size (Optional[int]): The hidden size of the feed-forward network.

    Returns:
        int: The hidden size of the feed-forward network.
    """
    if ffn_hidden_size is not None:
        log.info(
            f'`expansion_ratio` (={expansion_ratio}) ignored when `ffn_hidden_size` (={ffn_hidden_size}) is specified.',
        )
    else:
        ffn_hidden_size = int(d_model * expansion_ratio)
        if ffn_hidden_size != d_model * expansion_ratio:
            raise ValueError(
                f'`d_model * expansion_ratio` must be an integer ({d_model=}; {expansion_ratio=}; {d_model * expansion_ratio=}).',
            )
    return ffn_hidden_size


def dtensorify_param(
    param: nn.Parameter,
    mesh: DeviceMesh,
    placements: List[Placement],
):
    """Construct a DTensor from an already sharded local parameter."""
    param_dtensor = DTensor.from_local(
        param.data,
        device_mesh=mesh,
        placements=placements,
        run_check=False,
    )
    return nn.Parameter(param_dtensor)


class MPTMLP(nn.Module):

    def __init__(
        self,
        d_model: int,
        expansion_ratio: Union[int, float],
        fc_type: str = 'torch',
        ffn_hidden_size: Optional[int] = None,
        act_fn: Callable[[torch.Tensor], torch.Tensor] = _DEFAULT_ACT_FN,
        device: Optional[str] = None,
        bias: bool = True,
    ):
        super().__init__()
        ffn_hidden_size = resolve_ffn_hidden_size(
            d_model,
            expansion_ratio,
            ffn_hidden_size,
        )
        self.fc_kwargs: dict[str, Any] = {
            'bias': bias,
        }

        self.fc_kwargs['device'] = device

        self.up_proj = build_fc(
            name=fc_type,
            in_features=d_model,
            out_features=ffn_hidden_size,
            fc_kwargs=self.fc_kwargs,
        )
        self.act = act_fn
        self.down_proj = build_fc(
            name=fc_type,
            in_features=ffn_hidden_size,
            out_features=d_model,
            fc_kwargs=self.fc_kwargs,
        )
        self.down_proj._is_residual = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.up_proj(x)))


class MPTGLU(MPTMLP):

    def __init__(
        self,
        d_model: int,
        expansion_ratio: Union[int, float],
        fc_type: str = 'torch',
        ffn_hidden_size: Optional[int] = None,
        act_fn: Callable[[torch.Tensor], torch.Tensor] = _DEFAULT_ACT_FN,
        device: Optional[str] = None,
        bias: bool = True,
    ):
        super().__init__(
            d_model=d_model,
            expansion_ratio=expansion_ratio,
            fc_type=fc_type,
            ffn_hidden_size=ffn_hidden_size,
            act_fn=act_fn,
            device=device,
            bias=bias,
        )
        self.gate_proj = build_fc(
            name=fc_type,
            in_features=d_model,
            out_features=self.up_proj.out_features,
            fc_kwargs=self.fc_kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


def build_mptglu(
    d_model: int,
    expansion_ratio: Union[int, float],
    fc_type: str = 'torch',
    ffn_hidden_size: Optional[int] = None,
    ffn_act_fn: Optional[dict] = None,
    device: Optional[str] = None,
    bias: bool = True,
) -> nn.Module:
    return MPTGLU(
        d_model=d_model,
        expansion_ratio=expansion_ratio,
        fc_type=fc_type,
        act_fn=resolve_ffn_act_fn(ffn_act_fn),
        ffn_hidden_size=ffn_hidden_size,
        device=device,
        bias=bias,
    )


def build_mptmlp(
    d_model: int,
    expansion_ratio: Union[int, float],
    fc_type: str = 'torch',
    ffn_hidden_size: Optional[int] = None,
    ffn_act_fn: Optional[dict] = None,
    device: Optional[str] = None,
    bias: bool = True,
) -> nn.Module:
    return MPTMLP(
        d_model=d_model,
        expansion_ratio=expansion_ratio,
        fc_type=fc_type,
        act_fn=resolve_ffn_act_fn(ffn_act_fn),
        ffn_hidden_size=ffn_hidden_size,
        device=device,
        bias=bias,
    )


def build_te_ln_mlp(
    d_model: int,
    expansion_ratio: Union[int, float],
    fc_type: str = 'torch',
    ffn_hidden_size: Optional[int] = None,
    ffn_act_fn: Optional[dict] = None,
    device: Optional[str] = None,
    bias: bool = True,
    **kwargs: Any,
) -> nn.Module:
    assert te is not None
    ffn_hidden_size = resolve_ffn_hidden_size(
        d_model,
        expansion_ratio,
        ffn_hidden_size,
    )
    if ffn_act_fn is not None:
        raise ValueError(
            f'Transformer Engine block does not support custom activation functions.',
        )
    return te.LayerNormMLP(
        hidden_size=d_model,
        ffn_hidden_size=ffn_hidden_size,
        bias=bias,
        **kwargs,
    )


def build_torch_dmoe(
    d_model: int,
    expansion_ratio: Union[int, float],
    ffn_hidden_size: Optional[int] = None,
    ffn_act_fn: Optional[dict] = None,
    device: Optional[str] = None,
    bias: bool = True,
    **kwargs: Any,
) -> nn.Module:
    moe_num_experts = kwargs.pop('moe_num_experts')
    moe_top_k = kwargs.pop('moe_top_k')
    mlp_type = kwargs.pop('mlp_type')
    moe_jitter_eps = kwargs.pop('moe_jitter_eps')
    moe_normalize_expert_weights = kwargs.pop('moe_normalize_expert_weights')
    uniform_expert_assignment = kwargs.pop('uniform_expert_assignment')

    fc_type = kwargs.pop('fc_type', 'torch')
    del fc_type  # Unused

    if len(kwargs) > 0:
        raise ValueError(f'Invalid arguments to torch dmoe: {kwargs}.')

    return dMoE(
        hidden_size=d_model,
        ffn_hidden_size=resolve_ffn_hidden_size(
            d_model,
            expansion_ratio,
            ffn_hidden_size,
        ),
        moe_num_experts=moe_num_experts,
        moe_top_k=moe_top_k,
        mlp_type=mlp_type,
        bias=bias,
        moe_jitter_eps=moe_jitter_eps,
        activation_fn=resolve_ffn_act_fn(ffn_act_fn),
        moe_normalize_expert_weights=moe_normalize_expert_weights,
        uniform_expert_assignment=uniform_expert_assignment,
        device=torch.device(device) if device is not None else None,
    )


def _mb_setup_args(
    d_model: int,
    expansion_ratio: Union[int, float],
    ffn_hidden_size: Optional[int],
    ffn_act_fn: Optional[dict],
    device: Optional[str],
    bias: bool,
    kwargs: dict[str, Any],
) -> tuple['megablocks.layers.arguments.Arguments', int, ProcessGroup]:
    if megablocks is None:
        raise RuntimeError(
            'Requirements for megablocks not installed; see install instructions in `README.md`.',
        )
    args = kwargs['args']
    args.bias = bias
    args.hidden_size = d_model
    args.device = device

    ffn_hidden_size = resolve_ffn_hidden_size(
        d_model,
        expansion_ratio,
        ffn_hidden_size,
    )
    args.ffn_hidden_size = ffn_hidden_size

    if ffn_act_fn is not None:
        args.activation_fn = resolve_ffn_act_fn(ffn_act_fn)

    moe_world_size = 1
    expert_parallel_group = args.expert_parallel_group
    if expert_parallel_group is not None:
        moe_world_size = expert_parallel_group.size()
    if kwargs.get('moe_world_size') != moe_world_size:
        raise RuntimeError(
            f'MoE expert_parallel_group configured with incorrect world size.',
        )

    return args, moe_world_size, expert_parallel_group


def _patch_ffn_mb(
    ffn: nn.Module,
    moe_world_size: int,
    expert_parallel_group: ProcessGroup,
    device_mesh: DeviceMesh,
    args: 'megablocks.layers.arguments.Arguments',
):
    # Attach args to MLP directly for use in param_init_fn
    ffn.experts.mlp.hidden_size = args.ffn_hidden_size
    ffn.experts.mlp.expert_parallel_group = expert_parallel_group
    ffn.experts.mlp.weight_parallel_group = args.weight_parallel_group

    if moe_world_size > 1:
        expert_mesh = device_mesh['expert_parallel']
        expert_placements: List[Placement] = [Shard(0)]
        # Register in two loops as you cannot overwrite parameters while iterating over named_parameters()
        dtensorified_params = [(
            name,
            dtensorify_param(
                param=parameter,
                mesh=expert_mesh,
                placements=expert_placements,
            ),
        ) for name, parameter in ffn.experts.mlp.named_parameters()]
        for name, dtensorified_param in dtensorified_params:
            ffn.experts.mlp.register_parameter(name, dtensorified_param)

        if device_mesh.mesh.ndim == 2:
            submesh = device_mesh['weight_parallel']
        elif device_mesh.mesh.ndim == 3:
            raise RuntimeError(f'HSDP + MoE is not supported.')
        else:
            raise ValueError(f'{device_mesh.mesh.ndim=} not supported for MoE.')

        ffn.experts._fsdp_kwargs_dict = {
            'device_mesh': submesh,
        }


def build_mb_moe(
    d_model: int,
    expansion_ratio: Union[int, float],
    ffn_hidden_size: Optional[int] = None,
    ffn_act_fn: Optional[dict] = None,
    device: Optional[str] = None,
    bias: bool = True,
    **kwargs: Any,
) -> nn.Module:
    if not is_megablocks_imported:
        raise RuntimeError(
            'Requirements for megablocks not installed; see install instructions in `README.md`.',
        )

    args, moe_world_size, expert_parallel_group = _mb_setup_args(
        d_model=d_model,
        expansion_ratio=expansion_ratio,
        ffn_hidden_size=ffn_hidden_size,
        ffn_act_fn=ffn_act_fn,
        device=device,
        bias=bias,
        kwargs=kwargs,
    )

    ffn = megablocks.layers.moe.MoE(args)

    # Fused initialization setup
    # For param_init_fn, enables shape based init of stacked layers
    ffn.experts.mlp._stack_dim = 0

    _patch_ffn_mb(
        ffn=ffn,
        moe_world_size=moe_world_size,
        expert_parallel_group=expert_parallel_group,
        device_mesh=kwargs['device_mesh'],
        args=args,
    )

    return ffn


def build_mb_dmoe(
    d_model: int,
    expansion_ratio: Union[int, float],
    ffn_hidden_size: Optional[int] = None,
    ffn_act_fn: Optional[dict] = None,
    device: Optional[str] = None,
    bias: bool = True,
    **kwargs: Any,
) -> nn.Module:
    if not is_megablocks_imported:
        raise RuntimeError(
            'Requirements for megablocks not installed; see install instructions in `README.md`.',
        )

    args, moe_world_size, expert_parallel_group = _mb_setup_args(
        d_model=d_model,
        expansion_ratio=expansion_ratio,
        ffn_hidden_size=ffn_hidden_size,
        ffn_act_fn=ffn_act_fn,
        device=device,
        bias=bias,
        kwargs=kwargs,
    )

    ffn = megablocks.layers.dmoe.dMoE(args)

    # Fused initialization setup
    # For param_init_fn, enables shape based init of fused layers
    n_exp = min(1, args.moe_num_experts // moe_world_size)
    ffn.experts.mlp._fused = (
        0,
        [(n + 1) * args.ffn_hidden_size for n in range(n_exp - 1)],
    )

    _patch_ffn_mb(
        ffn=ffn,
        moe_world_size=moe_world_size,
        expert_parallel_group=expert_parallel_group,
        device_mesh=kwargs['device_mesh'],
        args=args,
    )

    return ffn


ffns.register('mptglu', func=build_mptglu)
ffns.register('mptmlp', func=build_mptmlp)
ffns.register('torch_dmoe', func=build_torch_dmoe)

if is_te_imported:
    ffns_with_norm.register('te_ln_mlp', func=build_te_ln_mlp)

if is_megablocks_imported:
    ffns_with_megablocks.register('mb_moe', func=build_mb_moe)
    ffns_with_megablocks.register('mb_dmoe', func=build_mb_dmoe)
