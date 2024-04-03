# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""MPT Blocks used for the MPT Model."""

import logging
from copy import deepcopy
from functools import partial
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn

from llmfoundry.layers_registry import ffns, ffns_with_norm
from llmfoundry.models.layers.fc import FC_CLASS_REGISTRY

try:
    import transformer_engine.pytorch as te
except:
    te = None

log = logging.getLogger(__name__)

_FFN_ACT_FN_DEFAULT = {
    'name': 'gelu',
    'approximate': 'none',
}


def resolve_ffn_act_fn(
    config: Optional[dict] = None,) -> Callable[[torch.Tensor], torch.Tensor]:
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
        raise ValueError(f'Unrecognised activation function name ({name}).')
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
            f'`expansion_ratio` (={expansion_ratio}) ignored when `ffn_hidden_size` (={ffn_hidden_size}) is specified.'
        )
    else:
        ffn_hidden_size = int(d_model * expansion_ratio)
        if ffn_hidden_size != d_model * expansion_ratio:
            raise ValueError(
                f'`d_model * expansion_ratio` must be an integer ({d_model=}; {expansion_ratio=}; {d_model * expansion_ratio=}).'
            )
    return ffn_hidden_size


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
        ffn_hidden_size = resolve_ffn_hidden_size(d_model, expansion_ratio,
                                                  ffn_hidden_size)
        self.fc_kwargs: dict[str, Any] = {
            'bias': bias,
        }

        self.fc_kwargs['device'] = device

        self.up_proj = FC_CLASS_REGISTRY[fc_type](
            d_model,
            ffn_hidden_size,
            **self.fc_kwargs,
        )
        self.act = act_fn
        self.down_proj = FC_CLASS_REGISTRY[fc_type](
            ffn_hidden_size,
            d_model,
            **self.fc_kwargs,
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
        self.gate_proj = FC_CLASS_REGISTRY[fc_type](
            d_model,
            self.up_proj.out_features,
            **self.fc_kwargs,
        )

    @torch.compile
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
    ffn_hidden_size = resolve_ffn_hidden_size(d_model, expansion_ratio,
                                              ffn_hidden_size)
    if ffn_act_fn is not None:
        raise ValueError(
            f'Transformer Engine block does not support custom activation functions.'
        )
    return te.LayerNormMLP(
        hidden_size=d_model,
        ffn_hidden_size=ffn_hidden_size,
        bias=bias,
        **kwargs,
    )


ffns.register('mptglu', func=build_mptglu)
ffns.register('mptmlp', func=build_mptmlp)

if te is not None:
    ffns_with_norm.register('te_ln_mlp', func=build_te_ln_mlp)
