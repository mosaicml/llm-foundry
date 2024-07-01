# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Type

import torch

from llmfoundry.utils.registry_utils import create_registry

_norms_description = (
    """The norms registry is used to register classes that implement normalization layers.

    One example of this is torch.nn.LayerNorm. See norm.py for examples.

    Args:
        normalized_shape Union[int, List[int], torch.Size]: The shape of the input tensor.
        device: Optional[torch.device]: The device to use for the normalization layer.

    Returns:
        torch.nn.Module: The normalization layer.
    """
)
norms = create_registry(
    'llmfoundry',
    'norms',
    generic_type=Type[torch.nn.Module],
    entry_points=True,
    description=_norms_description,
)

_fcs_description = (
    """The fcs registry is used to register classes that implement fully connected layers (i.e. torch.nn.Linear).

    See fc.py for examples.

    Args:
        in_features: int: The number of input features.
        out_features: int: The number of output features.
        kwargs: Dict[str, Any]: Additional keyword arguments to pass to the layer.

    Returns:
        torch.nn.Module: The fully connected layer.
    """
)
fcs = create_registry(
    'llmfoundry',
    'fcs',
    generic_type=Type[torch.nn.Module],
    entry_points=True,
    description=_fcs_description,
)

_ffns_description = (
    """The ffns registry is used to register functions that build FFN layers.

    These layers are generally composed of fc layers and activation functions.
    One example is MPTMLP. See ffn.py for examples.

    Args:
        d_model: int: The size of the input and output tensors.
        expansion_ratio: float: The expansion ratio for the hidden layer.
        device: Optional[str]: The device to use for the layer.
        bias: bool: Whether or not to include a bias term.
        kwargs: Dict[str, Any]: Additional keyword arguments to pass to the layer.

    Returns:
        torch.nn.Module: The FFN layer.
    """
)
ffns = create_registry(
    'llmfoundry',
    'ffns',
    generic_type=Callable,
    entry_points=True,
    description=_ffns_description,
)

_ffns_with_norm_description = (
    """The ffns_with_norm registry is used to register functions that build FFN layers with normalization.

    The resulting layer will have ._has_norm set on it.
    One example is te.LayerNormMLP. See ffn.py for examples.

    Args:
        d_model: int: The size of the input and output tensors.
        expansion_ratio: float: The expansion ratio for the hidden layer.
        device: Optional[str]: The device to use for the layer.
        bias: bool: Whether or not to include a bias term.
        kwargs: Dict[str, Any]: Additional keyword arguments to pass to the layer.

    Returns:
        torch.nn.Module: The FFN layer.
    """
)
ffns_with_norm = create_registry(
    'llmfoundry',
    'ffns_with_norm',
    generic_type=Callable,
    entry_points=True,
    description=_ffns_with_norm_description,
)

_ffns_with_megablocks_description = (
    'The ffns_with_megablocks registry is used to register functions that build ffn layers using MegaBlocks.'
    + 'See ffn.py for examples.'
)
_ffns_with_megablocks_description = (
    """The ffns_with_megablocks registry is used to register functions that build FFN layers using MegaBlocks.

    The resulting layer will have ._uses_megablocks set on it.
    One example is megablocks.layers.dmoe.dMoE. See ffn.py for examples.

    Returns:
        torch.nn.Module: The FFN layer.
    """
)
ffns_with_megablocks = create_registry(
    'llmfoundry',
    'ffns_with_megablocks',
    generic_type=Callable,
    entry_points=True,
    description=_ffns_with_megablocks_description,
)

_attention_classes_description = (
    """The attention_classes registry is used to register classes that implement attention layers.

    The kwargs are passed directly to the constructor of the class.
    One example is GroupedQueryAttention. See attention.py for examples.

    Args:
        kwargs: Dict[str, Any]: Additional keyword arguments to pass to the layer.

    Returns:
        torch.nn.Module: The attention layer.
    """
)
attention_classes = create_registry(
    'llmfoundry',
    'attention_classes',
    generic_type=Type[torch.nn.Module],
    entry_points=True,
    description=_attention_classes_description,
)

_attention_implementations_description = (
    """The attention_implementations registry is used to register functions that implement the attention operation.

    One example is 'flash'. See attention.py for examples.

    Args:
        query (torch.Tensor): The query tensor.
        key (torch.Tensor): The key tensor.
        value (torch.Tensor): The value tensor.
        n_heads (int): The number of attention heads.
        kv_n_heads (int): The number of attention heads for the key and value tensors.
        past_key_value (Optional[tuple[torch.Tensor, torch.Tensor]]): The past key and value tensors.
        softmax_scale (Optional[float]) = None
        attn_bias (Optional[torch.Tensor]) = None
        is_causal (bool) = False
        dropout_p (float) = 0.0
        training (bool) = True
        needs_weights (bool) = False
        kwargs: Dict[str, Any]: Additional keyword arguments the implementation accepts.

    Returns:
        tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor, torch.Tensor]]]:
            The output tensor, the attention weights, and the past key and value tensors.
    """
)
attention_implementations = create_registry(
    'llmfoundry',
    'attention_implementations',
    generic_type=Callable,
    entry_points=True,
    description=_attention_implementations_description,
)

_param_init_fns_description = (
    """The param_init_fns registry is used to register functions that initialize parameters.

    These functions should take in a torch.nn.Module, additional kwargs, and initialize the parameters of the module.
    Generally they can call generic_param_init_fn_ with an appropriate partial function. See param_init_fns.py for examples.

    Note: These functions should take in arbitrary kwargs, and discard any they don't need.

    Args:
        module: torch.nn.Module: The module to initialize.
        kwargs: Dict[str, Any]: Additional keyword arguments to use for initialization.
    """
)
param_init_fns = create_registry(
    'llmfoundry',
    'param_init_fns',
    generic_type=Callable[..., None],
    entry_points=True,
    description=_param_init_fns_description,
)

_module_init_fns_description = (
    """The module_init_fns registry is used to register functions that initialize specific modules.

    These functions should return True if they initialize the module, and False otherwise.
    This allows them to be called without knowing their contents. They should take in the module and additional kwargs.
    If multiple functions can initialize the module, the one that is registered first will be used, so it is recommended to
    override an existing function if you want to change existing initialization behavior, and add new functions if you have new
    layer types. See param_init_fns.py for details.

    Args:
        module: torch.nn.Module: The module to initialize.
        kwargs: Dict[str, Any]: Additional keyword arguments to use for initialization.

    Returns:
        bool: Whether or not the module was initialized.
    """
)
module_init_fns = create_registry(
    'llmfoundry',
    'module_init_fns',
    generic_type=Callable[..., bool],
    entry_points=True,
    description=_module_init_fns_description,
)

__all__ = [
    'norms',
    'param_init_fns',
    'module_init_fns',
    'ffns',
    'ffns_with_norm',
    'ffns_with_megablocks',
    'attention_classes',
    'attention_implementations',
    'fcs',
]
