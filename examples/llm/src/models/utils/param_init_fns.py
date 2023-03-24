# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0
import math
import warnings
from collections.abc import Sequence
from functools import partial
from typing import Optional, Tuple, Union

import torch
from torch import nn


def torch_default_param_init_fn_(
    module: nn.Module,
    verbose: int = 0,
    **kwargs,
):
    del kwargs  # unused, just to capture any extra args from the config
    if verbose > 1:
        warnings.warn(
            f"Initializing network using module's reset_parameters attribute")

    if hasattr(module, 'reset_parameters'):
        module.reset_parameters()  # type: ignore


def fused_init_helper_(module: nn.Module, init_fn_):
    # parameter initialization is often based on the parameters shape.
    # If a layer is fused, initialization should be based on the shapes
    # of the original tensor instead of the shape of the fused tensor.
    # Layers which are fused should have the _fused attibute defined.
    # The first element of _fused is the dimension along which the tensor is fused.
    # This is followed by an iterable of split indices."

    _fused = getattr(module, '_fused', None)

    if _fused is None:
        raise RuntimeError(f'Internal logic error')

    dim, splits = _fused
    splits = (0, *splits, module.weight.size(dim))  # type: ignore
    for s, e in zip(splits[:-1], splits[1:]):
        slice_indices = [slice(None)] * module.weight.ndim  # type: ignore
        slice_indices[dim] = slice(s, e)
        init_fn_(module.weight[slice_indices])  # type: ignore


def generic_param_init_fn_(
    module: nn.Module,
    init_fn_,
    n_layers: int,
    d_model: Optional[int] = None,
    init_div_is_residual: Union[int, float, str, bool] = True,
    emb_init_std: Optional[float] = None,
    emb_init_uniform_lim: Optional[Union[Tuple[float, float], float]] = None,
    verbose: int = 0,
    **kwargs,
):
    del kwargs  # unused, just to capture any extra args from the config
    if verbose > 1:
        warnings.warn(
            f'If model has bias parameters they are initialized to 0.')

    # enable user to divide _is_residual weights by
    # a value which defaults to math.sqrt(2 * cfg.n_layers)
    init_div_is_residual = init_div_is_residual

    if init_div_is_residual is False:
        # not used, for pyright
        div_is_residual = 1.0
    elif init_div_is_residual is True:
        div_is_residual = math.sqrt(2 * n_layers)
    elif isinstance(init_div_is_residual, float) or isinstance(
            init_div_is_residual, int):
        div_is_residual = init_div_is_residual
    elif isinstance(init_div_is_residual,
                    str) and init_div_is_residual.isnumeric():
        # do not trust YAML parsing to always convert numbers to numbers
        div_is_residual = float(init_div_is_residual)
    else:
        # not used, for pyright
        div_is_residual = 1.0
        raise ValueError(
            f'Expected init_div_is_residual to be boolean or numeric, got {init_div_is_residual}'
        )

    if init_div_is_residual is not False:
        if verbose > 1:
            warnings.warn(
                f'Initializing _is_residual layers then dividing them by {div_is_residual}.' +\
                f'set `init_div_is_residual: false` in model config to disable this.'
            )

    if isinstance(module, nn.Linear):
        # Linear
        if hasattr(module, '_fused'):
            fused_init_helper_(module, init_fn_)
        else:
            init_fn_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)

        if init_div_is_residual is not False and getattr(
                module, '_is_residual', False):
            with torch.no_grad():
                module.weight.div_(div_is_residual)

    elif isinstance(module, nn.Embedding):
        # Embedding
        if emb_init_std is not None:
            std = emb_init_std
            if std == 0:
                warnings.warn(f'Embedding layer initialized to 0.')
            emb_init_fn_ = partial(torch.nn.init.normal_, mean=0.0, std=std)
            if verbose > 1:
                warnings.warn(
                    f'Embedding layer initialized using normal distribution with mean=0 and {std=}.'
                )
        elif emb_init_uniform_lim is not None:
            lim = emb_init_uniform_lim
            if isinstance(lim, Sequence):
                if len(lim) > 2:
                    raise ValueError(
                        f'Uniform init requires a min and a max limit. User input: {lim}.'
                    )
                if lim[0] == lim[1]:
                    warnings.warn(f'Embedding layer initialized to {lim[0]}.')
            else:
                if lim == 0:
                    warnings.warn(f'Embedding layer initialized to 0.')
                lim = [-lim, lim]
            a, b = lim
            emb_init_fn_ = partial(torch.nn.init.uniform_, a=a, b=b)
            if verbose > 1:
                warnings.warn(
                    f'Embedding layer initialized using uniform distribution in range {lim}.'
                )
        else:
            emb_init_fn_ = init_fn_

        emb_init_fn_(module.weight)

    elif isinstance(module, nn.LayerNorm):
        # LayerNorm
        if verbose > 1:
            warnings.warn(
                f'LayerNorm gamma weights are set to 1. If the layer has a bias it is initialized to 0.'
            )
        torch.nn.init.ones_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)

    elif isinstance(module, nn.MultiheadAttention):
        # torch's MultiheadAttention
        if module._qkv_same_embed_dim:
            assert module.in_proj_weight is not None
            assert module.q_proj_weight is None and module.k_proj_weight is None and module.v_proj_weight is None
            assert d_model is not None
            # in_proj_weight is actually 3 layers and should be split up for width based init
            _d = d_model
            splits = (0, _d, 2 * _d, 3 * _d)
            for s, e in zip(splits[:-1], splits[1:]):
                init_fn_(module.in_proj_weight[s:e])
        else:
            assert module.q_proj_weight is not None and module.k_proj_weight is not None and module.v_proj_weight is not None
            assert module.in_proj_weight is None
            init_fn_(module.q_proj_weight)
            init_fn_(module.k_proj_weight)
            init_fn_(module.v_proj_weight)

        # bias
        if module.in_proj_bias is not None:
            torch.nn.init.zeros_(module.in_proj_bias)
        if module.bias_k is not None:
            torch.nn.init.zeros_(module.bias_k)
        if module.bias_v is not None:
            torch.nn.init.zeros_(module.bias_v)

        # out proj
        init_fn_(module.out_proj.weight)
        if init_div_is_residual is not False and getattr(
                module.out_proj, '_is_residual', False):
            with torch.no_grad():
                module.out_proj.weight.div_(div_is_residual)
        if module.out_proj.bias is not None:
            torch.nn.init.zeros_(module.out_proj.bias)

    else:
        for _ in module.parameters(recurse=False):
            # raise error if uninitialized module has any parameters
            raise NotImplementedError(
                f'{module.__class__.__name__} parameters are not initialized by param_init_fn.'
            )


def _normal_init_(std, mean=0.0):
    return partial(torch.nn.init.normal_, mean=mean, std=std)


def _normal_param_init_fn_(
    module: nn.Module,
    std: float,
    n_layers: int,
    d_model: Optional[int] = None,
    init_div_is_residual: Union[int, float, str, bool] = True,
    emb_init_std: Optional[float] = None,
    emb_init_uniform_lim: Optional[Union[Tuple[float, float], float]] = None,
    verbose: int = 0,
    **kwargs,
):
    del kwargs  # unused, just to capture any extra args from the config
    init_fn_ = _normal_init_(std=std)

    if verbose > 1:
        warnings.warn(
            f'Using torch.nn.init.normal_ init fn mean=0.0, std={std}')

    generic_param_init_fn_(
        module=module,
        init_fn_=init_fn_,
        d_model=d_model,
        n_layers=n_layers,
        init_div_is_residual=init_div_is_residual,
        emb_init_std=emb_init_std,
        emb_init_uniform_lim=emb_init_uniform_lim,
        verbose=verbose,
    )


def baseline_param_init_fn_(
    module: nn.Module,
    init_std: float,
    n_layers: int,
    d_model: Optional[int] = None,
    init_div_is_residual: Union[int, float, str, bool] = True,
    emb_init_std: Optional[float] = None,
    emb_init_uniform_lim: Optional[Union[Tuple[float, float], float]] = None,
    verbose: int = 0,
    **kwargs,
):
    del kwargs  # unused, just to capture any extra args from the config
    if init_std is None:
        raise ValueError(
            'You must set model.init_std to a float value to use the default initialization scheme.'
        )
    _normal_param_init_fn_(
        module=module,
        std=init_std,
        d_model=d_model,
        n_layers=n_layers,
        init_div_is_residual=init_div_is_residual,
        emb_init_std=emb_init_std,
        emb_init_uniform_lim=emb_init_uniform_lim,
        verbose=verbose,
    )


def small_param_init_fn_(
    module: nn.Module,
    n_layers: int,
    d_model: int,
    init_div_is_residual: Union[int, float, str, bool] = True,
    emb_init_std: Optional[float] = None,
    emb_init_uniform_lim: Optional[Union[Tuple[float, float], float]] = None,
    verbose: int = 0,
    **kwargs,
):
    del kwargs  # unused, just to capture any extra args from the config
    # very close to kaiming normal
    # from Transformers without Tears (2019) - Nguyen & Salazar
    std = math.sqrt(2 / (5 * d_model))
    _normal_param_init_fn_(
        module=module,
        std=std,
        d_model=d_model,
        n_layers=n_layers,
        init_div_is_residual=init_div_is_residual,
        emb_init_std=emb_init_std,
        emb_init_uniform_lim=emb_init_uniform_lim,
        verbose=verbose,
    )


def neox_param_init_fn_(
    module: nn.Module,
    n_layers: int,
    d_model: int,
    emb_init_std: Optional[float] = None,
    emb_init_uniform_lim: Optional[Union[Tuple[float, float], float]] = None,
    verbose: int = 0,
    **kwargs,
):
    """From section 2.3.1 of GPT-NeoX-20B:

    An Open-Source AutoregressiveLanguage Model — Black et. al. (2022)
    see https://github.com/EleutherAI/gpt-neox/blob/9610391ab319403cef079b438edd016a2443af54/megatron/model/init_functions.py#L151
    and https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/transformer.py
    """
    del kwargs  # unused, just to capture any extra args from the config
    residual_div = n_layers / math.sqrt(10)  # small std / wang std

    if verbose > 1:
        warnings.warn(f'setting init_div_is_residual to {residual_div}')

    small_param_init_fn_(
        module=module,
        d_model=d_model,
        n_layers=n_layers,
        init_div_is_residual=residual_div,
        emb_init_std=emb_init_std,
        emb_init_uniform_lim=emb_init_uniform_lim,
        verbose=verbose,
    )


def kaiming_uniform_param_init_fn_(
    module: nn.Module,
    n_layers: int,
    d_model: Optional[int] = None,
    init_div_is_residual: Union[int, float, str, bool] = True,
    emb_init_std: Optional[float] = None,
    emb_init_uniform_lim: Optional[Union[Tuple[float, float], float]] = None,
    init_gain: float = 0,
    fan_mode: str = 'fan_in',
    init_nonlinearity: str = 'leaky_relu',
    verbose: int = 0,
    **kwargs,
):
    del kwargs  # unused, just to capture any extra args from the config

    if verbose > 1:
        warnings.warn(
            f'Using nn.init.kaiming_uniform_ init fn with parameters: ' +\
            f'a={init_gain}, mode={fan_mode}, nonlinearity={init_nonlinearity}'
        )

    kaiming_uniform_ = partial(nn.init.kaiming_uniform_,
                               a=init_gain,
                               mode=fan_mode,
                               nonlinearity=init_nonlinearity)

    generic_param_init_fn_(
        module=module,
        init_fn_=kaiming_uniform_,
        d_model=d_model,
        n_layers=n_layers,
        init_div_is_residual=init_div_is_residual,
        emb_init_std=emb_init_std,
        emb_init_uniform_lim=emb_init_uniform_lim,
        verbose=verbose,
    )


def kaiming_normal_param_init_fn_(
    module: nn.Module,
    n_layers: int,
    d_model: Optional[int] = None,
    init_div_is_residual: Union[int, float, str, bool] = True,
    emb_init_std: Optional[float] = None,
    emb_init_uniform_lim: Optional[Union[Tuple[float, float], float]] = None,
    init_gain: float = 0,
    fan_mode: str = 'fan_in',
    init_nonlinearity: str = 'leaky_relu',
    verbose: int = 0,
    **kwargs,
):
    del kwargs  # unused, just to capture any extra args from the config

    if verbose > 1:
        warnings.warn(
            f'Using nn.init.kaiming_normal_ init fn with parameters: ' +\
            f'a={init_gain}, mode={fan_mode}, nonlinearity={init_nonlinearity}'
        )

    kaiming_normal_ = partial(torch.nn.init.kaiming_normal_,
                              a=init_gain,
                              mode=fan_mode,
                              nonlinearity=init_nonlinearity)

    generic_param_init_fn_(
        module=module,
        init_fn_=kaiming_normal_,
        d_model=d_model,
        n_layers=n_layers,
        init_div_is_residual=init_div_is_residual,
        emb_init_std=emb_init_std,
        emb_init_uniform_lim=emb_init_uniform_lim,
        verbose=verbose,
    )


def xavier_uniform_param_init_fn_(
    module: nn.Module,
    n_layers: int,
    d_model: Optional[int] = None,
    init_div_is_residual: Union[int, float, str, bool] = True,
    emb_init_std: Optional[float] = None,
    emb_init_uniform_lim: Optional[Union[Tuple[float, float], float]] = None,
    init_gain: float = 0,
    verbose: int = 0,
    **kwargs,
):
    del kwargs  # unused, just to capture any extra args from the config
    xavier_uniform_ = partial(torch.nn.init.xavier_uniform_, gain=init_gain)

    if verbose > 1:
        warnings.warn(
            f'Using torch.nn.init.xavier_uniform_ init fn with parameters: ' +\
            f'gain={init_gain}'
        )

    generic_param_init_fn_(
        module=module,
        init_fn_=xavier_uniform_,
        d_model=d_model,
        n_layers=n_layers,
        init_div_is_residual=init_div_is_residual,
        emb_init_std=emb_init_std,
        emb_init_uniform_lim=emb_init_uniform_lim,
        verbose=verbose,
    )


def xavier_normal_param_init_fn_(
    module: nn.Module,
    n_layers: int,
    d_model: Optional[int] = None,
    init_div_is_residual: Union[int, float, str, bool] = True,
    emb_init_std: Optional[float] = None,
    emb_init_uniform_lim: Optional[Union[Tuple[float, float], float]] = None,
    init_gain: float = 0,
    verbose: int = 0,
    **kwargs,
):
    xavier_normal_ = partial(torch.nn.init.xavier_normal_, gain=init_gain)

    if verbose > 1:
        warnings.warn(
            f'Using torch.nn.init.xavier_normal_ init fn with parameters: ' +\
            f'gain={init_gain}'
        )

    generic_param_init_fn_(
        module=module,
        init_fn_=xavier_normal_,
        d_model=d_model,
        n_layers=n_layers,
        init_div_is_residual=init_div_is_residual,
        emb_init_std=emb_init_std,
        emb_init_uniform_lim=emb_init_uniform_lim,
        verbose=verbose,
    )


MODEL_INIT_REGISTRY = {
    'default_': torch_default_param_init_fn_,
    'baseline_': baseline_param_init_fn_,
    'kaiming_uniform_': kaiming_uniform_param_init_fn_,
    'kaiming_normal_': kaiming_normal_param_init_fn_,
    'neox_init_': neox_param_init_fn_,
    'small_init_': small_param_init_fn_,
    'xavier_uniform_': xavier_uniform_param_init_fn_,
    'xavier_normal_': xavier_normal_param_init_fn_,
}
