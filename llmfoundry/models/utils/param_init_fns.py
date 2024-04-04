# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import math
import warnings
from collections.abc import Sequence
from functools import partial
from typing import Any, Callable, Optional, Tuple, Union

import torch
from torch import nn

from llmfoundry.layers_registry import module_init_fns, norms, param_init_fns
from llmfoundry.models.layers.fc import FC_CLASS_REGISTRY

try:
    import transformer_engine.pytorch as te
except:
    te = None


def torch_default_param_init_fn_(
    module: nn.Module,
    **kwargs: Any,
) -> None:
    del kwargs  # unused, just to capture any extra args from the config

    if hasattr(module, 'reset_parameters') and isinstance(
            module.reset_parameters, Callable):
        module.reset_parameters()


def fused_init_helper_(module: nn.Module, init_fn_: Callable) -> None:
    # parameter initialization is often based on the parameters shape.
    # If a layer is fused, initialization should be based on the shapes
    # of the original tensor instead of the shape of the fused tensor.
    # Layers which are fused should have the _fused attribute defined.
    # The first element of _fused is the dimension along which the tensor is fused.
    # This is followed by an iterable of split indices."

    _fused = getattr(module, '_fused', None)

    if _fused is None:
        raise RuntimeError(f'Internal logic error')

    assert isinstance(module.weight, torch.Tensor)

    dim, splits = _fused
    splits = (0, *splits, module.weight.size(dim))
    for s, e in zip(splits[:-1], splits[1:]):
        slice_indices = [slice(None)] * module.weight.ndim
        slice_indices[dim] = slice(s, e)
        init_fn_(module.weight[slice_indices])


def fc_init(
    module: nn.Module,
    init_fn_: Callable,
    init_div_is_residual: Union[int, float, str, bool],
    div_is_residual: Optional[float],
    **kwargs: Any,
) -> bool:
    del kwargs  # unused, just to capture any extra args

    if isinstance(module, tuple(set(FC_CLASS_REGISTRY.values()))):
        if hasattr(module, '_fused'):
            fused_init_helper_(module, init_fn_)
        else:
            init_fn_(module.weight)
        if module.bias is not None:
            assert isinstance(module.bias, torch.Tensor)
            torch.nn.init.zeros_(module.bias)

        if init_div_is_residual is not False and getattr(
                module, '_is_residual', False):
            with torch.no_grad():
                module.weight.div_(div_is_residual)  # type: ignore
        return True

    return False


def embedding_init(
    module: nn.Module,
    init_fn_: Callable,
    emb_init_std: Optional[float],
    emb_init_uniform_lim: Optional[Union[Tuple[float, float], float]],
    **kwargs: Any,
) -> bool:
    del kwargs  # unused, just to capture any extra args

    if isinstance(module, nn.Embedding):
        # Embedding
        if emb_init_std is not None:
            std = emb_init_std
            if std == 0:
                warnings.warn(f'Embedding layer initialized to 0.')
            emb_init_fn_ = partial(torch.nn.init.normal_, mean=0.0, std=std)
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
        else:
            emb_init_fn_ = init_fn_

        emb_init_fn_(module.weight)

        return True

    return False


def norm_init(
    module: nn.Module,
    **kwargs: Any,
) -> bool:
    del kwargs  # unused, just to capture any extra args

    if isinstance(module,
                  tuple(set([norms.get(name) for name in norms.get_all()]))):
        # Norm
        if hasattr(module, 'weight') and isinstance(module.weight,
                                                    torch.Tensor):
            torch.nn.init.ones_(module.weight)
        if hasattr(module, 'bias') and isinstance(module.bias, torch.Tensor):
            torch.nn.init.zeros_(module.bias)

        return True

    return False


def multihead_attention_init(
    module: nn.Module,
    init_fn_: Callable,
    d_model: Optional[int],
    init_div_is_residual: Union[int, float, str, bool],
    div_is_residual: float,
    **kwargs: Any,
) -> bool:
    del kwargs  # unused, just to capture any extra args

    if isinstance(module, nn.MultiheadAttention):
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

        return True

    return False


def te_layernorm_mlp_init(
    module: nn.Module,
    init_fn_: Callable,
    **kwargs: Any,
) -> bool:
    del kwargs  # unused, just to capture any extra args

    if te is not None and isinstance(module, te.LayerNormMLP):
        if isinstance(module.layer_norm_weight, torch.Tensor):
            torch.nn.init.ones_(module.layer_norm_weight)
        if isinstance(module.layer_norm_bias, torch.Tensor):
            torch.nn.init.zeros_(module.layer_norm_bias)

        init_fn_(module.fc1_weight)
        if module.fc1_bias is not None:
            assert isinstance(module.fc1_bias, torch.Tensor)
            torch.nn.init.zeros_(module.fc1_bias)
        init_fn_(module.fc2_weight)
        if module.fc2_bias is not None:
            assert isinstance(module.fc2_bias, torch.Tensor)
            torch.nn.init.zeros_(module.fc2_bias)

        with torch.no_grad():
            module.fc2_weight.div_(div_is_residual)  # type: ignore

        return True

    return False


def generic_param_init_fn_(
    module: nn.Module,
    init_fn_: Callable,
    n_layers: int,
    d_model: Optional[int] = None,
    init_div_is_residual: Union[int, float, str, bool] = True,
    emb_init_std: Optional[float] = None,
    emb_init_uniform_lim: Optional[Union[Tuple[float, float], float]] = None,
    **kwargs: Any,
) -> None:
    del kwargs  # unused, just to capture any extra args from the config
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
    elif init_div_is_residual.isnumeric():
        # do not trust YAML parsing to always convert numbers to numbers
        div_is_residual = float(init_div_is_residual)
    else:
        # not used, for pyright
        div_is_residual = 1.0
        raise ValueError(
            f'Expected init_div_is_residual to be boolean or numeric, got {init_div_is_residual}'
        )

    # all_module_init_fns = [
    #     module_init_fns.get(name) for name in module_init_fns.get_all()
    # ]
    # did_init = False
    # for module_init_fn in all_module_init_fns:
    #     did_init = module_init_fn(
    #         module=module,
    #         init_fn_=init_fn_,
    #         d_model=d_model,
    #         init_div_is_residual=init_div_is_residual,
    #         div_is_residual=div_is_residual,
    #         emb_init_std=emb_init_std,
    #         emb_init_uniform_lim=emb_init_uniform_lim,
    #     )

    #     if did_init:
    #         break

    # if not did_init:
    #     for _ in module.parameters(recurse=False):
    #         # raise error if uninitialized module has any parameters
    #         raise NotImplementedError(
    #             f'{module.__class__.__name__} parameters are not initialized by any of the registered module_init_fns. '
    #             +
    #             'Please add an appropriate module_init_fn to the registry. Currently registered module_init_fns are: '
    #             + ', '.join(module_init_fns.get_all()))


def _normal_init_(std: float, mean: float = 0.0) -> Callable:
    return partial(torch.nn.init.normal_, mean=mean, std=std)


def _normal_param_init_fn_(
    module: nn.Module,
    std: float,
    n_layers: int,
    d_model: Optional[int] = None,
    init_div_is_residual: Union[int, float, str, bool] = True,
    emb_init_std: Optional[float] = None,
    emb_init_uniform_lim: Optional[Union[Tuple[float, float], float]] = None,
    **kwargs: Any,
) -> None:
    del kwargs  # unused, just to capture any extra args from the config
    init_fn_ = _normal_init_(std=std)

    generic_param_init_fn_(
        module=module,
        init_fn_=init_fn_,
        d_model=d_model,
        n_layers=n_layers,
        init_div_is_residual=init_div_is_residual,
        emb_init_std=emb_init_std,
        emb_init_uniform_lim=emb_init_uniform_lim,
    )


def baseline_param_init_fn_(
    module: nn.Module,
    init_std: Optional[float],
    n_layers: int,
    d_model: Optional[int] = None,
    init_div_is_residual: Union[int, float, str, bool] = True,
    emb_init_std: Optional[float] = None,
    emb_init_uniform_lim: Optional[Union[Tuple[float, float], float]] = None,
    **kwargs: Any,
) -> None:
    del kwargs  # unused, just to capture any extra args from the config
    if init_std is None:
        raise ValueError(
            "You must set model.init_config['init_std'] to a float value to use the default initialization scheme."
        )
    _normal_param_init_fn_(
        module=module,
        std=init_std,
        d_model=d_model,
        n_layers=n_layers,
        init_div_is_residual=init_div_is_residual,
        emb_init_std=emb_init_std,
        emb_init_uniform_lim=emb_init_uniform_lim,
    )


def small_param_init_fn_(
    module: nn.Module,
    n_layers: int,
    d_model: int,
    init_div_is_residual: Union[int, float, str, bool] = True,
    emb_init_std: Optional[float] = None,
    emb_init_uniform_lim: Optional[Union[Tuple[float, float], float]] = None,
    **kwargs: Any,
) -> None:
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
    )


def neox_param_init_fn_(
    module: nn.Module,
    n_layers: int,
    d_model: int,
    emb_init_std: Optional[float] = None,
    emb_init_uniform_lim: Optional[Union[Tuple[float, float], float]] = None,
    **kwargs: Any,
) -> None:
    """From section 2.3.1 of GPT-NeoX-20B:

    An Open-Source AutoregressiveLanguage Model — Black et. al. (2022)
    see https://github.com/EleutherAI/gpt-neox/blob/9610391ab319403cef079b438edd016a2443af54/megatron/model/init_functions.py#L151
    and https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/transformer.py
    """
    del kwargs  # unused, just to capture any extra args from the config
    residual_div = n_layers / math.sqrt(10)  # small std / wang std

    small_param_init_fn_(
        module=module,
        d_model=d_model,
        n_layers=n_layers,
        init_div_is_residual=residual_div,
        emb_init_std=emb_init_std,
        emb_init_uniform_lim=emb_init_uniform_lim,
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
    **kwargs: Any,
) -> None:
    del kwargs  # unused, just to capture any extra args from the config

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
    **kwargs: Any,
) -> None:
    del kwargs  # unused, just to capture any extra args from the config

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
    )


def xavier_uniform_param_init_fn_(
    module: nn.Module,
    n_layers: int,
    d_model: Optional[int] = None,
    init_div_is_residual: Union[int, float, str, bool] = True,
    emb_init_std: Optional[float] = None,
    emb_init_uniform_lim: Optional[Union[Tuple[float, float], float]] = None,
    init_gain: float = 0,
    **kwargs: Any,
) -> None:
    del kwargs  # unused, just to capture any extra args from the config
    xavier_uniform_ = partial(torch.nn.init.xavier_uniform_, gain=init_gain)

    generic_param_init_fn_(
        module=module,
        init_fn_=xavier_uniform_,
        d_model=d_model,
        n_layers=n_layers,
        init_div_is_residual=init_div_is_residual,
        emb_init_std=emb_init_std,
        emb_init_uniform_lim=emb_init_uniform_lim,
    )


def xavier_normal_param_init_fn_(
    module: nn.Module,
    n_layers: int,
    d_model: Optional[int] = None,
    init_div_is_residual: Union[int, float, str, bool] = True,
    emb_init_std: Optional[float] = None,
    emb_init_uniform_lim: Optional[Union[Tuple[float, float], float]] = None,
    init_gain: float = 0,
    **kwargs: Any,
) -> None:
    del kwargs  # unused, just to capture any extra args from the config
    xavier_normal_ = partial(torch.nn.init.xavier_normal_, gain=init_gain)

    generic_param_init_fn_(
        module=module,
        init_fn_=xavier_normal_,
        d_model=d_model,
        n_layers=n_layers,
        init_div_is_residual=init_div_is_residual,
        emb_init_std=emb_init_std,
        emb_init_uniform_lim=emb_init_uniform_lim,
    )


param_init_fns.register('default_', func=torch_default_param_init_fn_)
param_init_fns.register('baseline_', func=baseline_param_init_fn_)
param_init_fns.register('kaiming_uniform_', func=kaiming_uniform_param_init_fn_)
param_init_fns.register('kaiming_normal_', func=kaiming_normal_param_init_fn_)
param_init_fns.register('neox_init_', func=neox_param_init_fn_)
param_init_fns.register('small_init_', func=small_param_init_fn_)
param_init_fns.register('xavier_uniform_', func=xavier_uniform_param_init_fn_)
param_init_fns.register('xavier_normal_', func=xavier_normal_param_init_fn_)

module_init_fns.register('fc', func=fc_init)
module_init_fns.register('embedding', func=embedding_init)
module_init_fns.register('norm', func=norm_init)
module_init_fns.register('multihead_attention', func=multihead_attention_init)
module_init_fns.register('te_layernorm_mlp', func=te_layernorm_mlp_init)
