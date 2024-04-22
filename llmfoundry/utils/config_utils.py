# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import copy
import logging
import math
import warnings
from typing import (Any, Callable, Dict, List, Literal, Mapping, Optional, Set,
                    Tuple, Union)

from composer.utils import dist
from omegaconf import DictConfig, ListConfig
from omegaconf import OmegaConf as om

from llmfoundry.layers_registry import ffns_with_megablocks
from llmfoundry.models.utils import init_empty_weights

log = logging.getLogger(__name__)

__all__ = [
    'pop_config',
    'calculate_batch_size_info',
    'update_batch_size_info',
    'process_init_device',
    'log_config',
]


def forbid_config_key(cfg_dict: Dict[str, Any], key: str):
    if key in cfg_dict:
        raise ValueError(
            f'Config key `{key}` should not be set. Please remove it from the config.'
        )


def to_dict_recursive(cfg: Union[DictConfig, Dict[str, Any]]) -> Dict[str, Any]:
    maybe_dict = to_container_recursive(cfg)
    if isinstance(maybe_dict, dict):
        return maybe_dict
    else:
        raise ValueError(f'Expected a dict-like type, got {type(maybe_dict)}')


def to_list_recursive(
        cfg: Union[ListConfig, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    maybe_list = to_container_recursive(cfg)
    if isinstance(maybe_list, list):
        return maybe_list
    else:
        raise ValueError(f'Expected a list-like type, got {type(maybe_list)}')


def to_container_recursive(
    cfg: Optional[Union[DictConfig, ListConfig, Dict[str, Any],
                        List[Dict[str, Any]]]]
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:

    def rh(x: Any) -> Any:  # recursive helper
        if isinstance(x, DictConfig):
            return {k: rh(v) for k, v in x.items()}
        elif isinstance(x, ListConfig):
            return [rh(v) for v in x]
        else:
            return x

    return rh(cfg)


def make_dataclass_and_log_config(
    cfg: DictConfig, dataclass_constructor: Callable[..., Any],
    dataclass_fields: Set[str],
    transforms: Optional[List[Callable[[Dict[str, Any]], Dict[str, Any]]]]
) -> Tuple[DictConfig, Any]:
    """Converts a DictConfig to a dataclass and creates a logged config."""
    # Resolve all interpolation variables as early as possible
    unstructured_config = om.to_container(cfg, resolve=True)
    assert isinstance(unstructured_config, dict)
    assert all(isinstance(k, str) for k in unstructured_config.keys())
    unstructured_config = {str(k): v for k, v in unstructured_config.items()}

    # Flatten union types before creating structured config:
    if 'eval_gauntlet' in unstructured_config:
        forbid_config_key(unstructured_config, 'eval_gauntlet_str')
        if isinstance(unstructured_config['eval_gauntlet'], str):
            unstructured_config['eval_gauntlet_str'] = unstructured_config.pop(
                'eval_gauntlet')
    if (loader := unstructured_config.get('eval_loader', None)) is not None:
        forbid_config_key(unstructured_config, 'eval_loaders')
        if isinstance(loader, list):
            unstructured_config['eval_loaders'] = unstructured_config.pop(
                'eval_loader')
    if 'icl_tasks' in unstructured_config:
        forbid_config_key(unstructured_config, 'icl_tasks_str')
        if isinstance(unstructured_config['icl_tasks'], str):
            unstructured_config['icl_tasks_str'] = unstructured_config.pop(
                'icl_tasks')
    else:
        raise ValueError('icl_tasks must be specified in the config')

    arg_config_keys = set(unstructured_config.keys())
    extraneous_keys = set.difference(arg_config_keys, dataclass_fields)

    if 'variables' not in unstructured_config:
        unstructured_config['variables'] = {}

    for key in extraneous_keys:
        warnings.warn(
            f'Unused parameter {key} found in cfg. Please check your yaml to ensure this parameter is necessary. Interpreting {key} as a variable for logging purposes. Top-level variables are deprecated and will not be supported in future releases.',
            DeprecationWarning)
        unstructured_config['variables'][key] = unstructured_config.pop(key)

    # Create copy of config for logging
    logged_cfg: DictConfig = copy.deepcopy(DictConfig(unstructured_config))

    # apply transforms to the unstructured config before constructing dataclass
    for transform in transforms or []:
        unstructured_config = transform(unstructured_config)

    logged_cfg.update(unstructured_config, merge=True)

    eval_config: DictConfig = om.structured(
        dataclass_constructor(**unstructured_config))

    return logged_cfg, eval_config


def pop_config(cfg: Union[Dict[str, Any], DictConfig],
               key: str,
               must_exist: bool = True,
               default_value: Any = None,
               convert: bool = False) -> Any:
    """Pop a value from the main config file and return it.

    If the key does not exist, return the default_value or raise a RuntimeError
    depending on the must_exist flag. If the convert flag is set to True, then
    we will convert the value to a python object using OmegaConf.to_container.
    """
    value = cfg.pop(key, None)
    if value is not None and convert:
        if not isinstance(value, DictConfig) and not isinstance(
                value, ListConfig):
            raise ValueError(
                f'The key {key} has a value of type {type(value)} that cannot be \
                            converted to a dict or list. Please check your yaml.'
            )
        return om.to_container(value)
    elif value is not None:
        return value
    elif must_exist:
        raise NameError(
            f'The {key} parameter is missing and must exist for execution. Please check your yaml.'
        )
    else:
        return default_value


def calculate_batch_size_info(
    global_batch_size: int, device_microbatch_size: Union[int, Literal['auto']]
) -> Tuple[int, Union[int, Literal['auto']], Union[int, Literal['auto']]]:
    if global_batch_size % dist.get_world_size() != 0:
        raise ValueError(
            f'Global batch size {global_batch_size} is not divisible by {dist.get_world_size()} '
            +
            'as a result, the batch size would be truncated, please adjust `global_batch_size` '
            + f'to be divisible by world size, {dist.get_world_size()}.')
    device_batch_size = global_batch_size // dist.get_world_size()
    if device_microbatch_size == 'auto':
        device_grad_accum = 'auto'
    elif isinstance(device_microbatch_size, int):
        if device_microbatch_size > device_batch_size:
            log.warn(
                f'device_microbatch_size > device_batch_size, ' +
                f'will be reduced from {device_microbatch_size} -> {device_batch_size}.'
            )
            device_microbatch_size = device_batch_size
        device_grad_accum = math.ceil(device_batch_size /
                                      device_microbatch_size)
    else:
        raise ValueError(f'Not sure how to parse {device_microbatch_size=}')

    return device_batch_size, device_microbatch_size, device_grad_accum


# Coming soon: this conversion math will be done inside Composer Trainer
def update_batch_size_info(cfg: Dict[str, Any]) -> Dict[str, Any]:
    device_train_batch_size, device_train_microbatch_size, device_train_grad_accum = calculate_batch_size_info(
        cfg['global_train_batch_size'], cfg['device_train_microbatch_size'])
    cfg['n_gpus'] = dist.get_world_size()
    cfg['device_train_batch_size'] = device_train_batch_size
    cfg['device_train_microbatch_size'] = device_train_microbatch_size
    cfg['device_train_grad_accum'] = device_train_grad_accum
    # Safely set `device_eval_batch_size` if not provided by user
    if 'device_eval_batch_size' not in cfg:
        if cfg['device_train_microbatch_size'] == 'auto':
            cfg['device_eval_batch_size'] = 1  # TODO debug auto eval microbatching
        else:
            cfg['device_eval_batch_size'] = cfg['device_train_microbatch_size']
    return cfg


def process_init_device(model_cfg: Dict[str, Any], fsdp_config: Optional[Dict]):
    # Restrict model init_device to 'meta' and 'cpu',
    # using 'cuda' vs. 'cuda:id' is tricky and can lead to common user errors
    # when multiple GPUs are available.
    # Also 'meta' is only valid when using FSDP
    init_context = contextlib.nullcontext()
    if 'init_device' in model_cfg:
        assert model_cfg['init_device'] in ['meta', 'cpu', 'mixed']
        if fsdp_config is None and model_cfg['init_device'] == 'meta':
            warnings.warn(
                "Using `cfg.model.init_device='meta'` is only valid when using FSDP! " +\
                "Reverting to `cfg.model.init_device='cpu'`.")
            model_cfg['init_device'] = 'cpu'
        if model_cfg['init_device'] == 'meta':
            init_context = init_empty_weights()
        if model_cfg['init_device'] == 'mixed':
            if fsdp_config is None:
                raise NotImplementedError(
                    'Using init_device `mixed` is only supported with FSDP. ' +
                    'Please add a FSDP config.')
            # Always set `sync_module_states` to True for mixed initialization
            if not fsdp_config.get('sync_module_states', False):
                warnings.warn((
                    'Setting `sync_module_states = True` for FSDP. This is required '
                    'when using mixed initialization.'))
                fsdp_config['sync_module_states'] = True

            # Set defaults for mixed initialization
            fsdp_config.setdefault('use_orig_params', False)
            fsdp_config.setdefault('load_monolith_rank0_only', True)

    # Set ffn_config.device_mesh to fsdp_config.device_mesh
    if fsdp_config is not None and 'device_mesh' in fsdp_config and 'ffn_config' in model_cfg and model_cfg[
            'ffn_config'].get('ffn_type', None) in ffns_with_megablocks:
        # Raise ValueError if not using device mesh with MoE expert parallelism
        if fsdp_config['device_mesh'] is None and model_cfg['ffn_config'].get(
                'moe_world_size', 1) > 1:
            raise ValueError(
                'device_mesh must be specified in fsdp_config when using MoE with moe_world_size > 1.'
            )
        model_cfg['ffn_config']['device_mesh'] = fsdp_config['device_mesh']

    # No mixed precision needed for weights when they're already 16 bits
    master_dtype = model_cfg.get('master_weights_dtype')
    small_dtypes = ('bf16', 'fp16', 'float16', 'bfloat16', 'amp_fp16',
                    'amp_bf16')
    if fsdp_config and master_dtype in small_dtypes:
        reduce_dtype = None
        buffer_dtype = None
        mixed_precision = fsdp_config.get('mixed_precision')
        if isinstance(mixed_precision, Mapping):
            reduce_dtype = mixed_precision.get('reduce_dtype')
            buffer_dtype = mixed_precision.get('buffer_dtype')
        fsdp_config['mixed_precision'] = {
            'param_dtype': None,
            'reduce_dtype': reduce_dtype,
            'buffer_dtype': buffer_dtype,
            'keep_low_precision_grads': True,
        }

    return init_context


def log_config(cfg: DictConfig) -> None:
    """Logs the current config and updates the wandb and mlflow configs.

    This function can be called multiple times to update the wandb and MLflow
    config with different variables.
    """
    print(om.to_yaml(cfg))
    if 'wandb' in cfg.get('loggers', {}):
        try:
            import wandb
        except ImportError as e:
            raise e
        if wandb.run:
            wandb.config.update(om.to_container(cfg, resolve=True))

    if 'mlflow' in cfg.get('loggers', {}):
        try:
            import mlflow
        except ImportError as e:
            raise e
        if mlflow.active_run():
            mlflow.log_params(params=om.to_container(cfg, resolve=True))
