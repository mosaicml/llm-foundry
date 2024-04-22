# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import copy
import logging
import math
import warnings
from dataclasses import dataclass, fields
from typing import (Any, Callable, Dict, List, Literal, Mapping, Optional, Set,
                    Tuple, TypeVar, Union)

from composer.utils import dist
from omegaconf import MISSING, DictConfig, ListConfig, MissingMandatoryValue
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


@dataclass
class EvalConfig:
    # Eval Config required parameters:
    models: List[Dict[str, Any]] = MISSING
    max_seq_len: int = MISSING
    device_eval_batch_size: int = MISSING

    # Eval Config optional parameters:
    code_paths: Optional[List[str]] = None

    # Eval hyperparameters
    eval_gauntlet: Optional[Dict[str, Any]] = None
    eval_gauntlet_str: Optional[str] = None
    eval_loader: Optional[Dict[str, Any]] = None
    eval_loaders: Optional[List[Dict[str, Any]]] = None
    eval_subset_num_batches: int = -1
    icl_subset_num_batches: Optional[int] = None
    # One of icl_tasks or icl_tasks_str must be specified
    icl_tasks: Optional[List[Dict[str, Any]]] = None
    icl_tasks_str: Optional[str] = None

    # Logging parameters
    python_log_level: str = 'debug'
    loggers: Optional[Dict[str, Any]] = None
    log_config: bool = True

    # Model/run parameters
    seed: int = 17
    precision: str = 'amp_bf16'
    run_name: Optional[str] = None
    metadata: Optional[Dict[str, str]] = None

    # Distributed parameters
    dist_timeout: Union[float, int] = 600.0
    fsdp_config: Optional[Dict[str, Any]] = None

    # Callback parameters
    callbacks: Optional[Dict[str, Any]] = None

    # Variables to ignore
    variables: Optional[Dict[str, Any]] = None


EVAL_CONFIG_KEYS = set(field.name for field in fields(EvalConfig))


@dataclass
class TrainConfig:
    """Dataclass for training configuration."""

    # Mandatory model training parameters
    model: Dict[str, Any] = MISSING
    tokenizer: Dict[str, Any] = MISSING
    optimizer: Dict[str, Any] = MISSING
    scheduler: Dict[str, Any] = MISSING
    train_loader: Dict[str, Any] = MISSING
    device_train_batch_size: int = MISSING
    device_eval_batch_size: int = MISSING
    max_duration: Union[int, str] = MISSING
    eval_interval: Union[int, str] = MISSING
    precision: str = 'amp_bf16'
    max_seq_len: int = MISSING
    seed: int = MISSING

    # Optional model training parameters

    # Code paths to import
    code_paths: Optional[List[str]] = None

    # Cuda allocation configuration
    max_split_size_mb: Optional[int] = None
    expandable_segments: bool = False
    cuda_load_lazy: bool = False

    # Distributed training parameters
    dist_timeout: Union[int, float] = 600.0
    fsdp_config: Optional[Dict[str, Any]] = None

    # Evaluation parameters
    eval_loader: Optional[Dict[str, Any]] = None
    eval_loaders: Optional[List[Dict[
        str, Any]]] = None  # should not be set by the user
    icl_tasks: Optional[List[Dict[str, Any]]] = None
    icl_tasks_str: Optional[str] = None  # should not be set by the user
    eval_gauntlet: Optional[Dict[str, Any]] = None
    eval_gauntlet_str: Optional[str] = None  # should not be set by the user
    icl_subset_num_batches: Optional[int] = None
    icl_seq_len: Optional[int] = None

    # Logging
    loggers: Optional[Dict[str, Any]] = None
    progress_bar: bool = False
    log_to_console: bool = True
    python_log_level: Optional[str] = 'debug'
    console_log_interval: Union[int, str] = '1ba'
    log_config: bool = True

    # Callbacks
    callbacks: Optional[Dict[str, Any]] = None
    algorithms: Optional[Dict[str, Any]] = None

    # Checkpoints
    save_folder: Optional[str] = None
    save_latest_filename: Optional[str] = None
    save_overwrite: bool = False
    save_weights_only: bool = False
    save_filename: Optional[str] = None
    save_interval: Union[str, int] = '1000ba'
    save_num_checkpoints_to_keep: int = -1
    load_path: Optional[str] = None
    load_weights_only: bool = False
    load_strict_model_weights: bool = True
    load_ignore_keys: Optional[List[str]] = None
    save_ignore_keys: Optional[List[str]] = None

    # Dataloader
    device_train_microbatch_size: Union[str, int] = 'auto'
    global_train_batch_size: Optional[int] = None

    # Eval dataloader
    eval_subset_num_batches: int = -1
    eval_first: bool = False
    compile_config: Optional[Dict[str, Any]] = None

    # Metadata
    metadata: Optional[Dict[str, Any]] = None
    run_name: Optional[str] = None

    # Resumption
    autoresume: bool = False

    # Profiling
    profiler: Optional[Dict[str, Any]] = None

    # Variables to ignore
    variables: Optional[Dict[str, Any]] = None


TRAIN_CONFIG_KEYS = set(field.name for field in fields(TrainConfig))


def forbid_config_key(cfg_dict: Dict[str, Any], key: str):
    if key in cfg_dict:
        raise ValueError(
            f'Config key `{key}` should not be set. Please remove it from the config.'
        )


def to_dict_container(cfg: Union[DictConfig, Dict[str, Any]]) -> Dict[str, Any]:
    maybe_dict = to_container(cfg)
    if isinstance(maybe_dict, dict):
        return maybe_dict
    else:
        raise ValueError(f'Expected a dict-like type, got {type(maybe_dict)}')


def to_list_container(
        cfg: Union[ListConfig, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    maybe_list = to_container(cfg)
    if isinstance(maybe_list, list):
        return maybe_list
    else:
        raise ValueError(f'Expected a list-like type, got {type(maybe_list)}')


def to_container(
    cfg: Optional[Union[DictConfig, ListConfig, Dict[str, Any],
                        List[Dict[str, Any]]]]
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Converts a DictConfig or ListConfig to a dict or list recursively.

    `omegaconf.to_container` does not handle nested DictConfig or ListConfig
    objects, so this function is used to convert them to dicts or lists.
    """

    def rh(x: Any) -> Any:  # recursive helper
        if isinstance(x, DictConfig):
            return {k: rh(v) for k, v in x.items()}
        elif isinstance(x, ListConfig):
            return [rh(v) for v in x]
        else:
            return x

    return rh(cfg)


T = TypeVar('T')


def make_dataclass_and_log_config(
        cfg: DictConfig,
        dataclass_constructor: Callable[..., T],
        dataclass_fields: Set[str],
        transforms: Optional[List[Callable[[Dict[str, Any]],
                                           Dict[str, Any]]]] = None,
        icl_tasks_required: bool = False) -> Tuple[Dict[str, Any], T]:
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
        if icl_tasks_required:
            raise MissingMandatoryValue(
                'icl_tasks must be specified in the config')

    # Create copy of config for logging
    logged_cfg: Dict[str, Any] = copy.deepcopy(unstructured_config)

    # apply transforms to the unstructured config before constructing dataclass
    for transform in transforms or []:
        unstructured_config = transform(unstructured_config)

    logged_cfg.update(unstructured_config, merge=True)

    arg_config_keys = set(unstructured_config.keys())
    extraneous_keys = set.difference(arg_config_keys, dataclass_fields)

    if 'variables' not in unstructured_config:
        unstructured_config['variables'] = {}

    for key in extraneous_keys:
        warnings.warn(
            f'Unused parameter {key} found in cfg. Please check your yaml to ensure this parameter is necessary. Interpreting {key} as a variable for logging purposes. Top-level variables are deprecated and will not be supported in future releases.',
            category=DeprecationWarning)
        unstructured_config['variables'][key] = unstructured_config.pop(key)

    dataclass_config: T = om.structured(
        dataclass_constructor(**unstructured_config))

    return logged_cfg, dataclass_config


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


def log_config(cfg: Dict[str, Any]) -> None:
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
            wandb.config.update(cfg)

    if 'mlflow' in cfg.get('loggers', {}):
        try:
            import mlflow
        except ImportError as e:
            raise e
        if mlflow.active_run():
            mlflow.log_params(params=cfg)
