# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import logging
import math
import os
import warnings
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Union

import mlflow
from composer.utils import dist, parse_uri
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


def pop_config(
    cfg: DictConfig,
    key: str,
    must_exist: bool = True,
    default_value: Any = None,
    convert: bool = False,
) -> Any:
    """Pop a value from the main config file and return it.

    If the key does not exist, return the default_value or raise a RuntimeError
    depending on the must_exist flag. If the convert flag is set to True, then
    we will convert the value to a python object using OmegaConf.to_container.
    """
    value = cfg.pop(key, None)
    if value is not None and convert:
        if not isinstance(value,
                          DictConfig) and not isinstance(value, ListConfig):
            raise ValueError(
                f'The key {key} has a value of type {type(value)} that cannot be \
                            converted to a dict or list. Please check your yaml.',
            )
        return om.to_container(value)
    elif value is not None:
        return value
    elif must_exist:
        raise NameError(
            f'The {key} parameter is missing and must exist for execution. Please check your yaml.',
        )
    else:
        return default_value


def calculate_batch_size_info(
    global_batch_size: int,
    device_microbatch_size: Union[int, float, Literal['auto']],
    data_replication_degree: int = 1,
) -> Tuple[Union[int, float], Union[int, float, Literal['auto']], Union[
    int, Literal['auto']]]:
    if dist.get_world_size() % data_replication_degree != 0:
        raise ValueError(
            f'World size {dist.get_world_size()} is not divisible by data replication degree {data_replication_degree}.',
        )
    if global_batch_size % (
        dist.get_world_size() // data_replication_degree
    ) != 0:
        raise ValueError(
            f'Global batch size {global_batch_size} is not divisible by {(dist.get_world_size() // data_replication_degree)=} '
            +
            'as a result, the batch size would be truncated, please adjust `global_batch_size` '
            + f'to be divisible by world size, {dist.get_world_size()}.',
        )
    device_batch_size = global_batch_size / dist.get_world_size()
    if device_batch_size == round(device_batch_size):
        device_batch_size = round(device_batch_size)
    if device_microbatch_size == 'auto':
        device_grad_accum = 'auto'
    elif isinstance(device_microbatch_size, (int, float)):
        if device_microbatch_size > device_batch_size:
            log.warn(
                f'device_microbatch_size > device_batch_size, ' +
                f'will be reduced from {device_microbatch_size} -> {device_batch_size}.',
            )
            device_microbatch_size = device_batch_size
        device_grad_accum = math.ceil(
            device_batch_size / device_microbatch_size,
        )
    else:
        raise ValueError(f'Not sure how to parse {device_microbatch_size=}')

    return device_batch_size, device_microbatch_size, device_grad_accum


# Coming soon: this conversion math will be done inside Composer Trainer
def update_batch_size_info(cfg: DictConfig) -> DictConfig:
    data_replication_degree = 1
    device_train_batch_size, device_train_microbatch_size, device_train_grad_accum = calculate_batch_size_info(
        cfg.global_train_batch_size,
        cfg.device_train_microbatch_size,
        data_replication_degree=data_replication_degree,
    )
    cfg.n_gpus = dist.get_world_size()
    cfg.device_train_batch_size = device_train_batch_size
    cfg.device_train_microbatch_size = device_train_microbatch_size
    cfg.device_train_grad_accum = device_train_grad_accum
    # Safely set `device_eval_batch_size` if not provided by user
    if 'device_eval_batch_size' not in cfg:
        if cfg.device_train_microbatch_size == 'auto':
            cfg.device_eval_batch_size = 1  # TODO debug auto eval microbatching
        else:
            cfg.device_eval_batch_size = cfg.device_train_microbatch_size
    return cfg


def process_init_device(model_cfg: DictConfig, fsdp_config: Optional[Dict]):
    # Restrict model init_device to 'meta' and 'cpu',
    # using 'cuda' vs. 'cuda:id' is tricky and can lead to common user errors
    # when multiple GPUs are available.
    # Also 'meta' is only valid when using FSDP
    init_context = contextlib.nullcontext()
    if 'init_device' in model_cfg:
        assert model_cfg.init_device in ['meta', 'cpu', 'mixed']
        if fsdp_config is None and model_cfg.init_device == 'meta':
            warnings.warn(
                "Using `cfg.model.init_device='meta'` is only valid when using FSDP! " +\
                "Reverting to `cfg.model.init_device='cpu'`.")
            model_cfg.init_device = 'cpu'
        if model_cfg.init_device == 'meta':
            init_context = init_empty_weights()
        if model_cfg.init_device == 'mixed':
            if fsdp_config is None:
                raise NotImplementedError(
                    'Using init_device `mixed` is only supported with FSDP. ' +
                    'Please add a FSDP config.',
                )
            # Always set `sync_module_states` to True for mixed initialization
            if not fsdp_config.get('sync_module_states', False):
                warnings.warn((
                    'Setting `sync_module_states = True` for FSDP. This is required '
                    'when using mixed initialization.'
                ))
                fsdp_config['sync_module_states'] = True

            # Set defaults for mixed initialization
            fsdp_config.setdefault('use_orig_params', False)
            fsdp_config.setdefault('load_monolith_rank0_only', True)

    # Set ffn_config.device_mesh to fsdp_config.device_mesh
    if fsdp_config is not None and 'device_mesh' in fsdp_config and 'ffn_config' in model_cfg and model_cfg[
        'ffn_config'].get('ffn_type', None) in ffns_with_megablocks:
        # Raise ValueError if not using device mesh with MoE expert parallelism
        if fsdp_config['device_mesh'] is None and model_cfg['ffn_config'].get(
            'moe_world_size',
            1,
        ) > 1:
            raise ValueError(
                'device_mesh must be specified in fsdp_config when using MoE with moe_world_size > 1.',
            )
        model_cfg.ffn_config.device_mesh = fsdp_config['device_mesh']

    # No mixed precision needed for weights when they're already 16 bits
    master_dtype = model_cfg.get('master_weights_dtype')
    small_dtypes = (
        'bf16',
        'fp16',
        'float16',
        'bfloat16',
        'amp_fp16',
        'amp_bf16',
    )
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

    if 'mlflow' in cfg.get('loggers', {}) and mlflow.active_run():
        mlflow.log_params(params=om.to_container(cfg, resolve=True))
        _log_dataset_uri(cfg)


def _parse_source_dataset(cfg: DictConfig) -> List[Tuple[str, str, str]]:
    """Parse a run config for dataset information.

    Given a config dictionary, parse through it to determine what the datasource
    should be categorized as. Possible data sources are Delta Tables, UC Volumes,
    HuggingFace paths, remote storage, or local storage.

    Args:
        cfg (DictConfig): A config dictionary of a run

    Returns:
        List[Tuple[str, str, str]]: A list of tuples formatted as (data type, path, split)
    """
    data_paths = []

    # Handle train loader if it exists
    train_dataset = cfg.get('train_loader', {}).get('dataset', {})
    train_split = train_dataset.get('split', None)
    train_source_path = cfg.get('source_dataset_train', None)
    _process_data_source(
        train_source_path,
        train_dataset,
        train_split,
        'train',
        data_paths,
    )

    # Handle eval_loader which might be a list or a single dictionary
    eval_data_loaders = cfg.get('eval_loader', {})
    if not isinstance(eval_data_loaders, ListConfig):
        eval_data_loaders = [
            eval_data_loaders,
        ]  # Normalize to list if it's a single dictionary

    for eval_data_loader in eval_data_loaders:
        eval_dataset = eval_data_loader.get('dataset', {})
        eval_split = eval_dataset.get('split', None)
        eval_source_path = cfg.get('source_dataset_eval', None)
        _process_data_source(
            eval_source_path,
            eval_dataset,
            eval_split,
            'eval',
            data_paths,
        )

    return data_paths


def _process_data_source(
    source_dataset_path: Optional[str],
    dataset: Dict[str, str],
    cfg_split: Optional[str],
    true_split: str,
    data_paths: List[Tuple[str, str, str]],
):
    """Add a data source by mutating data_paths.

    Given various dataset attributes, attempt to determine what type of dataset is being added, and parse
    the dataset accordingly.

    Args:
        source_dataset_path (Optional[str]): The source dataset in cfg metadata
        dataset (Dict[str, str]): The dataset from cfg
        cfg_split (str): The split listed for the dataset in cfg
        true_split (str): The split of the dataset to be added (i.e. train or eval)
        data_paths (List[Tuple[str, str, str]]): A list of tuples formatted as (data type, path, split)
    """
    # Check for Delta table
    if source_dataset_path and len(source_dataset_path.split('.')) == 3:
        data_paths.append(('delta_table', source_dataset_path, true_split))
    # Check for UC volume
    elif source_dataset_path and source_dataset_path.startswith('dbfs:'):
        data_paths.append(
            ('uc_volume', source_dataset_path[len('dbfs:'):], true_split),
        )
    # Check for HF path
    elif 'hf_name' in dataset:
        hf_path = dataset['hf_name']
        backend, _, _ = parse_uri(hf_path)
        if backend:
            hf_path = os.path.join(hf_path, cfg_split) if cfg_split else hf_path
            data_paths.append((backend, hf_path, true_split))
        elif os.path.exists(hf_path):
            data_paths.append(('local', hf_path, true_split))
        else:
            data_paths.append(('hf', hf_path, true_split))
    # Check for remote path
    elif 'remote' in dataset:
        remote_path = dataset['remote']
        backend, _, _ = parse_uri(remote_path)
        if backend:
            remote_path = os.path.join(
                remote_path,
                f'{cfg_split}/',
            ) if cfg_split else remote_path
            data_paths.append((backend, remote_path, true_split))
        else:
            data_paths.append(('local', remote_path, true_split))
    else:
        log.warning('DataSource Not Found.')


def _log_dataset_uri(cfg: DictConfig) -> None:
    """Logs dataset tracking information to MLflow.

    Args:
        cfg (DictConfig): A config dictionary of a run
    """
    # Figure out which data source to use
    data_paths = _parse_source_dataset(cfg)

    dataset_source_mapping = {
        's3': mlflow.data.http_dataset_source.HTTPDatasetSource,
        'oci': mlflow.data.http_dataset_source.HTTPDatasetSource,
        'azure': mlflow.data.http_dataset_source.HTTPDatasetSource,
        'gs': mlflow.data.http_dataset_source.HTTPDatasetSource,
        'https': mlflow.data.http_dataset_source.HTTPDatasetSource,
        'hf': mlflow.data.huggingface_dataset_source.HuggingFaceDatasetSource,
        'delta_table': mlflow.data.delta_dataset_source.DeltaDatasetSource,
        'uc_volume': mlflow.data.uc_volume_dataset_source.UCVolumeDatasetSource,
        'local': mlflow.data.http_dataset_source.HTTPDatasetSource,
    }

    # Map data source types to their respective MLFlow DataSource.
    for dataset_type, path, split in data_paths:

        if dataset_type in dataset_source_mapping:
            source_class = dataset_source_mapping[dataset_type]
            if dataset_type == 'delta_table':
                source = source_class(delta_table_name=path)
            elif dataset_type == 'hf' or dataset_type == 'uc_volume':
                source = source_class(path=path)
            else:
                source = source_class(url=path)
        else:
            log.info(
                f'{dataset_type} unknown, defaulting to http dataset source',
            )
            source = mlflow.data.http_dataset_source.HTTPDatasetSource(url=path)

        mlflow.log_input(
            mlflow.data.meta_dataset.MetaDataset(source, name=split),
        )
