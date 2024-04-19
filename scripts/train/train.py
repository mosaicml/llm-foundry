# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
import copy
import gc
import logging
import os
import sys
import time
import warnings
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from composer import ComposerModel, Trainer
from composer.core.callback import Callback
from composer.profiler import (JSONTraceHandler, Profiler, TraceHandler,
                               cyclic_schedule)
from composer.utils import dist, get_device, reproducibility
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from rich.traceback import install

from llmfoundry.eval.metrics.nlp import InContextLearningMetric
from llmfoundry.utils import (find_mosaicml_logger, log_train_analytics,
                              maybe_create_mosaicml_logger)

install()
from omegaconf import MISSING

from llmfoundry.callbacks import AsyncEval
from llmfoundry.data.dataloader import build_dataloader
from llmfoundry.layers_registry import ffns_with_megablocks
from llmfoundry.utils.builders import (add_metrics_to_eval_loaders,
                                       build_algorithm, build_callback,
                                       build_composer_model, build_evaluators,
                                       build_logger, build_optimizer,
                                       build_scheduler, build_tokenizer)
from llmfoundry.utils.config_utils import (forbid_config_key, log_config,
                                           pop_config, process_init_device,
                                           update_batch_size_info)
from llmfoundry.utils.registry_utils import import_file

log = logging.getLogger(__name__)


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
    data_local: Optional[str] = None
    data_remote: Optional[str] = None

    # Eval dataloader
    eval_subset_num_batches: int = -1
    eval_first: bool = False
    compile_config: Optional[Dict[str, Any]] = None

    # Metadata
    metadata: Optional[Dict[str, Any]] = None
    run_name: Optional[str] = None

    # Resumption
    autoresume: bool = False

    # Gradient accumulation
    device_train_grad_accum: Optional[int] = None

    # Profiling
    profiler: Optional[Dict[str, Any]] = None

    # Ignore keys
    global_seed: Optional[int] = None
    global_train_batch_size: Optional[int] = None
    n_gpus: Optional[int] = None
    variables: Optional[Dict[str, Any]] = None


TRAIN_CONFIG_KEYS = set(field.name for field in fields(TrainConfig))


def validate_config(train_config: TrainConfig):
    """Validates compatible model and dataloader selection."""
    # Check for missing mandatory fields
    for field in TRAIN_CONFIG_KEYS:
        _ = getattr(train_config, field)

    # Validate the rest of the config
    loaders = [train_config.train_loader]
    if train_config.eval_loaders is not None:
        for loader in (train_config.eval_loaders or []):  # pyright
            if 'label' not in loader or loader['label'] is None:
                raise ValueError(
                    'When specifying multiple evaluation datasets, each one must include the \
                            `label` attribute.')
            loaders.append(loader)
    if train_config.eval_loader is not None:
        loaders.append(train_config.eval_loader)
    for loader in loaders:
        if loader['name'] == 'text':
            if train_config.model['name'] == 'hf_t5':
                raise ValueError(
                    f'Model type "{train_config.model["name"]}" is not supported when using the "text " ' +\
                    f'dataloader. Only finetuning is supported.')

    if train_config.icl_tasks is not None or train_config.icl_tasks_str is not None:
        if train_config.model['name'] == 'hf_t5':
            raise ValueError(
                'ICL evaluation does not currently support Encoder-Decoder models, such as "hf_t5".'
            )

    if (train_config.model.get('fc_type', 'torch') != 'te' and
            'te' not in train_config.model.get('ffn_config', {}).get(
                'ffn_type', 'mptmlp') and 'fp8' in train_config.precision):
        warnings.warn(
            "fp8 only supported for te.Linear layers. Either set `cfg.model.fc_typ='te'` or "
            +
            "`cfg.model.ffn_config.ffn_type='te_ln_mlp'` to enable layers using fp8 precision."
        )

    if (train_config.model.get('fc_type', 'torch') == 'te' or
            'te' in train_config.model.get('ffn_config', {}).get(
                'ffn_type', 'mptmlp')):
        if train_config.fsdp_config is None:
            train_config.fsdp_config = {}
        fsdp_config = train_config.fsdp_config
        act_ckpt = fsdp_config.get('activation_checkpointing',
                                   False) if fsdp_config else False
        act_ckpt_reentrant = fsdp_config.get(
            'activation_checkpointing_reentrant', False)
        if act_ckpt == True and act_ckpt_reentrant == True:
            warnings.warn(
                '`te.Linear` layers do not support activation_checkpointing with '
                + '`activation_checkpointing_reentrant = True`. ' +
                'Setting cfg.fsdp_config.activation_checkpointing_reentrant=False.'
            )
            train_config.fsdp_config[
                'activation_checkpointing_reentrant'] = False

    if train_config.model.get('ffn_config', {}).get('ffn_type',
                                                    'mptmlp') == 'te_ln_mlp':
        warnings.warn(
            '`te.LayerNormMLP` requires has issues with torch._dynamo. ' +
            'Setting `torch._dynamo.config.suppress_errors = True` and falling back to eager.'
        )
        torch._dynamo.config.suppress_errors = True  # type: ignore (third-party)

    if train_config.model.get('load_in_8bit', False):
        raise ValueError(
            '`load_in_8bit` is only supported for evaluation rather than training.'
        )

    if train_config.model.get('ffn_config',
                              {}).get('ffn_type',
                                      'mptmlp') in ffns_with_megablocks:
        moe_world_size = train_config.model.get('ffn_config',
                                                {}).get('moe_world_size', 1)
        use_orig_params = train_config.fsdp_config.get(
            'use_orig_params',
            True) if train_config.fsdp_config is not None else True
        if moe_world_size > 1 and not use_orig_params:
            raise ValueError(
                f'MoEs with expert parallelism (moe_world_size {moe_world_size} > 1) require `use_orig_params=True`.'
            )


def _make_train_and_log_config(
        cfg: DictConfig) -> Tuple[DictConfig, TrainConfig]:
    # Resolve all interpolation variables as early as possible
    unstructured_config = om.to_container(cfg, resolve=True)
    assert isinstance(unstructured_config, dict)
    assert all(isinstance(k, str) for k in unstructured_config.keys())
    unstructured_config = {str(k): v for k, v in unstructured_config.items()}

    # Structured config does not support unions of containers, so separate single and plural containers
    if (loader := unstructured_config.get('eval_loader', None)) is not None:
        forbid_config_key(unstructured_config, 'eval_loaders')
        if isinstance(loader, list):
            unstructured_config['eval_loaders'] = unstructured_config.pop(
                'eval_loader')
    if (tasks := unstructured_config.get('icl_tasks', None)) is not None:
        forbid_config_key(unstructured_config, 'icl_tasks_str')
        if isinstance(tasks, str):
            if 'icl_tasks_str' in unstructured_config:
                raise ValueError(
                    'Only one of `icl_tasks` or `icl_tasks_str` should be provided.'
                )
            unstructured_config['icl_tasks_str'] = unstructured_config.pop(
                'icl_tasks')
    if (gauntlet := unstructured_config.get('eval_gauntlet', None)) is not None:
        forbid_config_key(unstructured_config, 'eval_gauntlet_str')
        if isinstance(gauntlet, str):
            if 'eval_gauntlet_str' in unstructured_config:
                raise ValueError(
                    'Only one of `eval_gauntlet` or `eval_gauntlet_str` should be provided.'
                )
            unstructured_config['eval_gauntlet_str'] = unstructured_config.pop(
                'eval_gauntlet')

    arg_config_keys = set(unstructured_config.keys())
    extraneous_keys = set.difference(arg_config_keys, TRAIN_CONFIG_KEYS)

    if 'variables' not in unstructured_config:
        unstructured_config['variables'] = {}

    for key in extraneous_keys:
        warnings.warn(
            f'Unused parameter {key} found in cfg. Please check your yaml to ensure this parameter is necessary. Interpreting {key} as a variable for logging purposes. Top-level variables are deprecated and will not be supported in future releases.',
            DeprecationWarning)
        # TODO (milo): delete the below line once we deprecate variables at the top level.
        unstructured_config['variables'][key] = unstructured_config.pop(key)

    # Create copy of config for logging
    logged_cfg: DictConfig = copy.deepcopy(DictConfig(unstructured_config))

    # Get global and device batch size information from distributed/single node setting
    unstructured_config = update_batch_size_info(unstructured_config)
    logged_cfg.update(unstructured_config, merge=True)

    train_cfg: TrainConfig = om.structured(
        TrainConfig(**unstructured_config)
    )  # type: ignore (TrainConfig does expect arguments, the type checker is wrong here)
    return logged_cfg, train_cfg


def _log_num_params(model: ComposerModel, logged_cfg: DictConfig):
    # Log number of parameters
    if hasattr(model, 'n_total_params'):
        n_params = model.n_total_params
        n_trainable_params = n_params  # TODO: we currently assume all parameters are trainable.
    else:
        n_params = sum(p.numel() for p in model.parameters())
        n_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
    if hasattr(model, 'n_active_params'):
        n_active_params = model.n_active_params
    else:
        n_active_params = n_params
    logged_cfg.update({
        'n_params': n_params,
        'n_active_params': n_active_params,
        'n_trainable_params': n_trainable_params,
    })


def main(cfg: DictConfig) -> Trainer:
    logged_cfg, train_cfg = _make_train_and_log_config(cfg)

    code_paths = train_cfg.code_paths if train_cfg.code_paths else []
    # Import any user provided code
    for code_path in code_paths:
        import_file(code_path)

    # Filter deprecation warning from torch internal usage
    warnings.filterwarnings(
        action='ignore',
        category=UserWarning,
        message=
        'torch.distributed.*_base is a private function and will be deprecated.*'
    )

    # Check for incompatibilities between the model and data loaders
    validate_config(train_cfg)

    cuda_alloc_conf = []
    # Get max split size mb
    max_split_size_mb: Optional[int] = train_cfg.max_split_size_mb
    if max_split_size_mb is not None:
        cuda_alloc_conf.append(f'max_split_size_mb:{max_split_size_mb}')

    # Expandable segments
    if train_cfg.expandable_segments:
        cuda_alloc_conf.append('expandable_segments:True')

    if len(cuda_alloc_conf) > 0:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = ','.join(cuda_alloc_conf)

    # Set CUDA lazy loading
    # This can save a bit of memory if not all modules are needed
    cuda_load_lazy: bool = train_cfg.cuda_load_lazy
    if cuda_load_lazy:
        os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

    # Set seed first
    seed: int = train_cfg.seed
    reproducibility.seed_all(seed)

    # Initialize pytorch distributed training process groups
    dist_timeout: Union[int, float] = train_cfg.dist_timeout
    dist.initialize_dist(get_device(None), timeout=dist_timeout)

    # Mandatory model training configs
    model_config = train_cfg.model
    train_loader_config = train_cfg.train_loader

    # Optional fsdp data, fine-tuning, and eval configs
    fsdp_config: Optional[Dict[str, Any]] = train_cfg.fsdp_config

    eval_loader_config = train_cfg.eval_loader if train_cfg.eval_loader is not None else train_cfg.eval_loaders
    icl_tasks_config = train_cfg.icl_tasks
    eval_gauntlet_config = train_cfg.eval_gauntlet

    # Optional parameters will be set to default values if not specified.
    default_run_name: str = os.environ.get('RUN_NAME', 'llm')
    run_name: str = train_cfg.run_name if train_cfg.run_name else default_run_name
    is_state_dict_sharded: bool = (fsdp_config.get('state_dict_type', 'full')
                                   == 'sharded') if fsdp_config else False
    save_latest_filename: str = train_cfg.save_latest_filename if train_cfg.save_latest_filename else 'latest-sharded-rank{rank}' if is_state_dict_sharded else 'latest-rank{rank}.pt'
    save_filename: str = train_cfg.save_filename if train_cfg.save_filename else 'ep{epoch}-ba{batch}-rank{rank}.pt'

    # Enable autoresume from model checkpoints if possible
    autoresume_default: bool = False
    if logged_cfg.get('run_name', None) is not None \
        and train_cfg.save_folder is not None \
        and not train_cfg.save_overwrite \
        and not train_cfg.save_weights_only:
        autoresume_default = True

    if not train_cfg.autoresume and autoresume_default:
        log.info('As run_name, save_folder, and save_latest_filename are set, \
                changing autoresume default to True...')

    # Warn if fsdp is enabled but user only has 1 GPU
    if dist.get_world_size() == 1 and fsdp_config is not None:
        warnings.warn(
            'FSDP is not applicable for single-GPU training. Reverting to DDP.')
        fsdp_config = None

    # set logging level
    if train_cfg.python_log_level is not None:
        logging.basicConfig(
            # Example of format string
            # 2022-06-29 11:22:26,152: rank0[822018][MainThread]: INFO: Message here
            format=
            f'%(asctime)s: rank{dist.get_global_rank()}[%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s'
        )
        logging.getLogger('llmfoundry').setLevel(
            train_cfg.python_log_level.upper())  # Foundry module
        logging.getLogger(__name__).setLevel(
            train_cfg.python_log_level.upper())  # Train script

    # Initialize context
    init_context = process_init_device(model_config, fsdp_config)
    logged_cfg.update({'fsdp_config': fsdp_config}, merge=True)

    # Build tokenizer
    log.info('Building tokenizer...')
    tokenizer_name = train_cfg.tokenizer['name']
    tokenizer_kwargs = train_cfg.tokenizer.get('kwargs', {})
    tokenizer = build_tokenizer(tokenizer_name, tokenizer_kwargs)

    # Scheduler
    scheduler_name: str = train_cfg.scheduler.pop('name')
    scheduler = build_scheduler(scheduler_name, train_cfg.scheduler)

    # Loggers
    loggers = [
        build_logger(str(name), logger_cfg)
        for name, logger_cfg in train_cfg.loggers.items()
    ] if train_cfg.loggers else []

    mosaicml_logger = find_mosaicml_logger(loggers)
    if mosaicml_logger is None:
        mosaicml_logger = maybe_create_mosaicml_logger()
        if mosaicml_logger is not None:
            # mosaicml_logger will be None if run isn't on MosaicML platform
            loggers.append(mosaicml_logger)

    if train_cfg.metadata is not None:
        # Flatten the metadata for logging
        logged_cfg.pop('metadata', None)
        logged_cfg.update(train_cfg.metadata, merge=True)
        if mosaicml_logger is not None:
            mosaicml_logger.log_metrics(train_cfg.metadata)
            mosaicml_logger._flush_metadata(force_flush=True)

    # Profiling
    profiler: Optional[Profiler] = None
    profiler_cfg = train_cfg.profiler
    if profiler_cfg:
        profiler_schedule_cfg: Dict = pop_config(profiler_cfg,
                                                 'schedule',
                                                 must_exist=True)
        profiler_schedule = cyclic_schedule(**profiler_schedule_cfg)
        # Only support json trace handler
        profiler_trace_handlers: List[TraceHandler] = []
        profiler_trace_cfg: Optional[Dict] = pop_config(profiler_cfg,
                                                        'json_trace_handler',
                                                        must_exist=False,
                                                        default_value=None)
        if profiler_trace_cfg:
            profiler_trace_handlers.append(
                JSONTraceHandler(**profiler_trace_cfg))
        profiler = Profiler(**profiler_cfg,
                            trace_handlers=profiler_trace_handlers,
                            schedule=profiler_schedule)

    # Callbacks
    callbacks: List[Callback] = [
        build_callback(str(name), callback_cfg, logged_cfg)
        for name, callback_cfg in train_cfg.callbacks.items()
    ] if train_cfg.callbacks else []

    use_async_eval = any(isinstance(c, AsyncEval) for c in callbacks)

    # Algorithms
    algorithms = [
        build_algorithm(str(name), algorithm_cfg)
        for name, algorithm_cfg in train_cfg.algorithms.items()
    ] if train_cfg.algorithms else None

    # Dataloaders
    log.info('Building train loader...')
    try:
        train_loader = build_dataloader(
            train_loader_config,
            tokenizer,
            train_cfg.device_train_batch_size,
        )
    except Exception as e:
        if mosaicml_logger is not None:
            mosaicml_logger.log_exception(e)
        raise e

    if mosaicml_logger is not None:
        mosaicml_logger.log_metrics({'data_validated': time.time()})

    ## Evaluation
    if use_async_eval:
        evaluators = []
        if train_cfg.eval_first:
            warnings.warn(
                'AsyncEval callback does not support eval_first=True. Ignoring.'
            )
            train_cfg.eval_first = False

    else:
        log.info('Building eval loader...')
        eval_icl_seq_len: int = train_cfg.icl_seq_len if train_cfg.icl_seq_len else train_cfg.max_seq_len
        evaluators, _, eval_gauntlet_callback = build_evaluators(
            eval_loader_config,
            icl_tasks_config,
            eval_gauntlet_config,
            tokenizer=tokenizer,
            device_eval_batch_size=train_cfg.device_eval_batch_size,
            icl_seq_len=eval_icl_seq_len,
            icl_subset_num_batches=train_cfg.icl_subset_num_batches,
        )
        if eval_gauntlet_callback is not None:
            callbacks.append(eval_gauntlet_callback)

    if mosaicml_logger is not None:
        log_train_analytics(mosaicml_logger, model_config, train_loader_config,
                            eval_loader_config, train_cfg.callbacks,
                            tokenizer_name, train_cfg.load_path,
                            icl_tasks_config, eval_gauntlet_config)
    # Build Model
    log.info('Initializing model...')
    name = model_config.pop('name')
    model = build_composer_model(
        name=name,
        tokenizer=tokenizer,
        init_context=init_context,
        master_weights_dtype=model_config.get('master_weights_dtype', None),
        cfg=model_config,
    )

    _log_num_params(model, logged_cfg)

    # Optimizer
    optimizer_name: str = train_cfg.optimizer.pop('name')
    optimizer = build_optimizer(model, optimizer_name, train_cfg.optimizer)

    # Now add the eval metrics
    try:
        if eval_loader_config is not None and not use_async_eval:
            eval_metrics = model.get_metrics(is_train=False)
            non_icl_metrics = [
                metric_name for metric_name, metric in eval_metrics.items()
                if not isinstance(metric, InContextLearningMetric)
            ]
            evaluators = add_metrics_to_eval_loaders(evaluators,
                                                     non_icl_metrics)
    except Exception as e:
        if mosaicml_logger is not None:
            mosaicml_logger.log_exception(e)
        raise e

    # Build the Trainer
    log.info('Building trainer...')
    trainer = Trainer(
        run_name=run_name,
        seed=seed,
        model=model,
        train_dataloader=train_loader,
        eval_dataloader=evaluators,
        optimizers=optimizer,
        schedulers=scheduler,
        max_duration=train_cfg.max_duration,
        eval_interval=train_cfg.eval_interval,
        eval_subset_num_batches=train_cfg.eval_subset_num_batches,
        progress_bar=train_cfg.progress_bar,
        log_to_console=train_cfg.log_to_console,
        console_log_interval=train_cfg.console_log_interval,
        loggers=loggers,
        callbacks=callbacks,
        precision=train_cfg.precision,
        algorithms=algorithms,
        device_train_microbatch_size=train_cfg.device_train_microbatch_size,
        fsdp_config=fsdp_config,
        save_folder=train_cfg.save_folder,
        save_filename=save_filename,
        save_latest_filename=save_latest_filename,
        save_interval=train_cfg.save_interval,
        save_num_checkpoints_to_keep=train_cfg.save_num_checkpoints_to_keep,
        save_overwrite=train_cfg.save_overwrite,
        save_weights_only=train_cfg.save_weights_only,
        load_path=train_cfg.load_path,
        load_weights_only=train_cfg.load_weights_only,
        load_strict_model_weights=train_cfg.load_strict_model_weights,
        load_ignore_keys=train_cfg.load_ignore_keys,
        save_ignore_keys=train_cfg.save_ignore_keys,
        autoresume=train_cfg.autoresume,
        python_log_level=train_cfg.python_log_level,
        dist_timeout=dist_timeout,
        profiler=profiler,
        compile_config=train_cfg.compile_config,
    )

    if train_cfg.log_config:
        log.info('Logging config')
        log_config(logged_cfg)
    torch.cuda.empty_cache()
    gc.collect()

    # Eval first if requested
    if train_cfg.eval_first and trainer.state.timestamp.batch.value == 0:
        trainer.eval()

    log.info('Starting training...')
    trainer.fit()

    log.info('Done.')
    return trainer


if __name__ == '__main__':
    yaml_path, args_list = sys.argv[1], sys.argv[2:]

    # Disable resolving environment variables through omegaconf.
    om.clear_resolver('oc.env')

    # Load yaml and cli arguments.
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg = om.merge(yaml_cfg, cli_cfg)
    om.resolve(cfg)
    assert isinstance(cfg, DictConfig)
    main(cfg)
