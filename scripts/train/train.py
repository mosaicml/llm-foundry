# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
import copy
import gc
import logging
import os
import sys
import time
import warnings
from typing import Any, Dict, List, Optional, Union

import torch
import torch.distributed
from composer import Trainer
from composer.core.callback import Callback
from composer.profiler import (JSONTraceHandler, Profiler, TraceHandler,
                               cyclic_schedule)
from composer.utils import dist, get_device, reproducibility
from omegaconf import DictConfig, ListConfig
from omegaconf import OmegaConf as om
from rich.traceback import install

from llmfoundry.eval.metrics.nlp import InContextLearningMetric
from llmfoundry.utils import (find_mosaicml_logger, log_train_analytics,
                              maybe_create_mosaicml_logger)

install()
from llmfoundry.callbacks import AsyncEval
from llmfoundry.data.dataloader import build_dataloader
from llmfoundry.layers_registry import ffns_with_megablocks
from llmfoundry.utils.builders import (add_metrics_to_eval_loaders,
                                       build_algorithm, build_callback,
                                       build_composer_model, build_evaluators,
                                       build_logger, build_optimizer,
                                       build_scheduler, build_tokenizer)
from llmfoundry.utils.config_utils import (log_config, pop_config,
                                           process_init_device,
                                           update_batch_size_info)
from llmfoundry.utils.registry_utils import import_file

log = logging.getLogger(__name__)


def validate_config(cfg: DictConfig):
    """Validates compatible model and dataloader selection."""
    loaders = [cfg.train_loader]
    if 'eval_loader' in cfg:
        eval_loader = cfg.eval_loader
        if isinstance(eval_loader, ListConfig):
            for loader in eval_loader:
                if loader.label is None:
                    raise ValueError(
                        'When specifying multiple evaluation datasets, each one must include the \
                            `label` attribute.')
                loaders.append(loader)
        else:
            loaders.append(eval_loader)
    for loader in loaders:
        if loader.name == 'text':
            if cfg.model.name == 'hf_t5':
                raise ValueError(
                    f'Model type "{cfg.model.name}" is not supported when using the "text " ' +\
                    f'dataloader. Only finetuning is supported.')

    if 'icl_tasks' in cfg:
        if cfg.model.name == 'hf_t5':
            raise ValueError(
                'ICL evaluation does not currently support Encoder-Decoder models, such as "hf_t5".'
            )

    if (cfg.model.get('fc_type', 'torch') != 'te' and 'te' not in cfg.model.get(
            'ffn_config', {}).get('ffn_type', 'mptmlp') and
            'fp8' in cfg.precision):
        warnings.warn(
            "fp8 only supported for te.Linear layers. Either set `cfg.model.fc_typ='te'` or "
            +
            "`cfg.model.ffn_config.ffn_type='te_ln_mlp'` to enable layers using fp8 precision."
        )

    if (cfg.model.get('fc_type', 'torch') == 'te' or
            'te' in cfg.model.get('ffn_config', {}).get('ffn_type', 'mptmlp')):
        fsdp_config = cfg.get('fsdp_config', None)
        act_ckpt = fsdp_config.get('activation_checkpointing', False)
        act_ckpt_reentrant = fsdp_config.get(
            'activation_checkpointing_reentrant', False)
        if fsdp_config is not None and act_ckpt == True and act_ckpt_reentrant == True:
            warnings.warn(
                '`te.Linear` layers do not support activation_checkpointing with '
                + '`activation_checkpointing_reentrant = True`. ' +
                'Setting cfg.fsdp_config.activation_checkpointing_reentrant=False.'
            )
            cfg.fsdp_config.activation_checkpointing_reentrant = False

    if cfg.model.get('ffn_config', {}).get('ffn_type', 'mptmlp') == 'te_ln_mlp':
        warnings.warn(
            '`te.LayerNormMLP` requires has issues with torch._dynamo. ' +
            'Setting `torch._dynamo.config.suppress_errors = True` and falling back to eager.'
        )
        torch._dynamo.config.suppress_errors = True  # type: ignore (third-party)

    if cfg.model.get('load_in_8bit', False):
        raise ValueError(
            '`load_in_8bit` is only supported for evaluation rather than training.'
        )

    if cfg.model.get('ffn_config', {}).get('ffn_type',
                                           'mptmlp') in ffns_with_megablocks:
        moe_world_size = cfg.model.get('ffn_config',
                                       {}).get('moe_world_size', 1)
        use_orig_params = cfg.get('fsdp_config',
                                  {}).get('use_orig_params', True)
        if moe_world_size > 1 and not use_orig_params:
            raise ValueError(
                f'MoEs with expert parallelism (moe_world_size {moe_world_size} > 1) require `use_orig_params=True`.'
            )


def _initialize_dist_with_barrier(dist_timeout: Union[int, float]):
    """Initialize distributed and test setup with a barrier.

    Args:
        dist_timeout (Union[int, float]): Timeout for initializing the process group
    """
    log.debug('Initializing dist with device...')
    dist.initialize_dist(get_device(None), timeout=dist_timeout)
    log.debug('Testing barrier with device...')
    dist.barrier()
    log.debug('Barrier test passed with device.')


def main(cfg: DictConfig) -> Trainer:
    # Run user provided code if specified
    code_paths = pop_config(cfg,
                            'code_paths',
                            must_exist=False,
                            default_value=[],
                            convert=True)
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
    validate_config(cfg)

    # Resolve all interpolation variables as early as possible
    om.resolve(cfg)

    # Create copy of config for logging
    logged_cfg: DictConfig = copy.deepcopy(cfg)

    cuda_alloc_conf = []
    # Get max split size mb
    max_split_size_mb: Optional[int] = cfg.pop('max_split_size_mb', None)
    if max_split_size_mb is not None:
        cuda_alloc_conf.append(f'max_split_size_mb:{max_split_size_mb}')

    # Expandable segments
    if cfg.pop('expandable_segments', False):
        cuda_alloc_conf.append('expandable_segments:True')

    if len(cuda_alloc_conf) > 0:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = ','.join(cuda_alloc_conf)

    # Set CUDA lazy loading
    # This can save a bit of memory if not all modules are needed
    cuda_load_lazy: bool = cfg.pop('cuda_load_lazy', False)
    if cuda_load_lazy:
        os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

    # Set seed first
    seed: int = pop_config(cfg, 'seed', must_exist=True)
    reproducibility.seed_all(seed)

    # Initialize pytorch distributed training process groups
    dist_timeout: Union[int, float] = pop_config(cfg,
                                                 'dist_timeout',
                                                 must_exist=False,
                                                 default_value=600.0)
    python_log_level: Optional[str] = pop_config(cfg,
                                                 'python_log_level',
                                                 must_exist=False,
                                                 default_value='debug')
    # Set logging level
    if python_log_level is not None:
        logging.basicConfig(
            # Example of format string
            # 2022-06-29 11:22:26,152: rank0[822018][MainThread]: INFO: Message here
            format=
            f'%(asctime)s: rank{dist.get_global_rank()}[%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s'
        )
        logging.getLogger('llmfoundry').setLevel(
            python_log_level.upper())  # Foundry module
        logging.getLogger(__name__).setLevel(
            python_log_level.upper())  # Train script

    _initialize_dist_with_barrier(dist_timeout=dist_timeout)

    # Get global and device batch size information from distributed/single node setting
    cfg = update_batch_size_info(cfg)
    logged_cfg.update(cfg, merge=True)

    # Mandatory model training configs
    model_config: DictConfig = pop_config(cfg, 'model', must_exist=True)
    tokenizer_config: Dict[str, Any] = pop_config(cfg,
                                                  'tokenizer',
                                                  must_exist=True,
                                                  convert=True)
    optimizer_config: Dict[str, Any] = pop_config(cfg,
                                                  'optimizer',
                                                  must_exist=True,
                                                  convert=True)
    scheduler_config: Dict[str, Any] = pop_config(cfg,
                                                  'scheduler',
                                                  must_exist=True,
                                                  convert=True)
    train_loader_config: DictConfig = pop_config(cfg,
                                                 'train_loader',
                                                 must_exist=True)

    # Optional fsdp data, fine-tuning, and eval configs
    fsdp_config: Optional[Dict[str, Any]] = pop_config(cfg,
                                                       'fsdp_config',
                                                       must_exist=False,
                                                       default_value=None,
                                                       convert=True)
    eval_loader_config: Optional[Union[DictConfig, ListConfig]] = pop_config(
        cfg, 'eval_loader', must_exist=False, default_value=None)
    icl_tasks_config: Optional[Union[ListConfig,
                                     str]] = pop_config(cfg,
                                                        'icl_tasks',
                                                        must_exist=False,
                                                        default_value=None)
    eval_gauntlet_config: Optional[Union[DictConfig,
                                         str]] = pop_config(cfg,
                                                            'eval_gauntlet',
                                                            must_exist=False,
                                                            default_value=None)
    icl_subset_num_batches: Optional[int] = pop_config(cfg,
                                                       'icl_subset_num_batches',
                                                       must_exist=False,
                                                       default_value=None)
    icl_seq_len: Optional[int] = pop_config(cfg,
                                            'icl_seq_len',
                                            must_exist=False,
                                            default_value=None)
    # Optional logging, evaluation and callback configs
    logger_configs: Optional[DictConfig] = pop_config(cfg,
                                                      'loggers',
                                                      must_exist=False,
                                                      default_value=None,
                                                      convert=True)
    callback_configs: Optional[DictConfig] = pop_config(cfg,
                                                        'callbacks',
                                                        must_exist=False,
                                                        default_value=None,
                                                        convert=True)
    algorithm_configs: Optional[DictConfig] = pop_config(cfg,
                                                         'algorithms',
                                                         must_exist=False,
                                                         default_value=None)

    # Mandatory hyperparameters for training
    device_train_batch_size: int = pop_config(cfg,
                                              'device_train_batch_size',
                                              must_exist=True)
    device_eval_batch_size: int = pop_config(cfg,
                                             'device_eval_batch_size',
                                             must_exist=True)
    max_duration: Union[int, str] = pop_config(cfg,
                                               'max_duration',
                                               must_exist=True)
    eval_interval: Union[int, str] = pop_config(cfg,
                                                'eval_interval',
                                                default_value=1,
                                                must_exist=False)
    precision: str = pop_config(cfg, 'precision', must_exist=True)
    max_seq_len: int = pop_config(cfg, 'max_seq_len', must_exist=True)

    # Optional parameters will be set to default values if not specified.
    default_run_name: str = os.environ.get('RUN_NAME', 'llm')
    run_name: str = pop_config(cfg,
                               'run_name',
                               must_exist=False,
                               default_value=default_run_name)
    save_folder: Optional[str] = pop_config(cfg,
                                            'save_folder',
                                            must_exist=False,
                                            default_value=None)
    is_state_dict_sharded: bool = (fsdp_config.get('state_dict_type', 'full')
                                   == 'sharded') if fsdp_config else False
    save_latest_filename: str = pop_config(
        cfg,
        'save_latest_filename',
        must_exist=False,
        default_value='latest-sharded-rank{rank}'
        if is_state_dict_sharded else 'latest-rank{rank}.pt')
    save_overwrite: bool = pop_config(cfg,
                                      'save_overwrite',
                                      must_exist=False,
                                      default_value=False)
    save_weights_only: bool = pop_config(cfg,
                                         'save_weights_only',
                                         must_exist=False,
                                         default_value=False)
    save_filename: str = pop_config(
        cfg,
        'save_filename',
        must_exist=False,
        default_value='ep{epoch}-ba{batch}-rank{rank}.pt')
    save_interval: Union[str, int] = pop_config(cfg,
                                                'save_interval',
                                                must_exist=False,
                                                default_value='1000ba')
    save_num_checkpoints_to_keep: int = pop_config(
        cfg, 'save_num_checkpoints_to_keep', must_exist=False, default_value=-1)
    progress_bar = pop_config(cfg,
                              'progress_bar',
                              must_exist=False,
                              default_value=False)
    log_to_console: bool = pop_config(cfg,
                                      'log_to_console',
                                      must_exist=False,
                                      default_value=True)
    console_log_interval: Union[int, str] = pop_config(cfg,
                                                       'console_log_interval',
                                                       must_exist=False,
                                                       default_value='1ba')
    device_train_microbatch_size: Union[str, int] = pop_config(
        cfg,
        'device_train_microbatch_size',
        must_exist=False,
        default_value='auto')
    eval_subset_num_batches: int = pop_config(cfg,
                                              'eval_subset_num_batches',
                                              must_exist=False,
                                              default_value=-1)
    eval_first: bool = pop_config(cfg,
                                  'eval_first',
                                  must_exist=False,
                                  default_value=False)
    load_path: str = pop_config(cfg,
                                'load_path',
                                must_exist=False,
                                default_value=None)
    load_weights_only: bool = pop_config(cfg,
                                         'load_weights_only',
                                         must_exist=False,
                                         default_value=False)
    load_strict_model_weights: bool = pop_config(cfg,
                                                 'load_strict_model_weights',
                                                 must_exist=False,
                                                 default_value=True)
    load_ignore_keys: Optional[List[str]] = pop_config(cfg,
                                                       'load_ignore_keys',
                                                       must_exist=False,
                                                       default_value=None)
    save_ignore_keys: Optional[List[str]] = pop_config(cfg,
                                                       'save_ignore_keys',
                                                       must_exist=False,
                                                       default_value=None)
    compile_config: Optional[Dict[str, Any]] = pop_config(cfg,
                                                          'compile_config',
                                                          must_exist=False,
                                                          default_value=None)
    metadata: Optional[Dict[str, str]] = pop_config(cfg,
                                                    'metadata',
                                                    must_exist=False,
                                                    default_value=None,
                                                    convert=True)
    should_log_config: bool = pop_config(cfg,
                                         'log_config',
                                         must_exist=False,
                                         default_value=True)

    # Enable autoresume from model checkpoints if possible
    autoresume_default: bool = False
    if logged_cfg.get('run_name', None) is not None \
        and save_folder is not None \
        and not save_overwrite \
        and not save_weights_only:
        autoresume_default = True

    if cfg.get('autoresume') is None and autoresume_default:
        log.info('As run_name, save_folder, and save_latest_filename are set, \
                changing autoresume default to True...')

    autoresume: bool = pop_config(cfg,
                                  'autoresume',
                                  must_exist=False,
                                  default_value=autoresume_default)

    # Pop known unused parameters that are used as interpolation variables or
    # created by update_batch_size_info.
    pop_config(cfg, 'data_local', must_exist=False)
    pop_config(cfg, 'data_remote', must_exist=False)
    pop_config(cfg, 'global_seed', must_exist=False)
    pop_config(cfg, 'global_train_batch_size', must_exist=False)
    pop_config(cfg, 'n_gpus', must_exist=False)
    pop_config(cfg, 'device_train_grad_accum', must_exist=False)

    # Warn users for unused parameters
    for key in cfg:
        warnings.warn(
            f'Unused parameter {key} found in cfg. Please check your yaml to ensure this parameter is necessary.'
        )

    # Warn if fsdp is enabled but user only has 1 GPU
    if dist.get_world_size() == 1 and fsdp_config is not None:
        warnings.warn(
            'FSDP is not applicable for single-GPU training. Reverting to DDP.')
        fsdp_config = None

    # Initialize context
    init_context = process_init_device(model_config, fsdp_config)
    logged_cfg.update({'fsdp_config': fsdp_config}, merge=True)

    # Build tokenizer
    log.info('Building tokenizer...')
    tokenizer_name = tokenizer_config['name']
    tokenizer_kwargs = tokenizer_config.get('kwargs', {})
    tokenizer = build_tokenizer(tokenizer_name, tokenizer_kwargs)

    # Scheduler
    scheduler_name: str = scheduler_config.pop('name')
    scheduler = build_scheduler(scheduler_name, scheduler_config)

    # Loggers
    loggers = [
        build_logger(str(name), logger_cfg)
        for name, logger_cfg in logger_configs.items()
    ] if logger_configs else []

    mosaicml_logger = find_mosaicml_logger(loggers)
    if mosaicml_logger is None:
        mosaicml_logger = maybe_create_mosaicml_logger()
        if mosaicml_logger is not None:
            # mosaicml_logger will be None if run isn't on MosaicML platform
            loggers.append(mosaicml_logger)

    if metadata is not None:
        # Flatten the metadata for logging
        logged_cfg.pop('metadata', None)
        logged_cfg.update(metadata, merge=True)
        if mosaicml_logger is not None:
            mosaicml_logger.log_metrics(metadata)
            mosaicml_logger._flush_metadata(force_flush=True)

    # Profiling
    profiler: Optional[Profiler] = None
    profiler_cfg: Optional[DictConfig] = pop_config(cfg,
                                                    'profiler',
                                                    must_exist=False,
                                                    convert=False,
                                                    default_value=None)
    if profiler_cfg:
        profiler_schedule_cfg: Dict = pop_config(profiler_cfg,
                                                 'schedule',
                                                 must_exist=True,
                                                 convert=True)
        profiler_schedule = cyclic_schedule(**profiler_schedule_cfg)
        # Only support json trace handler
        profiler_trace_handlers: List[TraceHandler] = []
        profiler_trace_cfg: Optional[Dict] = pop_config(profiler_cfg,
                                                        'json_trace_handler',
                                                        must_exist=False,
                                                        default_value=None,
                                                        convert=True)
        if profiler_trace_cfg:
            profiler_trace_handlers.append(
                JSONTraceHandler(**profiler_trace_cfg))
        profiler = Profiler(**profiler_cfg,
                            trace_handlers=profiler_trace_handlers,
                            schedule=profiler_schedule)

    # Callbacks
    callbacks: List[Callback] = [
        build_callback(str(name), callback_cfg, om.to_container(logged_cfg))
        for name, callback_cfg in callback_configs.items()
    ] if callback_configs else []

    use_async_eval = any(isinstance(c, AsyncEval) for c in callbacks)

    # Algorithms
    algorithms = [
        build_algorithm(str(name), algorithm_cfg)
        for name, algorithm_cfg in algorithm_configs.items()
    ] if algorithm_configs else None

    # Dataloaders
    log.info('Building train loader...')
    try:
        train_loader = build_dataloader(
            train_loader_config,
            tokenizer,
            device_train_batch_size,
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
        if eval_first:
            warnings.warn(
                'AsyncEval callback does not support eval_first=True. Ignoring.'
            )
            eval_first = False

    else:
        log.info('Building eval loader...')
        eval_icl_seq_len: int = icl_seq_len if icl_seq_len else max_seq_len
        evaluators, _, eval_gauntlet_callback = build_evaluators(
            eval_loader_config,
            icl_tasks_config,
            eval_gauntlet_config,
            tokenizer=tokenizer,
            device_eval_batch_size=device_eval_batch_size,
            icl_seq_len=eval_icl_seq_len,
            icl_subset_num_batches=icl_subset_num_batches,
        )
        if eval_gauntlet_callback is not None:
            callbacks.append(eval_gauntlet_callback)

    if mosaicml_logger is not None:
        log_train_analytics(mosaicml_logger, model_config, train_loader_config,
                            eval_loader_config, callback_configs,
                            tokenizer_name, load_path, icl_tasks_config,
                            eval_gauntlet_config)
    # Build Model
    log.info('Initializing model...')
    model = build_composer_model(
        name=model_config.name,
        cfg=model_config,
        tokenizer=tokenizer,
        init_context=init_context,
        master_weights_dtype=model_config.get('master_weights_dtype', None),
    )

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

    # Optimizer
    optimizer_name: str = optimizer_config.pop('name')
    optimizer = build_optimizer(model, optimizer_name, optimizer_config)

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
        max_duration=max_duration,
        eval_interval=eval_interval,
        eval_subset_num_batches=eval_subset_num_batches,
        progress_bar=progress_bar,
        log_to_console=log_to_console,
        console_log_interval=console_log_interval,
        loggers=loggers,
        callbacks=callbacks,
        precision=precision,
        algorithms=algorithms,
        device_train_microbatch_size=device_train_microbatch_size,
        fsdp_config=fsdp_config,
        save_folder=save_folder,
        save_filename=save_filename,
        save_latest_filename=save_latest_filename,
        save_interval=save_interval,
        save_num_checkpoints_to_keep=save_num_checkpoints_to_keep,
        save_overwrite=save_overwrite,
        save_weights_only=save_weights_only,
        load_path=load_path,
        load_weights_only=load_weights_only,
        load_strict_model_weights=load_strict_model_weights,
        load_ignore_keys=load_ignore_keys,
        save_ignore_keys=save_ignore_keys,
        autoresume=autoresume,
        python_log_level=python_log_level,
        dist_timeout=dist_timeout,
        profiler=profiler,
        compile_config=compile_config,
    )

    if should_log_config:
        log.info('Logging config')
        log_config(logged_cfg)
    torch.cuda.empty_cache()
    gc.collect()

    # Eval first if requested
    if eval_first and trainer.state.timestamp.batch.value == 0:
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
