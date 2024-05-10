# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
import gc
import logging
import os
import sys
import time
import warnings
from typing import Any, Dict, List, Optional, Union

import torch
import torch.distributed
from composer import ComposerModel, Trainer
from composer.core.callback import Callback
from composer.profiler import (
    JSONTraceHandler,
    Profiler,
    TraceHandler,
    cyclic_schedule,
)
from composer.utils import dist, get_device, reproducibility
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from rich.traceback import install

from llmfoundry.eval.metrics.nlp import InContextLearningMetric
from llmfoundry.utils import (
    find_mosaicml_logger,
    log_train_analytics,
    maybe_create_mosaicml_logger,
)

install()

from llmfoundry.callbacks import AsyncEval
from llmfoundry.data.dataloader import build_dataloader
from llmfoundry.layers_registry import ffns_with_megablocks
from llmfoundry.utils.builders import (
    add_metrics_to_eval_loaders,
    build_algorithm,
    build_callback,
    build_composer_model,
    build_evaluators,
    build_logger,
    build_optimizer,
    build_scheduler,
    build_tokenizer,
)
from llmfoundry.utils.config_utils import (
    TRAIN_CONFIG_KEYS,
    TrainConfig,
    log_config,
    make_dataclass_and_log_config,
    pop_config,
    process_init_device,
    update_batch_size_info,
)
from llmfoundry.utils.registry_utils import import_file

log = logging.getLogger(__name__)


def validate_config(train_config: TrainConfig):
    """Validates compatible model and dataloader selection."""
    # Validate the rest of the config
    loaders = [train_config.train_loader]
    if train_config.eval_loaders is not None:
        for loader in (train_config.eval_loaders or []):  # pyright
            if 'label' not in loader or loader['label'] is None:
                raise ValueError(
                    'When specifying multiple evaluation datasets, each one must include the \
                            `label` attribute.',
                )
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
                'ICL evaluation does not currently support Encoder-Decoder models, such as "hf_t5".',
            )

    if (
        train_config.model.get('fc_type', 'torch') != 'te' and
        'te' not in train_config.model.get('ffn_config',
                                           {}).get('ffn_type', 'mptmlp') and
        'fp8' in train_config.precision
    ):
        warnings.warn(
            "fp8 only supported for te.Linear layers. Either set `cfg.model.fc_typ='te'` or "
            +
            "`cfg.model.ffn_config.ffn_type='te_ln_mlp'` to enable layers using fp8 precision.",
        )

    if (
        train_config.model.get('fc_type', 'torch') == 'te' or 'te'
        in train_config.model.get('ffn_config', {}).get('ffn_type', 'mptmlp')
    ):
        fsdp_config = train_config.fsdp_config
        act_ckpt = fsdp_config.get(
            'activation_checkpointing',
            False,
        ) if fsdp_config else False
        act_ckpt_reentrant = fsdp_config.get(
            'activation_checkpointing_reentrant',
            False,
        ) if fsdp_config else False
        if fsdp_config is not None and act_ckpt == True and act_ckpt_reentrant == True:
            warnings.warn(
                '`te.Linear` layers do not support activation_checkpointing with '
                + '`activation_checkpointing_reentrant = True`. ' +
                'Setting cfg.fsdp_config.activation_checkpointing_reentrant=False.',
            )
            assert train_config.fsdp_config is not None  # pyright (this is known because fsdp_config is not None)
            train_config.fsdp_config['activation_checkpointing_reentrant'
                                    ] = False

    if train_config.model.get('ffn_config',
                              {}).get('ffn_type', 'mptmlp') == 'te_ln_mlp':
        warnings.warn(
            '`te.LayerNormMLP` requires has issues with torch._dynamo. ' +
            'Setting `torch._dynamo.config.suppress_errors = True` and falling back to eager.',
        )
        torch._dynamo.config.suppress_errors = True  # type: ignore (third-party)

    if train_config.model.get('load_in_8bit', False):
        raise ValueError(
            '`load_in_8bit` is only supported for evaluation rather than training.',
        )

    if train_config.model.get('ffn_config', {}).get(
        'ffn_type',
        'mptmlp',
    ) in ffns_with_megablocks:
        moe_world_size = train_config.model.get('ffn_config',
                                                {}).get('moe_world_size', 1)
        use_orig_params = train_config.fsdp_config.get(
            'use_orig_params',
            True,
        ) if train_config.fsdp_config is not None else True
        if moe_world_size > 1 and not use_orig_params:
            raise ValueError(
                f'MoEs with expert parallelism (moe_world_size {moe_world_size} > 1) require `use_orig_params=True`.',
            )

    attn_config = train_config.model.get('attn_config', None)
    if attn_config is not None:
        seq_parallel_world_size = attn_config.get(
            'seq_parallel_world_size',
            None,
        )
        if seq_parallel_world_size is not None:
            raise ValueError('Training does not support sequence parallelism.')


def _log_num_params(model: ComposerModel, logged_cfg: Dict[str, Any]):
    # Log number of parameters
    if hasattr(model, 'n_total_params'):
        n_params = model.n_total_params
        n_trainable_params = n_params  # TODO: we currently assume all parameters are trainable.
    else:
        n_params = sum(p.numel() for p in model.parameters())
        n_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
    if hasattr(model, 'n_active_params'):
        n_active_params = model.n_active_params
    else:
        n_active_params = n_params
    logged_cfg.update({
        'n_params': n_params,
        'n_active_params': n_active_params,
        'n_trainable_params': n_trainable_params,
    })


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
    code_paths = cfg.get('code_paths', [])
    # Import any user provided code
    for code_path in code_paths:
        import_file(code_path)

    logged_cfg, train_cfg = make_dataclass_and_log_config(
        cfg,
        TrainConfig,
        TRAIN_CONFIG_KEYS,
        transforms=[update_batch_size_info],
    )

    # Set logging level
    if train_cfg.python_log_level is not None:
        logging.basicConfig(
            # Example of format string
            # 2022-06-29 11:22:26,152: rank0[822018][MainThread]: INFO: Message here
            format=
            f'%(asctime)s: rank{dist.get_global_rank()}[%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s',
        )
        logging.getLogger('llmfoundry').setLevel(
            train_cfg.python_log_level.upper(),
        )  # Foundry module
        logging.getLogger(__name__).setLevel(
            train_cfg.python_log_level.upper(),
        )  # Train script

    _initialize_dist_with_barrier(dist_timeout=train_cfg.dist_timeout)

    # Filter deprecation warning from torch internal usage
    warnings.filterwarnings(
        action='ignore',
        category=UserWarning,
        message=
        'torch.distributed.*_base is a private function and will be deprecated.*',
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

    # Mandatory model training configs
    model_config = train_cfg.model
    train_loader_config = train_cfg.train_loader

    # Optional fsdp data, fine-tuning, and eval configs
    fsdp_config: Optional[Dict[str, Any]] = train_cfg.fsdp_config

    eval_loader_config = train_cfg.eval_loader if train_cfg.eval_loader is not None else train_cfg.eval_loaders
    icl_tasks_config = train_cfg.icl_tasks or train_cfg.icl_tasks_str
    eval_gauntlet_config = train_cfg.eval_gauntlet or train_cfg.eval_gauntlet_str

    # Optional parameters will be set to default values if not specified.
    default_run_name: str = os.environ.get('RUN_NAME', 'llm')
    run_name: str = train_cfg.run_name if train_cfg.run_name else default_run_name
    is_state_dict_sharded: bool = (
        fsdp_config.get('state_dict_type', 'full') == 'sharded'
    ) if fsdp_config else False
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
        log.info(
            'As run_name, save_folder, and save_latest_filename are set, \
                changing autoresume default to True...',
        )

    # Warn if fsdp is enabled but user only has 1 GPU
    if dist.get_world_size() == 1 and fsdp_config is not None:
        warnings.warn(
            'FSDP is not applicable for single-GPU training. Reverting to DDP.',
        )
        fsdp_config = None

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
        profiler_schedule_cfg: Dict = pop_config(
            profiler_cfg,
            'schedule',
            must_exist=True,
        )
        profiler_schedule = cyclic_schedule(**profiler_schedule_cfg)
        # Only support json trace handler
        profiler_trace_handlers: List[TraceHandler] = []
        profiler_trace_cfg: Optional[Dict] = pop_config(
            profiler_cfg,
            'json_trace_handler',
            must_exist=False,
            default_value=None,
        )
        if profiler_trace_cfg:
            profiler_trace_handlers.append(
                JSONTraceHandler(**profiler_trace_cfg),
            )
        profiler = Profiler(
            **profiler_cfg,
            trace_handlers=profiler_trace_handlers,
            schedule=profiler_schedule,
        )

    callback_configs = train_cfg.callbacks or {}

    # Callbacks
    if callback_configs is not None:
        for name, callback_cfg in callback_configs.items():
            if name == 'inactivity':
                callback_cfg['mosaicml_logger'] = mosaicml_logger

    callbacks: List[Callback] = [
        build_callback(
            name=str(name),
            kwargs=callback_cfg,
            train_config=logged_cfg,
        ) for name, callback_cfg in callback_configs.items()
    ]

    use_async_eval = any(isinstance(c, AsyncEval) for c in callbacks)

    algorithm_configs = train_cfg.algorithms or {}

    # Algorithms
    algorithms = [
        build_algorithm(str(name), algorithm_cfg)
        for name, algorithm_cfg in algorithm_configs.items()
    ]

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
                'AsyncEval callback does not support eval_first=True. Ignoring.',
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
        log_train_analytics(
            mosaicml_logger,
            model_config,
            train_loader_config,
            eval_loader_config,
            train_cfg.callbacks,
            tokenizer_name,
            train_cfg.load_path,
            icl_tasks_config,
            eval_gauntlet_config,
        )
    # Build Model
    log.info('Initializing model...')
    name = model_config.pop('name')
    assert isinstance(name, str)
    assert isinstance(model_config, dict)
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
    optimizer_cfg = train_cfg.optimizer
    optimizer = build_optimizer(model, optimizer_name, optimizer_cfg)

    # Now add the eval metrics
    try:
        if eval_loader_config is not None and not use_async_eval:
            eval_metrics = model.get_metrics(is_train=False)
            non_icl_metrics = [
                metric_name for metric_name, metric in eval_metrics.items()
                if not isinstance(metric, InContextLearningMetric)
            ]
            evaluators = add_metrics_to_eval_loaders(
                evaluators,
                non_icl_metrics,
            )
    except Exception as e:
        if mosaicml_logger is not None:
            mosaicml_logger.log_exception(e)
        raise e

    compile_config = train_cfg.compile_config
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
        dist_timeout=train_cfg.dist_timeout,
        profiler=profiler,
        compile_config=compile_config,
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
