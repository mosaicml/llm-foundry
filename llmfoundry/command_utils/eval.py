# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import time
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd
import torch
from composer.core import Callback
from composer.loggers.logger_destination import LoggerDestination
from composer.trainer import Trainer
from composer.utils import dist, get_device, reproducibility
from omegaconf import DictConfig
from omegaconf import OmegaConf as om

from llmfoundry.utils import (
    find_mosaicml_logger,
    log_eval_analytics,
    maybe_create_mosaicml_logger,
)
from llmfoundry.utils.builders import (
    add_metrics_to_eval_loaders,
    build_callback,
    build_composer_model,
    build_evaluators,
    build_logger,
    build_tokenizer,
)
from llmfoundry.utils.config_utils import (
    EVAL_CONFIG_KEYS,
    EvalConfig,
    log_config,
    make_dataclass_and_log_config,
    process_init_device,
)
from llmfoundry.utils.registry_utils import import_file

log = logging.getLogger(__name__)


def evaluate_model(
    tokenizer: Dict[str, Any],
    model_name: str,
    model: Dict[str, Any],
    dist_timeout: Union[float, int],
    run_name: str,
    seed: int,
    icl_tasks: Union[str, list[Dict[str, Any]]],
    max_seq_len: int,
    device_eval_batch_size: Union[int, float],
    eval_gauntlet_config: Optional[Union[str, Dict[str, Any]]],
    eval_loader_config: Optional[Union[Dict[str, Any], list[Dict[str, Any]]]],
    fsdp_config: Optional[Dict[str, Any]],
    loggers: list[LoggerDestination],
    python_log_level: Optional[str],
    precision: str,
    eval_gauntlet_df: Optional[pd.DataFrame],
    eval_subset_num_batches: int,
    icl_subset_num_batches: Optional[int],
    callback_configs: Optional[Dict[str, Any]],
    metadata: Optional[Dict[str, str]],
    logged_config: Dict[str, Any],
    should_log_config: bool = True,
    load_path: Optional[str] = None,
):
    log.info(f'Evaluating model: {model_name}')
    # Build tokenizer and model
    tokenizer_cfg = tokenizer
    tokenizer_name = tokenizer_cfg['name']
    tokenizer_kwargs = tokenizer_cfg.get('kwargs', {})
    tokenizer = build_tokenizer(tokenizer_name, tokenizer_kwargs)

    evaluators, logger_keys, eval_gauntlet_callback = build_evaluators(
        eval_loader_config,
        icl_tasks,
        eval_gauntlet_config,
        tokenizer=tokenizer,
        device_eval_batch_size=device_eval_batch_size,
        icl_seq_len=max_seq_len,
        icl_subset_num_batches=icl_subset_num_batches,
    )

    # Callbacks
    callbacks: list[Callback] = [
        build_callback(name=str(name), kwargs=callback_cfg)
        for name, callback_cfg in callback_configs.items()
    ] if callback_configs else []

    if eval_gauntlet_callback is not None:
        callbacks.append(eval_gauntlet_callback)

    if metadata is not None:
        # Find the MosaicMLLogger
        mosaicml_logger = find_mosaicml_logger(loggers)

        if mosaicml_logger is not None:
            mosaicml_logger.log_metrics(metadata)
            mosaicml_logger._flush_metadata(force_flush=True)

    if fsdp_config and model.get('load_in_8bit', False):
        raise ValueError(
            'The FSDP config block is not supported when loading ' +
            'Hugging Face models in 8bit.',
        )

    init_context = process_init_device(model, fsdp_config)

    name = model.pop('name')
    composer_model = build_composer_model(
        name=name,
        tokenizer=tokenizer,
        init_context=init_context,
        cfg=model,
    )

    # Now add the eval metrics
    if eval_loader_config is not None:
        train_metrics = composer_model.get_metrics(is_train=True)
        evaluators = add_metrics_to_eval_loaders(
            evaluators,
            list(train_metrics.keys()),
        )

    if eval_gauntlet_df is None and eval_gauntlet_callback is not None:
        eval_gauntlet_df = pd.DataFrame(
            columns=['model_name'] + list(eval_gauntlet_callback.averages) +
            [t['name'] for t in eval_gauntlet_callback.categories],
        )

    if name == 'mpt_causal_lm' and load_path is None:
        raise ValueError(
            'MPT causal LMs require a load_path to the checkpoint for model evaluation.'
            +
            ' Please check your yaml and the model_cfg to ensure that load_path is set.',
        )

    assert composer_model is not None

    log.info(f'Building trainer for {model_name}...')
    trainer = Trainer(
        run_name=run_name,
        seed=seed,
        model=composer_model,
        callbacks=callbacks,
        loggers=loggers,
        precision=precision,
        fsdp_config=fsdp_config,
        load_path=load_path,
        load_weights_only=True,
        progress_bar=False,
        log_to_console=True,
        dist_timeout=dist_timeout,
        python_log_level=python_log_level,
    )

    if should_log_config:
        log.info('Evaluation config:')
        log_config(logged_config)

    log.info(f'Starting eval for {model_name}...')
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    a = time.time()
    trainer.eval(
        eval_dataloader=evaluators,
        subset_num_batches=eval_subset_num_batches,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    b = time.time()

    log.info(f'Ran {model_name} eval in: {b-a} seconds')
    return (trainer, logger_keys, eval_gauntlet_callback, eval_gauntlet_df)


def allow_toplevel_keys(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Transform the config to allow top-level keys for model configuration.

    This function allows users to use the 'train.py' syntax in 'eval.py'.
    It converts a config with top-level 'model', 'tokenizer', and (optionally) 'load_path' keys
    into the nested 'models' list format required by 'eval.py'.

    Input config format (train.py style):
    ```yaml
    model:
      <model_kwargs>
    load_path: /path/to/checkpoint
    tokenizer:
      <tokenizer_kwargs>
    ```

    Output config format (eval.py style):
    ```yaml
    models:
      - model:
          <model_kwargs>
        tokenizer:
          <tokenizer_kwargs>
        load_path: /path/to/checkpoint
    ```
    """
    if 'model' in cfg:
        if 'models' in cfg:
            raise ValueError(
                'Please specify either model or models in the config, not both',
            )
        default_name = cfg.get('model').get('name') # type: ignore
        model_cfg = {
            'model': cfg.pop('model'),
            'tokenizer': cfg.pop('tokenizer', None),
            'model_name': cfg.pop('model_name', default_name),
        }
        if 'tokenizer' not in model_cfg or model_cfg['tokenizer'] is None:
            raise ValueError(
                'When specifying model, "tokenizer" must be provided in the config',
            )
        if 'load_path' in cfg:
            model_cfg['load_path'] = cfg.pop('load_path')
        cfg['models'] = [model_cfg]

    return cfg


def evaluate(cfg: DictConfig) -> Tuple[list[Trainer], pd.DataFrame]:
    # Run user provided code if specified
    for code_path in cfg.get('code_paths', []):
        import_file(code_path)

    logged_cfg, eval_config = make_dataclass_and_log_config(
        cfg,
        EvalConfig,
        EVAL_CONFIG_KEYS,
        transforms=[allow_toplevel_keys],
        icl_tasks_required=True,
    )

    model_configs = eval_config.models
    eval_gauntlet_config = eval_config.eval_gauntlet or eval_config.eval_gauntlet_str

    fsdp_config = eval_config.fsdp_config

    # Mandatory Evaluation Parameters
    icl_tasks = eval_config.icl_tasks or eval_config.icl_tasks_str
    if icl_tasks is None:
        raise ValueError('icl_tasks must be specified in the config')

    # Optional Evaluation Parameters with default values
    eval_loader_config = eval_config.eval_loader or eval_config.eval_loaders
    default_run_name: str = os.environ.get('RUN_NAME', 'llm')
    run_name = eval_config.run_name if eval_config.run_name else default_run_name

    reproducibility.seed_all(eval_config.seed)
    dist.initialize_dist(get_device(None), timeout=eval_config.dist_timeout)

    if eval_config.python_log_level is not None:
        logging.basicConfig(
            # Example of format string
            # 2022-06-29 11:22:26,152: rank0[822018][MainThread]: INFO: Message here
            format=
            f'%(asctime)s: rank{dist.get_global_rank()}[%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s',
        )
        logging.getLogger('llmfoundry').setLevel(
            eval_config.python_log_level.upper(),
        )

    # Default argument values for evaluate_model
    eval_gauntlet_df = None
    models_df = None
    composite_scores = None
    trainers = []

    # Build loggers
    loggers: list[LoggerDestination] = [
        build_logger(name, logger_cfg)
        for name, logger_cfg in (eval_config.loggers or {}).items()
    ]

    mosaicml_logger = find_mosaicml_logger(loggers)
    if mosaicml_logger is None:
        mosaicml_logger = maybe_create_mosaicml_logger()
        # mosaicml_logger will be None if run isn't on MosaicML platform
        if mosaicml_logger is not None:
            loggers.append(mosaicml_logger)

    # mosaicml_logger will be None if the run isn't from the MosaicML platform
    if mosaicml_logger is not None:
        log_eval_analytics(
            mosaicml_logger,
            model_configs,
            icl_tasks,
            eval_gauntlet_config,
        )

    for model_cfg in model_configs:

        attn_config = model_cfg['model'].get('attn_config', None)
        if attn_config is not None:
            seq_parallel_world_size = attn_config.get(
                'seq_parallel_world_size',
                None,
            )
            if seq_parallel_world_size is not None and seq_parallel_world_size != 1:
                raise ValueError(
                    'Offline eval does not support sequence parallelism.',
                )

        (trainer, logger_keys, eval_gauntlet_callback,
         eval_gauntlet_df) = evaluate_model(
             dist_timeout=eval_config.dist_timeout,
             run_name=run_name,
             seed=eval_config.seed,
             icl_tasks=icl_tasks,
             max_seq_len=eval_config.max_seq_len,
             device_eval_batch_size=eval_config.device_eval_batch_size,
             eval_gauntlet_config=eval_gauntlet_config,
             eval_loader_config=eval_loader_config,
             fsdp_config=fsdp_config,
             loggers=loggers,
             python_log_level=eval_config.python_log_level,
             precision=eval_config.precision,
             eval_gauntlet_df=eval_gauntlet_df,
             callback_configs=eval_config.callbacks,
             eval_subset_num_batches=eval_config.eval_subset_num_batches,
             icl_subset_num_batches=eval_config.icl_subset_num_batches,
             metadata=eval_config.metadata,
             logged_config=logged_cfg,
             should_log_config=eval_config.log_config,
             **model_cfg,
         )
        trainers.append(trainer)

        if eval_gauntlet_callback is not None:
            composite_scores = eval_gauntlet_callback.eval_after_all(
                trainer.state,
                trainer.logger,
            )

        benchmark_to_taxonomy = {}
        if eval_gauntlet_callback is not None:
            for t in eval_gauntlet_callback.categories:
                for b in t['benchmarks']:
                    benchmark_to_taxonomy[b['name']] = t['name']

        assert 'model_name' in model_cfg, 'model_name must be specified in model config'
        model_results = calculate_markdown_results(
            logger_keys,
            trainer,
            benchmark_to_taxonomy,
            model_cfg['model_name'],
        )

        if models_df is None:
            models_df = model_results
        else:
            models_df = pd.concat([models_df, model_results], ignore_index=True)

        if eval_gauntlet_df is not None and eval_gauntlet_callback is not None:
            assert composite_scores is not None
            row = {'model_name': model_cfg['model_name']}
            row.update({
                k.split('/')[-1]: v for k, v in composite_scores.items()
            })
            eval_gauntlet_df = pd.concat([
                eval_gauntlet_df,
                pd.DataFrame([row]),
            ],
                                         ignore_index=True)

            print(f'Printing gauntlet results for all models')

            print(
                eval_gauntlet_df.sort_values(
                    list(eval_gauntlet_callback.averages.keys())[0],
                    ascending=False,
                ).to_markdown(index=False),
            )
        print(f'Printing complete results for all models')
        assert models_df is not None
        print(models_df.to_markdown(index=False))

        trainer.close()

    return trainers, eval_gauntlet_df


def calculate_markdown_results(
    logger_keys: list[str],
    trainer: Trainer,
    benchmark_to_taxonomy: Dict[str, str],
    model_name: str,
):
    results = {}

    for key in logger_keys:
        # dl_name is either 2-tuple (benchmark_name, num_fewshot)
        # or 3-tuple (benchmark_name, num_fewshot, subcategory)
        dl_name, metric_name = key.split('/')[1:-1], key.split('/')[-1]
        if 'Accuracy' not in metric_name:
            continue

        metric = trainer.state.eval_metrics.get('/'.join(dl_name),
                                                {}).get(metric_name, None)

        if metric is None:
            continue
        if dl_name[1] not in results:
            results[dl_name[1]] = {}

        if dl_name[0] not in results[dl_name[1]]:
            results[dl_name[1]][dl_name[0]] = {}

        if metric_name not in results[dl_name[1]][dl_name[0]]:
            results[dl_name[1]][dl_name[0]][metric_name] = []

        results[dl_name[1]][dl_name[0]][metric_name].append({
            'val': metric.compute(),
            'subcat': dl_name[-1] if len(dl_name) == 3 else 'no_subcat',
        })

    df = pd.DataFrame(
        columns=[
            'Category',
            'Benchmark',
            'Subtask',
            'Accuracy',
            'Number few shot',
            'Model',
        ],
    )

    for num_shot in results:
        for benchmark in results[num_shot]:
            for metric in results[num_shot][benchmark]:
                subscores = results[num_shot][benchmark][metric]
                if len(subscores) == 1:
                    row = {
                        'Category': benchmark_to_taxonomy.get(benchmark, ''),
                        'Benchmark': benchmark,
                        'Subtask': None,
                        'Accuracy': subscores[0]['val'],
                        'Number few shot': num_shot,
                        'Model': model_name,
                    }
                    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                else:
                    row = {
                        'Category':
                            benchmark_to_taxonomy.get(benchmark, ''),
                        'Benchmark':
                            benchmark,
                        'Subtask':
                            'Average',
                        'Accuracy':
                            sum(s['val'] for s in subscores) / len(subscores),
                        'Number few shot':
                            num_shot,
                        'Model':
                            model_name,
                    }
                    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                    for sub in subscores:
                        row = {
                            'Category':
                                benchmark_to_taxonomy.get(benchmark, ''),
                            'Benchmark':
                                None,
                            'Subtask':
                                sub['subcat'],
                            'Accuracy':
                                sub['val'],
                            'Number few shot':
                                num_shot,
                            'Model':
                                model_name,
                        }
                        df = pd.concat([df, pd.DataFrame([row])],
                                       ignore_index=True)
    return df


def eval_from_yaml(
    yaml_path: str,
    args_list: Optional[list[str]],
) -> Tuple[list[Trainer], pd.DataFrame]:
    """Run the evaluation with optional overrides from CLI."""
    # Load yaml and CLI arguments.
    om.clear_resolver('oc.env')
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    if args_list:
        cli_cfg = om.from_cli(args_list)
        yaml_cfg = om.merge(yaml_cfg, cli_cfg)
    assert isinstance(yaml_cfg, DictConfig)
    return evaluate(yaml_cfg)
