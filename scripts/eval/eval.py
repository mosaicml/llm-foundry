# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import os
import re
import sys
import time
from typing import List

import pandas as pd
import torch
from composer.loggers import InMemoryLogger, LoggerDestination
from composer.trainer import Trainer
from composer.utils import dist, get_device, reproducibility
from omegaconf import OmegaConf as om

from llmfoundry.callbacks import ModelGauntlet
from llmfoundry.models.model_registry import COMPOSER_MODEL_REGISTRY
from llmfoundry.utils.builders import (build_icl_evaluators, build_logger,
                                       build_tokenizer)
from llmfoundry.utils.config_utils import process_init_device


def load_model(model_cfg, tokenizer, fsdp_config, num_retries):
    init_context = process_init_device(model_cfg, fsdp_config)

    retries = 0
    with init_context:
        while retries < num_retries:
            try:
                composer_model = COMPOSER_MODEL_REGISTRY[model_cfg.name](
                    model_cfg, tokenizer)
                return composer_model
            except Exception as e:
                retries += 1
                if retries >= num_retries:
                    raise e
                else:
                    print(
                        f'Got exception {str(e)} while loading model {model_cfg.name}. {num_retries-retries} retries remaining'
                    )


def evaluate_model(model_cfg, run_name, model_gauntlet_df):
    print(f'Evaluating model: {model_cfg.model_name}', flush=True)
    # Build tokenizer and model
    tokenizer = build_tokenizer(model_cfg.tokenizer)

    evaluators, logger_keys = build_icl_evaluators(cfg.icl_tasks, tokenizer,
                                                   cfg.max_seq_len,
                                                   cfg.device_eval_batch_size)
    if hasattr(cfg, 'model_gauntlet'):
        if isinstance(cfg.model_gauntlet, str):
            with open(cfg.model_gauntlet, 'r') as icl_f:
                model_gauntlet_cfg = om.load(icl_f)
            model_gauntlet = model_gauntlet_cfg.model_gauntlet
        else:
            model_gauntlet = cfg.model_gauntlet
        model_gauntlet.logger_keys = logger_keys
        model_gauntlet.benchmark_sizes = {
            e.label: e.dataloader.num_samples for e in evaluators
        }
        model_gauntlet_callback = ModelGauntlet(**model_gauntlet)
    else:
        model_gauntlet = None
        model_gauntlet_callback = None

    fsdp_config = cfg.get('fsdp_config', None)
    fsdp_config = om.to_container(
        fsdp_config, resolve=True) if fsdp_config is not None else None

    composer_model = load_model(model_cfg.model, tokenizer, fsdp_config,
                                cfg.get('num_retries', 3))

    if model_gauntlet_df is None and model_gauntlet is not None:
        model_gauntlet_df = pd.DataFrame(
            columns=['model_name', 'average'] +
            [t.name for t in model_gauntlet.categories])

    in_memory_logger = InMemoryLogger()  # track metrics in the in_memory_logger
    loggers: List[LoggerDestination] = [
        build_logger(name, logger_cfg)
        for name, logger_cfg in (cfg.get('loggers') or {}).items()
    ]
    loggers.append(in_memory_logger)

    load_path = model_cfg.get('load_path', None)

    trainer = Trainer(
        run_name=run_name,
        model=composer_model,
        loggers=loggers,
        precision=cfg.precision,
        fsdp_config=fsdp_config,  # type: ignore
        load_path=load_path,
        load_weights_only=True,
        progress_bar=False,
        log_to_console=True,
        dist_timeout=cfg.dist_timeout,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    a = time.time()
    trainer.eval(eval_dataloader=evaluators)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    b = time.time()
    print(f'Ran {model_cfg.model_name} eval in: {b-a} seconds')
    return (in_memory_logger, logger_keys, model_gauntlet_callback,
            model_gauntlet, model_gauntlet_df)


def main(cfg):
    cfg.dist_timeout = cfg.get('dist_timeout', 600.0)
    if cfg.get('run_name') is None:
        cfg.run_name = os.environ.get('RUN_NAME', 'llm')

    reproducibility.seed_all(cfg.seed)
    dist.initialize_dist(get_device(None), timeout=cfg.dist_timeout)

    model_gauntlet_df = None
    models_df = None
    for model_cfg in cfg.models:
        (in_memory_logger, logger_keys, model_gauntlet_callback, model_gauntlet,
         model_gauntlet_df) = evaluate_model(model_cfg, cfg.run_name,
                                             model_gauntlet_df)

        if model_gauntlet_callback is not None:
            composite_scores = model_gauntlet_callback.eval_end(
                None, in_memory_logger)

        benchmark_to_taxonomy = {}
        if model_gauntlet is not None:
            for t in model_gauntlet.categories:
                for b in t.benchmarks:
                    benchmark_to_taxonomy[b.name] = t.name

        model_results = calculate_markdown_results(logger_keys,
                                                   in_memory_logger.data,
                                                   benchmark_to_taxonomy,
                                                   model_cfg.model_name)

        if models_df is None:
            models_df = model_results
        else:
            models_df = pd.concat([models_df, model_results], ignore_index=True)

        if model_gauntlet_df is not None and model_gauntlet is not None and model_gauntlet_df is not None:
            row = {'model_name': model_cfg['model_name']}
            row.update({
                t.name: composite_scores[f'metrics/model_gauntlet/{t.name}']
                for t in model_gauntlet.categories
            })
            row.update({
                'average': composite_scores[f'metrics/model_gauntlet/average']
            })
            model_gauntlet_df = pd.concat(
                [model_gauntlet_df, pd.DataFrame([row])], ignore_index=True)

            print(f'Printing gauntlet results for all models')
            print(
                model_gauntlet_df.sort_values(
                    'average', ascending=False).to_markdown(index=False))
        print(f'Printing complete results for all models')
        print(models_df.to_markdown(index=False))


def calculate_markdown_results(logger_keys, logger_data, benchmark_to_taxonomy,
                               model_name):
    results = {}
    pat = re.compile('metrics/(.*?)/(\d+)-shot(/.*?)?/InContextLearning(.*)')
    for key in logger_keys:
        match = pat.match(key)
        val = logger_data[key][0][1].item()
        if match:
            eval_name = match.group(1)
            num_shot = match.group(2)
            subcat = match.group(3)
            if subcat is not None:
                subcat = subcat[1:]
            else:
                subcat = 'no_subcat'
            metric = match.group(4)
            if num_shot not in results:
                results[num_shot] = {}
            if eval_name not in results[num_shot]:
                results[num_shot][eval_name] = {}
            if metric not in results[num_shot][eval_name]:
                results[num_shot][eval_name][metric] = []

            results[num_shot][eval_name][metric].append({
                'val': val,
                'subcat': subcat
            })
    df = pd.DataFrame(columns=[
        'Category', 'Benchmark', 'Subtask', 'Accuracy', 'Number few shot',
        'Model'
    ])
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
                        'Model': model_name
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
                            model_name
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
                                model_name
                        }
                        df = pd.concat([df, pd.DataFrame([row])],
                                       ignore_index=True)
    return df


if __name__ == '__main__':
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg = om.merge(yaml_cfg, cli_cfg)
    main(cfg)
