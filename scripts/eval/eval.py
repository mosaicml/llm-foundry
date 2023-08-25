# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import os
import re
import sys
import time
import warnings
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import torch
from composer.loggers import InMemoryLogger, LoggerDestination
from composer.models.base import ComposerModel
from composer.trainer import Trainer
from composer.utils import dist, get_device, reproducibility
from omegaconf import DictConfig, ListConfig
from omegaconf import OmegaConf as om
from transformers import (AutoModelForCausalLM, PreTrainedTokenizerBase,
                          T5ForConditionalGeneration)

from llmfoundry.callbacks import ModelGauntlet
from llmfoundry.models.model_registry import COMPOSER_MODEL_REGISTRY
from llmfoundry.models.mpt import MPTForCausalLM
from llmfoundry.utils.builders import (build_icl_evaluators, build_logger,
                                       build_tokenizer)
from llmfoundry.utils.config_utils import pop_config, process_init_device


def load_peft_model(model_cfg: DictConfig, tokenizer: PreTrainedTokenizerBase,
                    num_retries: int) -> Optional[ComposerModel]:
    try:
        from peft import PeftModel
    except ImportError as e:
        raise ImportError(
            f'Error importing from peft. Run `pip install -e .[gpu,peft]`. \n {e}'
        )

    model_registry = {
        'mpt_causal_lm': MPTForCausalLM,
        'hf_causal_lm': AutoModelForCausalLM,
        'hf_prefix_lm': AutoModelForCausalLM,
        'hf_t5': T5ForConditionalGeneration,
    }

    retries = 0
    while retries < num_retries:
        try:
            trust_remote_code = model_cfg.get('trust_remote_code', True)
            use_auth_token = model_cfg.get('use_auth_token', False)
            model = model_registry[model_cfg.name].from_pretrained(
                model_cfg.pretrained_model_name_or_path,
                trust_remote_code=trust_remote_code,
                use_auth_token=use_auth_token,
            )

            peft_model = PeftModel.from_pretrained(
                model, model_cfg.pretrained_lora_id_or_path)

            composer_model = COMPOSER_MODEL_REGISTRY[model_cfg.name](peft_model,
                                                                     tokenizer)
            return composer_model
        except Exception as e:
            retries += 1
            if retries >= num_retries:
                raise e
            else:
                print(
                    f'Got exception {str(e)} while loading model {model_cfg.name}. {num_retries-retries} retries remaining'
                )


def load_model(model_cfg: DictConfig, tokenizer: PreTrainedTokenizerBase,
               fsdp_config: Optional[Dict],
               num_retries: int) -> Optional[ComposerModel]:
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


def evaluate_model(model_cfg: DictConfig, dist_timeout: Union[float, int],
                   run_name: str, icl_tasks: Union[str, ListConfig],
                   max_seq_len: int, device_eval_batch_size: int,
                   model_gauntlet_config: Optional[Union[str, DictConfig]],
                   fsdp_config: Optional[Dict], num_retries: int,
                   loggers_cfg: Dict[str, Any], python_log_level: str,
                   precision: str, model_gauntlet_df: Optional[pd.DataFrame]):
    print(f'Evaluating model: {model_cfg.model_name}', flush=True)
    # Build tokenizer and model
    tokenizer = build_tokenizer(model_cfg.tokenizer)
    evaluators, logger_keys = build_icl_evaluators(icl_tasks, tokenizer,
                                                   max_seq_len,
                                                   device_eval_batch_size)
    model_gauntlet: Optional[DictConfig] = None
    model_gauntlet_callback: Optional[ModelGauntlet] = None
    if model_gauntlet_config is not None:
        if isinstance(model_gauntlet_config, str):
            with open(model_gauntlet_config, 'r', encoding='utf-8') as icl_f:
                loaded_model_gauntlet_config = om.load(icl_f)
            model_gauntlet = loaded_model_gauntlet_config.model_gauntlet
        else:
            model_gauntlet = model_gauntlet_config
        model_gauntlet.logger_keys = logger_keys  # type: ignore
        model_gauntlet.benchmark_sizes = {  # type: ignore
            e.label: e.dataloader.num_samples for e in evaluators
        }
        model_gauntlet_callback = ModelGauntlet(**
                                                model_gauntlet)  # type: ignore

    if fsdp_config and model_cfg.model.load_in_8bit:
        raise ValueError(
            'The FSDP config block is not supported when loading ' +
            'Hugging Face models in 8bit.')

    if hasattr(model_cfg.model, 'pretrained_lora_id_or_path'):
        composer_model = load_peft_model(model_cfg.model, tokenizer,
                                         num_retries)
    else:
        composer_model = load_model(model_cfg.model, tokenizer, fsdp_config,
                                    num_retries)

    if model_gauntlet_df is None and model_gauntlet is not None:
        model_gauntlet_df = pd.DataFrame(
            columns=['model_name', 'average'] +
            [t.name for t in model_gauntlet.categories])

    in_memory_logger = InMemoryLogger()  # track metrics in the in_memory_logger
    loggers: List[LoggerDestination] = [
        build_logger(name, logger_cfg)
        for name, logger_cfg in loggers_cfg.items()
    ]
    loggers.append(in_memory_logger)

    load_path = model_cfg.get('load_path', None)

    assert composer_model is not None
    trainer = Trainer(
        run_name=run_name,
        model=composer_model,
        loggers=loggers,
        precision=precision,
        fsdp_config=fsdp_config,  # type: ignore
        load_path=load_path,
        load_weights_only=True,
        progress_bar=False,
        log_to_console=True,
        dist_timeout=dist_timeout,
        python_log_level=python_log_level,
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


def main(cfg: DictConfig):
    om.resolve(cfg)
    model_configs: ListConfig = pop_config(cfg, 'models', must_exist=True)
    model_gauntlet_config: Optional[Union[str, DictConfig]] = pop_config(
        cfg, 'model_gauntlet', must_exist=False, default_value=None)
    fsdp_dict_cfg: Optional[DictConfig] = pop_config(cfg,
                                                     'fsdp_config',
                                                     must_exist=False,
                                                     default_value=None)
    fsdp_config: Optional[Dict] = om.to_container(
        fsdp_dict_cfg,
        resolve=True) if fsdp_dict_cfg is not None else None  # type: ignore
    assert isinstance(fsdp_config, Dict) or fsdp_config is None

    # Mandatory Evaluation Parameters
    icl_tasks: Union[str, ListConfig] = pop_config(cfg,
                                                   'icl_tasks',
                                                   must_exist=True)
    max_seq_len: int = pop_config(cfg, 'max_seq_len', must_exist=True)
    device_eval_batch_size: int = pop_config(cfg,
                                             'device_eval_batch_size',
                                             must_exist=True)
    precision: str = pop_config(cfg, 'precision', must_exist=True)
    python_log_level: str = pop_config(cfg,
                                       'python_log_level',
                                       must_exist=False,
                                       default_value='debug')

    # Optional Evaluation Parameters with default values
    seed: int = pop_config(cfg, 'seed', must_exist=False, default_value=17)
    dist_timeout: Union[float, int] = pop_config(cfg,
                                                 'dist_timeout',
                                                 must_exist=False,
                                                 default_value=600.0)
    default_run_name: str = os.environ.get('RUN_NAME', 'llm')
    run_name: str = pop_config(cfg,
                               'run_name',
                               must_exist=False,
                               default_value=default_run_name)
    num_retries: int = pop_config(cfg,
                                  'num_retries',
                                  must_exist=False,
                                  default_value=3)
    loggers_cfg: Dict[str, Any] = pop_config(cfg,
                                             'loggers',
                                             must_exist=False,
                                             default_value={})

    # Pop out interpolation variables.
    pop_config(cfg, 'model_name_or_path', must_exist=False, default_value=None)

    # Warn for unused parameters
    for key in cfg:
        warnings.warn(
            f'Unused parameter {key} found in cfg. Please check your yaml to ensure this parameter is necessary.'
        )

    reproducibility.seed_all(seed)
    dist.initialize_dist(get_device(None), timeout=dist_timeout)

    model_gauntlet_df = None
    models_df = None
    composite_scores = None
    for model_cfg in model_configs:
        (in_memory_logger, logger_keys, model_gauntlet_callback, model_gauntlet,
         model_gauntlet_df) = evaluate_model(
             model_cfg=model_cfg,
             dist_timeout=dist_timeout,
             run_name=run_name,
             icl_tasks=icl_tasks,
             max_seq_len=max_seq_len,
             device_eval_batch_size=device_eval_batch_size,
             model_gauntlet_config=model_gauntlet_config,
             fsdp_config=fsdp_config,
             num_retries=num_retries,
             loggers_cfg=loggers_cfg,
             python_log_level=python_log_level,
             precision=precision,
             model_gauntlet_df=model_gauntlet_df,
         )

        if model_gauntlet_callback is not None:
            # TODO(bmosaicml) This needs to be refactored to fix the typing issue
            composite_scores = model_gauntlet_callback.eval_end(
                None, in_memory_logger)  # type: ignore

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

        if model_gauntlet_df is not None and model_gauntlet is not None:
            assert composite_scores is not None
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


def calculate_markdown_results(logger_keys: List[str], logger_data: Dict,
                               benchmark_to_taxonomy: Dict[str, str],
                               model_name: str):
    results = {}
    pat = re.compile(
        'metrics/(.*?)/(\d+)-shot(/.*?)?/InContextLearning(.*)')  # type: ignore
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
    assert isinstance(cfg, DictConfig)
    main(cfg)
