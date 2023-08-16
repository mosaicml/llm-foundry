# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import time
from typing import Dict, List, Optional

import pandas as pd
import torch
from composer.models.base import ComposerModel
from composer.trainer import Trainer
from composer.utils import dist, get_device, reproducibility
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from transformers import (AutoModelForCausalLM, PreTrainedTokenizerBase,
                          T5ForConditionalGeneration)

from llmfoundry.callbacks import ModelGauntlet
from llmfoundry.models import MPTForCausalLM
from llmfoundry.models.model_registry import COMPOSER_MODEL_REGISTRY
from llmfoundry.utils.builders import build_icl_evaluators, build_tokenizer
from llmfoundry.utils.config_utils import process_init_device


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
            underlying_model = model_registry[model_cfg.name].from_pretrained(
                model_cfg.pretrained_model_name_or_path,
                trust_remote_code=trust_remote_code,
                use_auth_token=use_auth_token,
            )

            peft_model = PeftModel.from_pretrained(
                underlying_model, model_cfg.pretrained_lora_id_or_path)

            composer_model_wrapper = COMPOSER_MODEL_REGISTRY[model_cfg.name](
                peft_model, tokenizer)
            return composer_model_wrapper
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


def evaluate_model(model_cfg: DictConfig, cfg: DictConfig, run_name: str,
                   model_gauntlet_df: Optional[pd.DataFrame]):
    print(f'Evaluating model: {model_cfg.model_name}', flush=True)
    # Build tokenizer and model
    tokenizer = build_tokenizer(model_cfg.tokenizer)

    evaluators, metric_names = build_icl_evaluators(
        cfg.icl_tasks,
        tokenizer,
        cfg.max_seq_len,
        cfg.device_eval_batch_size,
        icl_subset_num_batches=cfg.get('icl_subset_num_batches', None))

    callbacks = []
    if hasattr(cfg, 'model_gauntlet'):
        if isinstance(cfg.model_gauntlet, str):
            with open(cfg.model_gauntlet, 'r') as icl_f:
                model_gauntlet_cfg = om.load(icl_f)
            model_gauntlet = model_gauntlet_cfg.model_gauntlet
        else:
            model_gauntlet = cfg.model_gauntlet
        model_gauntlet.metric_names = metric_names
        model_gauntlet.benchmark_sizes = {
            e.label: e.dataloader.num_samples for e in evaluators
        }
        model_gauntlet_callback = ModelGauntlet(**model_gauntlet)
        callbacks.append(model_gauntlet_callback)
    else:
        model_gauntlet = None
        model_gauntlet_callback = None

    fsdp_config = cfg.get('fsdp_config', None)
    fsdp_config = om.to_container(
        fsdp_config, resolve=True) if fsdp_config is not None else None
    assert isinstance(fsdp_config, Dict) or fsdp_config is None

    if hasattr(model_cfg.model, 'pretrained_lora_id_or_path'):
        composer_model = load_peft_model(model_cfg.model, tokenizer,
                                         cfg.get('num_retries', 3))
    else:
        composer_model = load_model(model_cfg.model, tokenizer, fsdp_config,
                                    cfg.get('num_retries', 3))

    if model_gauntlet_df is None and model_gauntlet is not None:
        model_gauntlet_df = pd.DataFrame(
            columns=['model_name', 'average'] +
            [t.name for t in model_gauntlet.categories])

    load_path = model_cfg.get('load_path', None)

    assert composer_model is not None

    trainer = Trainer(
        run_name=run_name,
        model=composer_model,
        callbacks=callbacks,
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
    return (trainer, metric_names, model_gauntlet_callback, model_gauntlet,
            model_gauntlet_df)


def main(cfg: DictConfig):
    cfg.dist_timeout = cfg.get('dist_timeout', 600.0)
    if cfg.get('run_name') is None:
        cfg.run_name = os.environ.get('RUN_NAME', 'llm')

    reproducibility.seed_all(cfg.seed)
    dist.initialize_dist(get_device(None), timeout=cfg.dist_timeout)

    model_gauntlet_df = None
    models_df = None
    composite_scores = None
    for model_cfg in cfg.models:
        (trainer, metric_names, model_gauntlet_callback, model_gauntlet,
         model_gauntlet_df) = evaluate_model(model_cfg, cfg, cfg.run_name,
                                             model_gauntlet_df)

        if model_gauntlet_callback is not None:
            composite_scores = model_gauntlet_callback.eval_after_all(
                trainer.state, trainer.logger)

        benchmark_to_taxonomy = {}
        if model_gauntlet is not None:
            for t in model_gauntlet.categories:
                for b in t.benchmarks:
                    benchmark_to_taxonomy[b.name] = t.name

        model_results = calculate_markdown_results(metric_names, trainer,
                                                   benchmark_to_taxonomy,
                                                   model_cfg.model_name)

        if models_df is None:
            models_df = model_results
        else:
            models_df = pd.concat([models_df, model_results], ignore_index=True)

        if model_gauntlet_df is not None and model_gauntlet is not None and model_gauntlet_df is not None:
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
        assert models_df is not None
        print(models_df.to_markdown(index=False))


def calculate_markdown_results(metric_keys: List[str], trainer: Trainer,
                               benchmark_to_taxonomy: Dict[str, str],
                               model_name: str):
    results = {}

    for key in metric_keys:
        # dl_name consists is either 2-tuple (benchmark_name, num_fewshot)
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
            'subcat': dl_name[-1] if len(dl_name) == 3 else 'no_subcat'
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
