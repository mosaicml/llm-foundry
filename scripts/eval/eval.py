# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import sys
import time
from typing import List
import re
import torch
from composer.loggers import InMemoryLogger, LoggerDestination
from composer.trainer import Trainer
from composer.utils import dist, get_device, reproducibility
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from llmfoundry.callbacks import EvalTaxonomy
from llmfoundry.models.model_registry import COMPOSER_MODEL_REGISTRY
from llmfoundry.utils.builders import (build_icl_evaluators, build_logger,
                                       build_tokenizer)
import pandas as pd

def main(cfg):
    cfg.dist_timeout = cfg.get('dist_timeout', 600.0)

    reproducibility.seed_all(cfg.seed)
    dist.initialize_dist(get_device(None), timeout=cfg.dist_timeout)

    

    taxonomy_df = None
    for model_cfg in cfg.models:
        # Build tokenizer and model
        
        
        tokenizer = build_tokenizer(model_cfg.tokenizer)
        composer_model = COMPOSER_MODEL_REGISTRY[model_cfg.model.name](model_cfg.model,
                                                                tokenizer)

        evaluators, logger_keys = build_icl_evaluators(cfg.icl_tasks, tokenizer,
            cfg.max_seq_len,
            cfg.device_eval_batch_size)     

        if hasattr(cfg, "icl_taxonomy"):
            if isinstance(cfg.icl_taxonomy, str):
                with open(cfg.icl_taxonomy, 'r') as icl_f:
                    taxonomy_cfg = om.load(icl_f)
                taxonomy = taxonomy_cfg.icl_taxonomy
            else:
                taxonomy = cfg.icl_taxonomy
            taxonomy.logger_keys = logger_keys
            taxonomy_callback = EvalTaxonomy(**taxonomy)

        if taxonomy_df is None:
            taxonomy_df = pd.DataFrame(columns=["model_name"] + [t.name for t in taxonomy.tasks])

        in_memory_logger = InMemoryLogger()  # track metrics in the in_memory_logger
        loggers: List[LoggerDestination] = [
            build_logger(name, logger_cfg)
            for name, logger_cfg in (cfg.get('loggers') or {}).items()
        ]
        loggers.append(in_memory_logger)

        fsdp_config = cfg.get('fsdp_config', None)
        fsdp_config = om.to_container(
            fsdp_config, resolve=True) if fsdp_config is not None else None

        load_path = cfg.get('load_path', None)

        trainer = Trainer(
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
        taxonomy_scores = taxonomy_callback.eval_end(None, in_memory_logger)
        calculate_markdown_results(logger_keys, in_memory_logger.data, model_cfg.model_name)
        row = {
            "model_name": model_cfg['model_name']
        }

        row.update({t.name: taxonomy_scores[f"metrics/icl_taxonomy/{t.name}"] for t in taxonomy.tasks})
        taxonomy_df = pd.concat([taxonomy_df, pd.DataFrame([row])],
                                       ignore_index=True)
        
        print(f"Printing taxonomy results for all models")
        print(taxonomy_df.to_markdown(index=False))


def calculate_markdown_results(logger_keys, logger_data, model_name):
    results = {}
    pat = re.compile("metrics/(.*?)/(\d+)-shot(/.*?)?/InContextLearning(.*)")
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
                subcat = "no_subcat"
            metric = match.group(4)
            if num_shot not in results:
                results[num_shot] = {}
            if eval_name not in results[num_shot]:
                results[num_shot][eval_name] = {}
            if metric not in results[num_shot][eval_name]:
                results[num_shot][eval_name][metric] = []

            results[num_shot][eval_name][metric].append({
               "val": val, "subcat": subcat
            })
    df = pd.DataFrame(columns=[
            "Benchmark", "Subcategory", "Accuracy", "Number few shot", "Model"
        ])
    for num_shot in results:
        for benchmark in results[num_shot]:
            for metric in results[num_shot][benchmark]:
                subscores = results[num_shot][benchmark][metric]
                if len(subscores) == 1:
                    row = {
                        "Benchmark": benchmark, "Subcategory": None, "Accuracy": subscores[0]['val'], "Number few shot": num_shot, "Model": model_name
                    }
                    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                else:
                    row = {
                        "Benchmark": benchmark, "Subcategory": "Average", "Accuracy": sum(s['val'] for s in subscores) / len(subscores), "Number few shot": num_shot, "Model": model_name
                    }
                    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                    for sub in subscores:
                        row = {
                            "Benchmark": None, "Subcategory": sub['subcat'], "Accuracy": sub['val'], "Number few shot": num_shot, "Model": model_name
                        }
                        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    print(f"Printing results for model={model_name}")
    print(df.to_markdown(index=False))


if __name__ == '__main__':
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg = om.merge(yaml_cfg, cli_cfg)
    main(cfg)
