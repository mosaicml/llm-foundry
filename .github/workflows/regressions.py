# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import argparse
import datetime
import os
import subprocess

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
REGRESSIONS_DIR = os.path.join(DIR_PATH, 'regression_yamls')

from mcli import RunConfig, create_run


def get_configs(cluster: str, mpt_7b_ckpt_path: str, wandb_entity: str,
                wandb_project: str):
    eval_7b_hf = RunConfig.from_file(
        os.path.join(REGRESSIONS_DIR, 'eval-7b-hf.yaml'))
    eval_7b_composer = RunConfig.from_file(
        os.path.join(REGRESSIONS_DIR, 'eval-7b-composer.yaml'))
    llama2_finetune = RunConfig.from_file(
        os.path.join(REGRESSIONS_DIR, 'llama2-finetune.yaml'))
    mpt_125m_chinchilla = RunConfig.from_file(
        os.path.join(REGRESSIONS_DIR, 'mpt-125m-chinchilla.yaml'))
    mpt_125m_sharded_resumption = RunConfig.from_file(
        os.path.join(REGRESSIONS_DIR, 'mpt-125m-sharded-resumption.yaml'))

    # make specific changes
    eval_7b_composer.parameters['models'][0]['load_path'] = mpt_7b_ckpt_path

    all_configs = [
        eval_7b_hf, eval_7b_composer, llama2_finetune, mpt_125m_chinchilla,
        mpt_125m_sharded_resumption
    ]

    commit_hash = subprocess.check_output(['git', 'rev-parse',
                                           'HEAD']).strip().decode('utf-8')
    timestamp = datetime.datetime.now().strftime('%m-%d-%Y::%H:%M:%S')
    wandb_group = f'{timestamp}::{commit_hash}'

    # make general changes
    wandb_config = {
        'entity': wandb_entity,
        'project': wandb_project,
        'group': wandb_group
    }
    for config in all_configs:
        config.cluster = cluster
        config.parameters['loggers'] = config.parameters.get('loggers', {})
        config.parameters['loggers']['wandb'] = wandb_config

    return all_configs, []


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cluster', type=str)
    parser.add_argument('--mpt-7b-ckpt-path', type=str)
    parser.add_argument('--wandb-entity', type=str)
    parser.add_argument('--wandb-project', type=str)

    args = parser.parse_args()

    run_configs, _ = get_configs(args.cluster, args.mpt_7b_ckpt_path,
                                 args.wandb_entity, args.wandb_project)
    for run_config in run_configs:
        run = create_run(run_config)
