# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import argparse
import datetime
import os
import subprocess

from mcli import RunConfig, RunStatus, create_run, wait_for_run_status

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
REGRESSIONS_DIR = os.path.join(DIR_PATH, 'regression_yamls')

COMMIT_HASH = subprocess.check_output(['git', 'rev-parse',
                                       'HEAD']).strip().decode('utf-8')
TIMESTAMP = datetime.datetime.now().strftime('%m-%d-%Y::%H:%M:%S')


def _get_regression_config(yaml_name: str) -> RunConfig:
    """Get the yaml config from regressions directory."""
    return RunConfig.from_file(os.path.join(REGRESSIONS_DIR, yaml_name))


def _set_general_configs(config: RunConfig, cluster: str, wandb_entity: str,
                         wandb_project: str, git_repo: str, git_branch: str):
    """Set general configuration arguments."""
    config.cluster = cluster
    wandb_group = f'{TIMESTAMP}::{COMMIT_HASH}'
    wandb_config = {
        'entity': wandb_entity,
        'project': wandb_project,
        'group': wandb_group
    }
    config.parameters['loggers'] = config.parameters.get('loggers', {})
    config.parameters['loggers']['wandb'] = wandb_config
    config.integrations[0]['git_repo'] = git_repo
    config.integrations[0]['git_branch'] = git_branch


def test_elastic_resumption(cluster: str, save_folder: str, wandb_entity: str,
                            wandb_project: str, git_repo: str, git_branch: str):
    """Regression test for elastic resumption."""

    def create_run_and_wait(gpus: int, resume: bool, subdir: str):
        config = _get_regression_config('mpt-125m-elastic-resumption.yaml')

        # Add the command to train our model
        composer_command = '\ncomposer train/train.py /mnt/config/parameters.yaml'
        if resume:
            composer_command += ' autoresume=true'
        else:
            composer_command += ' autoresume=false'
        config.command += composer_command

        # Add suffix to name
        name_suffix = f'-{gpus}'
        if resume:
            name_suffix += '-resume'
        config.name += name_suffix

        # Set other parameters
        config.compute['gpus'] = gpus
        config.parameters['save_folder'] = os.path.join(save_folder, subdir)
        config.parameters['max_duration'] = '20ba' if resume else '10ba'

        _set_general_configs(config,
                             cluster=cluster,
                             wandb_entity=wandb_entity,
                             wandb_project=wandb_project,
                             git_repo=git_repo,
                             git_branch=git_branch)

        # Start run
        run = create_run(config)
        wait_for_run_status(
            run,
            RunStatus.COMPLETED)  # Wait for the run to complete or terminate.
        if run.status != RunStatus.COMPLETED:
            raise Exception(
                f'Failure on run {run.name}. Run status is {run.status}. ' +
                'Terminating elastic resumption regression test.')

    # Test 1 node => 2 node elastic resumption
    subdir = f'1_to_2_node_{TIMESTAMP}_{COMMIT_HASH}'
    create_run_and_wait(gpus=8, resume=False, subdir=subdir)
    create_run_and_wait(gpus=16, resume=True, subdir=subdir)

    # Test 2 node => 1 node elastic resumption
    subdir = f'2_to_1_node_{TIMESTAMP}_{COMMIT_HASH}'
    create_run_and_wait(gpus=16, resume=False, subdir=subdir)
    create_run_and_wait(gpus=8, resume=True, subdir=subdir)


def test_basic(cluster: str, mpt_7b_ckpt_path: str, wandb_entity: str,
               wandb_project: str, git_repo: str, git_branch: str):
    eval_7b_hf = _get_regression_config('eval-7b-hf.yaml')
    eval_7b_composer = _get_regression_config('eval-7b-composer.yaml')
    llama2_finetune = _get_regression_config('llama2-finetune.yaml')
    mpt_125m_chinchilla = _get_regression_config('mpt-125m-chinchilla.yaml')
    mpt_125m_sharded_resumption = _get_regression_config(
        'mpt-125m-sharded-resumption.yaml')

    # make specific changes
    eval_7b_composer.parameters['models'][0]['load_path'] = mpt_7b_ckpt_path

    all_configs = [
        eval_7b_hf, eval_7b_composer, llama2_finetune, mpt_125m_chinchilla,
        mpt_125m_sharded_resumption
    ]

    for config in all_configs:
        _set_general_configs(config,
                             cluster=cluster,
                             wandb_entity=wandb_entity,
                             wandb_project=wandb_project,
                             git_repo=git_repo,
                             git_branch=git_branch)
        create_run(config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cluster', type=str)
    parser.add_argument('--mpt-7b-ckpt-path', type=str)
    parser.add_argument('--wandb-entity', type=str)
    parser.add_argument('--wandb-project', type=str)
    parser.add_argument('--remote-save-folder', type=str)
    parser.add_argument('--git-repo', type=str, default='mosaicml/llm-foundry')
    parser.add_argument('--git-branch', type=str, default='main')

    args = parser.parse_args()

    print(f'Running regression tests on {args.git_repo} {args.git_branch}.')

    test_basic(args.cluster, args.mpt_7b_ckpt_path, args.wandb_entity,
               args.wandb_project, args.git_repo, args.git_branch)
    test_elastic_resumption(args.cluster, args.remote_save_folder,
                            args.wandb_entity, args.wandb_project,
                            args.git_repo, args.git_branch)
