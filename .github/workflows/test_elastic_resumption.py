# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import datetime
import os
import subprocess
import logging

log = logging.getLogger(__name__)

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
REGRESSIONS_DIR = os.path.join(DIR_PATH, 'regression_yamls')

from mcli import RunConfig, RunStatus, create_run, wait_for_run_status

def test_elastic_resumption(cluster: str, save_folder: str, wandb_entity: str,
                            wandb_project: str, git_repo: str, git_branch: str):
    def create_run_and_wait(gpus: int, resume: bool, subdir: str):
        config = RunConfig.from_file(
            os.path.join(REGRESSIONS_DIR, 'mpt-125m-elastic-resumption.yaml'))
        
        # Add the command to train our model
        composer_command = '\ncomposer train/train.py /mnt/config/parameters.yaml'
        if resume:
            composer_command += ' autoresume=true' # TODO: autoresume and save_overwrite can't both be true, but i have to overwrite if i run multiple runs with same save folder
        else:
            composer_command += ' save_overwrite=true autoresume=false'
        config.command += composer_command

        # Add suffix to name
        name_suffix = f'-{gpus}'
        if resume:
            name_suffix += '-resume'
        config.name += name_suffix

        # Set other parameters
        config.cluster = cluster
        config.compute['gpus'] = gpus
        config.parameters['save_folder'] = os.path.join(save_folder, subdir)
        config.parameters['max_duration'] = '20ba' if resume else '10ba'
        commit_hash = subprocess.check_output(['git', 'rev-parse',
                                            'HEAD']).strip().decode('utf-8')
        timestamp = datetime.datetime.now().strftime('%m-%d-%Y::%H:%M:%S')
        wandb_group = f'{timestamp}::{commit_hash}'
        wandb_config = {
            'entity': wandb_entity,
            'project': wandb_project,
            'group': wandb_group
        }
        config.parameters['loggers'] = config.parameters.get('loggers', {})
        config.parameters['loggers']['wandb'] = wandb_config
        config.integrations[0]['git_repo'] = git_repo
        config.integrations[0]['git_branch'] = git_branch

        # Start run
        run = create_run(config)
        log.info(f'Starting run {run.name}')
        wait_for_run_status(run, RunStatus.COMPLETED) # Wait for the run to complete or terminate.
        if run.status != RunStatus.COMPLETED:
            raise Exception(f'Failure on run {run.name}. Run status is {run.status}. Terminating test.')
        log.info(f'Completed run {run.name}')
    
    # Test 1 node => 2 node elastic resumption
    subdir = '1_to_2_node'
    create_run_and_wait(gpus=8, resume=False, subdir=subdir)
    create_run_and_wait(gpus=16, resume=True, subdir=subdir)

    # Test 2 node => 1 node elastic resumption
    subdir = '2_to_1_node'
    create_run_and_wait(gpus=16, resume=False, subdir=subdir)
    create_run_and_wait(gpus=8, resume=True, subdir=subdir)

if __name__ == '__main__':
    # TODO: Either call the above function in regressions or put an entry point here.
