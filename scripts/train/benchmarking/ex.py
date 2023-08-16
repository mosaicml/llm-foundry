# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
import argparse
import math
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
import yaml
from mcli.models.run_config import SchedulingConfig
from mcli.sdk import RunConfig, create_run, get_clusters, follow_run_logs

config = RunConfig(
    name='hello-composer',
    gpu_num=1,
    cluster="r1z2",
    image='mosaicml/pytorch:1.13.1_cu117-python3.10-ubuntu20.04',
    integrations=[
        {
         'integration_type': 'git_repo',
         'git_repo': 'mosaicml/llm-foundry',
        #  'git_branch': 'main',
         'pip_install': '-e .[gpu]',
        #  'ssh_clone': 'false'
        }
        # , {
        #     'integration_type': 'wandb',
        #     'entity': 'mosaic-ml',
        #     'project': 'chris-scripting'
        # }
    ],
    command="""
    cd llm-foundry/scripts
    composer train/train.py train/yamls/pretrain/mpt-1b.yaml
    """,
    scheduling={'priority': 'lowest','resumable': True}
    # data_remote='oci://mosaicml-internal-dataset-c4/preconcat-gpt_neox/',
    # max_duration=100
)

run = create_run(config)