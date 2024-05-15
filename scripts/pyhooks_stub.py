# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

# git clone git@github.com:databricks-mosaic/mcloud.git to /tmp/mcloud
# git checkout pyhookbuffered
import os
import shutil
import subprocess
import sys
import time

# execute some shell commands

rank = int(os.environ.get('RANK')) % 8 # hack

if rank == 0:
    shutil.rmtree(f'/tmp/mcloud-{rank}', ignore_errors=True)
    subprocess.run([
        'git',
        'clone',
        'git@github.com:databricks-mosaic/mcloud.git',
        f'/tmp/mcloud-{rank}',
    ],
                   check=True)
    subprocess.run(['git', 'checkout', 'pyhookbuffered'],
                   cwd=f'/tmp/mcloud-{rank}',
                   check=True)
    subprocess.run(['pip', 'install', 'fickling'], check=True)

sys.path.append(f'/tmp/mcloud-0/finetuning/')

# extremely primitive dist barrier
while True:
    if os.path.exists(
        '/tmp/mcloud-0/finetuning/pyhook_scripts/setup_pyhook.py',
    ):
        break
    else:
        time.sleep(1)
