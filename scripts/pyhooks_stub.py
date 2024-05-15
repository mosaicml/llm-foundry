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

rank = os.environ.get('RANK')
if rank is None:
    raise ValueError('RANK not set')

if rank == '0':
    shutil.rmtree('/tmp/mcloud', ignore_errors=True)
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

while True:
    try:
        import pyhook
        break
    except:
        print('Failed to import pyhook from mcloud')
        time.sleep(5)
