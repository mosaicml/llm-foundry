# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys

# execute some shell commands

# git clone git@github.com:databricks-mosaic/mcloud.git to /tmp/mcloud
# git checkout pyhookbuffered
try:
    subprocess.run([
        'git',
        'clone',
        'git@github.com:databricks-mosaic/mcloud.git',
        '/tmp/mcloud',
    ],
                   check=True)
    subprocess.run(['git', 'checkout', 'pyhookbuffered'],
                   cwd='/tmp/mcloud',
                   check=True)
except Exception as e:
    print(
        f'Failed to clone mcloud: {e}, probably because it already exists, no big deal.',
    )

# add the following to the PYTHONPATH:
# /tmp/mcloud/finetuning/pyhook
sys.path.append('/tmp/mcloud/finetuning/')

try:
    import pyhook
except:
    raise ImportError('Failed to import pyhook from mcloud')
