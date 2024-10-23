# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import tempfile

from composer.utils import dist


def dist_mkdtemp() -> str:
    """Creates a temp directory on local rank 0 to use for other ranks.

    Returns:
        str: The path to the temporary directory.
    """
    tempdir = None
    if dist.get_local_rank() == 0:
        tempdir = tempfile.mkdtemp()
    tempdir = dist.all_gather_object(tempdir)[0]
    if tempdir is None:
        raise RuntimeError('Dist operation to get tempdir failed.')
    return tempdir
