# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import gc
import os
import sys

import pytest
import torch
from composer.utils import dist, get_device, reproducibility

# Add llm-foundry repo root to path so we can import scripts in the tests
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(REPO_DIR)


@pytest.fixture(autouse=True)
def initialize_dist(request: pytest.FixtureRequest):
    """Initialize the default PyTorch distributed process group for tests."""
    # should we just always initialize dist like in train.py?
    _default = pytest.mark.world_size(1).mark
    world_size = request.node.get_closest_marker('world_size', _default).args[0]
    gpu = request.node.get_closest_marker('gpu')
    if world_size > 1:
        dist.initialize_dist(get_device('gpu' if gpu is not None else 'cpu'))


@pytest.fixture(autouse=True)
def clear_cuda_cache(request: pytest.FixtureRequest):
    """Clear memory between GPU tests."""
    marker = request.node.get_closest_marker('gpu')
    if marker is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()  # Only gc on GPU tests as it 2x slows down CPU tests


@pytest.fixture
def random_seed() -> int:
    return 17


@pytest.fixture
def foundry_dir() -> str:
    return REPO_DIR


@pytest.fixture(autouse=True)
def seed_all(random_seed: int):
    """Sets the seed for reproducibility."""
    reproducibility.seed_all(random_seed)
