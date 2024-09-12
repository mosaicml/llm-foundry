# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
from contextlib import nullcontext
from typing import Optional
from unittest import mock

import pytest

from llmfoundry.data.finetuning.tasks import dataset_constructor
from llmfoundry.utils.exceptions import DatasetTooSmallError


@pytest.mark.parametrize('num_canonical_nodes', [None, 8, 2])
def test_finetuning_streaming_dataset_too_small(
    num_canonical_nodes: Optional[int],
):
    num_samples = 2

    class MockDataset:

        def __init__(self):
            self.num_canonical_nodes = num_canonical_nodes
            self.num_samples = num_samples

    class MockDist:

        def get_world_size(self):
            return 32

        def get_local_world_size(self):
            return 8

    result_context = nullcontext(
    ) if num_canonical_nodes == 2 else pytest.raises(DatasetTooSmallError)
    with result_context:
        with mock.patch(
            'llmfoundry.data.finetuning.tasks.dist',
            new=MockDist(),
        ):
            with mock.patch(
                'llmfoundry.data.finetuning.tasks.DatasetConstructor.streaming_dataset_class',
                new=MockDataset,
            ):
                dataset_constructor.build_from_streaming()
