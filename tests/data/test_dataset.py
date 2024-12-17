# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
import os
from contextlib import nullcontext
from typing import Optional
from unittest import mock

import pytest

from llmfoundry.data.finetuning.tasks import (
    QA_format_preprocessor,
    _get_num_processes,
    dataset_constructor,
    messages_format_preprocessor,
)
from llmfoundry.utils.exceptions import DatasetTooSmallError


def test_get_num_processes():
    with mock.patch.dict(os.environ, {'MAX_NUM_PROC': '4'}):
        with mock.patch('os.cpu_count', return_value=16):
            assert _get_num_processes() == 4

    with mock.patch.dict(os.environ, {'MAX_NUM_PROC': '32'}):
        with mock.patch('os.cpu_count', return_value=16):
            assert _get_num_processes() == 8

    with mock.patch.dict(os.environ, {}):
        with mock.patch('os.cpu_count', return_value=16):
            assert _get_num_processes() == 8


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


def test_QA_format_preprocessor():
    inp = {
        'Q': 'What is the capital of France?',
        'A': 'Paris',
        'meta': {
            'a': 'b',
        },
    }

    expected_messages = [{
        'role': 'user',
        'content': 'What is the capital of France?',
    }, {
        'role': 'assistant',
        'content': 'Paris',
    }]
    output = QA_format_preprocessor(inp)
    assert len(output) == 1
    assert 'messages' in output
    for i, message in enumerate(output['messages']):
        expected_message = expected_messages[i]
        for k, v in message.items():
            assert k in expected_message
            assert v == expected_message[k]


def test_messages_format_preprocessor():
    messages = [{
        'role': 'user',
        'content': 'What is the capital of France?',
    }, {
        'role': 'assistant',
        'content': 'Paris',
    }]
    inp = {
        'messages': messages,
        'other_key': 'other_value',
    }

    output = messages_format_preprocessor(inp)
    assert len(output) == 1
    assert 'messages' in output
    assert output['messages'] == messages
