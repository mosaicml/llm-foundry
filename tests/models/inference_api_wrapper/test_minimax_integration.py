# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for MiniMax inference API wrapper.

These tests verify the MiniMax wrapper works correctly with the
MiniMax API. They require a valid MINIMAX_API_KEY environment variable.

Run with: pytest tests/models/inference_api_wrapper/test_minimax_integration.py -v
"""

import os

import pytest
from omegaconf import DictConfig
from unittest.mock import MagicMock


@pytest.fixture(autouse=True)
def clean_env():
    """Save and restore env vars around each test."""
    old_openai = os.environ.get('OPENAI_API_KEY')
    old_minimax = os.environ.get('MINIMAX_API_KEY')
    yield
    if old_openai is not None:
        os.environ['OPENAI_API_KEY'] = old_openai
    else:
        os.environ.pop('OPENAI_API_KEY', None)
    if old_minimax is not None:
        os.environ['MINIMAX_API_KEY'] = old_minimax
    else:
        os.environ.pop('MINIMAX_API_KEY', None)


@pytest.mark.skipif(
    not os.environ.get('MINIMAX_API_KEY'),
    reason='MINIMAX_API_KEY not set',
)
@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_minimax_chat_completion_live():
    """Integration test: verify a real chat completion call to MiniMax API."""
    openai = pytest.importorskip('openai')

    os.environ.pop('OPENAI_API_KEY', None)

    from llmfoundry.models.inference_api_wrapper.minimax import (
        MiniMaxChatAPIEvalWrapper,
    )

    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 1

    model = MiniMaxChatAPIEvalWrapper(
        om_model_config=DictConfig({
            'version': 'MiniMax-M3',
        }),
        tokenizer=mock_tokenizer,
    )

    # Make a real API call
    completion = model.client.chat.completions.create(
        model='MiniMax-M3',
        messages=[{
            'role': 'user',
            'content': 'Say "hello" and nothing else.',
        }],
        max_tokens=10,
        temperature=0.0,
    )

    assert completion is not None
    assert len(completion.choices) > 0
    assert completion.choices[0].message.content is not None
    assert len(completion.choices[0].message.content) > 0


@pytest.mark.skipif(
    not os.environ.get('MINIMAX_API_KEY'),
    reason='MINIMAX_API_KEY not set',
)
@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_minimax_m27_highspeed_live():
    """Integration test: verify MiniMax-M2.7-highspeed model works."""
    openai = pytest.importorskip('openai')

    os.environ.pop('OPENAI_API_KEY', None)

    from llmfoundry.models.inference_api_wrapper.minimax import (
        MiniMaxChatAPIEvalWrapper,
    )

    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 1

    model = MiniMaxChatAPIEvalWrapper(
        om_model_config=DictConfig({
            'version': 'MiniMax-M2.7-highspeed',
        }),
        tokenizer=mock_tokenizer,
    )

    completion = model.client.chat.completions.create(
        model='MiniMax-M2.7-highspeed',
        messages=[{
            'role': 'user',
            'content': 'Say "hello world" and nothing else.',
        }],
        max_tokens=20,
        temperature=0.0,
    )

    assert completion is not None
    assert len(completion.choices) > 0
    assert completion.choices[0].message.content is not None
    assert len(completion.choices[0].message.content) > 0


@pytest.mark.skipif(
    not os.environ.get('MINIMAX_API_KEY'),
    reason='MINIMAX_API_KEY not set',
)
@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_minimax_streaming_live():
    """Integration test: verify streaming chat completion works."""
    openai = pytest.importorskip('openai')

    os.environ.pop('OPENAI_API_KEY', None)

    from llmfoundry.models.inference_api_wrapper.minimax import (
        MiniMaxChatAPIEvalWrapper,
    )

    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 1

    model = MiniMaxChatAPIEvalWrapper(
        om_model_config=DictConfig({
            'version': 'MiniMax-M3',
        }),
        tokenizer=mock_tokenizer,
    )

    stream = model.client.chat.completions.create(
        model='MiniMax-M3',
        messages=[{
            'role': 'user',
            'content': 'Say "test" and nothing else.',
        }],
        max_tokens=10,
        temperature=0.0,
        stream=True,
    )

    chunks = list(stream)
    assert len(chunks) > 0
