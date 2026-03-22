# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import DictConfig

from llmfoundry.models.inference_api_wrapper.minimax import (
    MINIMAX_API_BASE_URL,
    MiniMaxChatAPIEvalWrapper,
    MiniMaxEvalInterface,
)


@pytest.fixture(autouse=True)
def clean_env():
    """Save and restore env vars around each test."""
    old_openai = os.environ.get('OPENAI_API_KEY')
    old_minimax = os.environ.get('MINIMAX_API_KEY')
    yield
    # Restore
    if old_openai is not None:
        os.environ['OPENAI_API_KEY'] = old_openai
    else:
        os.environ.pop('OPENAI_API_KEY', None)
    if old_minimax is not None:
        os.environ['MINIMAX_API_KEY'] = old_minimax
    else:
        os.environ.pop('MINIMAX_API_KEY', None)


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_minimax_api_base_url_constant():
    """Test that the MiniMax API base URL constant is correct."""
    assert MINIMAX_API_BASE_URL == 'https://api.minimax.io/v1'


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_minimax_model_registered():
    """Test that minimax_chat is registered in the models registry."""
    from llmfoundry.registry import models

    assert 'minimax_chat' in models.get_all()


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_minimax_missing_api_key():
    """Test that missing API key raises ValueError."""
    _ = pytest.importorskip('openai')

    os.environ.pop('OPENAI_API_KEY', None)
    os.environ.pop('MINIMAX_API_KEY', None)

    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 1

    with pytest.raises(ValueError, match='No MiniMax API key found'):
        MiniMaxChatAPIEvalWrapper(
            om_model_config=DictConfig({
                'version': 'MiniMax-M2.7',
            }),
            tokenizer=mock_tokenizer,
        )


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_minimax_uses_minimax_api_key():
    """Test that MINIMAX_API_KEY is used when OPENAI_API_KEY is not set."""
    _ = pytest.importorskip('openai')

    os.environ.pop('OPENAI_API_KEY', None)
    os.environ['MINIMAX_API_KEY'] = 'test-minimax-key-123'

    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 1

    model = MiniMaxChatAPIEvalWrapper(
        om_model_config=DictConfig({
            'version': 'MiniMax-M2.7',
        }),
        tokenizer=mock_tokenizer,
    )
    assert model.model_name == 'MiniMax-M2.7'
    assert model.client.base_url.host == 'api.minimax.io'


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_minimax_uses_openai_key_if_set():
    """Test that OPENAI_API_KEY takes precedence when already set."""
    _ = pytest.importorskip('openai')

    os.environ['OPENAI_API_KEY'] = 'existing-openai-key'
    os.environ.pop('MINIMAX_API_KEY', None)

    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 1

    model = MiniMaxChatAPIEvalWrapper(
        om_model_config=DictConfig({
            'version': 'MiniMax-M2.7',
        }),
        tokenizer=mock_tokenizer,
    )
    # Should succeed without MINIMAX_API_KEY when OPENAI_API_KEY is set
    assert model.model_name == 'MiniMax-M2.7'


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_minimax_default_base_url():
    """Test that the MiniMax wrapper sets the correct default base URL."""
    _ = pytest.importorskip('openai')

    os.environ['MINIMAX_API_KEY'] = 'test-minimax-key'
    os.environ.pop('OPENAI_API_KEY', None)

    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 1

    model = MiniMaxChatAPIEvalWrapper(
        om_model_config=DictConfig({
            'version': 'MiniMax-M2.7',
        }),
        tokenizer=mock_tokenizer,
    )
    assert model.client.base_url.host == 'api.minimax.io'
    assert '/v1' in str(model.client.base_url)


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_minimax_custom_base_url():
    """Test that a custom base URL overrides the default MiniMax URL."""
    _ = pytest.importorskip('openai')

    os.environ['MINIMAX_API_KEY'] = 'test-minimax-key'
    os.environ.pop('OPENAI_API_KEY', None)

    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 1

    model = MiniMaxChatAPIEvalWrapper(
        om_model_config=DictConfig({
            'version': 'MiniMax-M2.7',
            'base_url': 'https://custom.minimax.io/v1',
        }),
        tokenizer=mock_tokenizer,
    )
    assert 'custom.minimax.io' in str(model.client.base_url)


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_minimax_model_name_from_version():
    """Test that model_name is set from the version config field."""
    _ = pytest.importorskip('openai')

    os.environ['MINIMAX_API_KEY'] = 'test-minimax-key'
    os.environ.pop('OPENAI_API_KEY', None)

    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 1

    model = MiniMaxChatAPIEvalWrapper(
        om_model_config=DictConfig({
            'version': 'MiniMax-M2.5-highspeed',
        }),
        tokenizer=mock_tokenizer,
    )
    assert model.model_name == 'MiniMax-M2.5-highspeed'


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_minimax_model_name_from_name_field():
    """Test model_name fallback to name field when version is not set."""
    _ = pytest.importorskip('openai')

    os.environ['MINIMAX_API_KEY'] = 'test-minimax-key'
    os.environ.pop('OPENAI_API_KEY', None)

    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 1

    model = MiniMaxChatAPIEvalWrapper(
        om_model_config=DictConfig({
            'name': 'MiniMax-M2.7',
        }),
        tokenizer=mock_tokenizer,
    )
    assert model.model_name == 'MiniMax-M2.7'


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_minimax_inherits_chat_api_methods():
    """Test MiniMax wrapper inherits OpenAI chat API behavior correctly."""
    _ = pytest.importorskip('openai')

    os.environ['MINIMAX_API_KEY'] = 'test-minimax-key'
    os.environ.pop('OPENAI_API_KEY', None)

    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 1

    chatmodel = MiniMaxChatAPIEvalWrapper(
        om_model_config=DictConfig({
            'version': 'MiniMax-M2.7',
        }),
        tokenizer=mock_tokenizer,
    )

    # Verify the model has the expected attributes from OpenAIChatAPIEvalWrapper
    assert hasattr(chatmodel, 'client')
    assert hasattr(chatmodel, 'model_name')
    assert hasattr(chatmodel, 'tokenizer')
    assert hasattr(chatmodel, 'generate_completion')
    assert hasattr(chatmodel, 'eval_forward')
    assert hasattr(chatmodel, 'rebatch')
    assert hasattr(chatmodel, 'process_result')
    assert hasattr(chatmodel, 'try_generate_completion')
