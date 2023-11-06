# Copyright 2023 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from typing import List

import unittest.mock as mock
from unittest.mock import MagicMock

import pytest

from llmfoundry.utils.model_download_utils import (
    download_from_cache_server,
    download_from_hf_hub,
    PYTORCH_WEIGHTS_NAME,
    PYTORCH_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    DEFAULT_IGNORE_PATTERNS,
    PYTORCH_WEIGHTS_PATTERN,
    SAFE_WEIGHTS_PATTERN
)


@pytest.mark.parametrize(['prefer_safetensors', 'repo_files', 'expected_ignore_patterns'], [
    [   # Should use default ignore if only safetensors available
        True,
        [SAFE_WEIGHTS_NAME],
        DEFAULT_IGNORE_PATTERNS,
    ],
    [
        # Should use default ignore if only safetensors available
        False,
        [SAFE_WEIGHTS_NAME],
        DEFAULT_IGNORE_PATTERNS,
    ],
    [   # Should use default ignore if only sharded safetensors available
        True,
        [SAFE_WEIGHTS_INDEX_NAME],
        DEFAULT_IGNORE_PATTERNS,
    ],
    [
        # Should use default ignore if only sharded safetensors available
        False,
        [SAFE_WEIGHTS_INDEX_NAME],
        DEFAULT_IGNORE_PATTERNS,
    ],
    [
        # Should use default ignore if only pytorch available
        True,
        [PYTORCH_WEIGHTS_NAME],
        DEFAULT_IGNORE_PATTERNS,
    ],
    [
        # Should use default ignore if only pytorch available
        False,
        [PYTORCH_WEIGHTS_NAME],
        DEFAULT_IGNORE_PATTERNS,
    ],
    [
        # Should use default ignore if only sharded pytorch available
        True,
        [PYTORCH_WEIGHTS_INDEX_NAME],
        DEFAULT_IGNORE_PATTERNS,
    ],
    [
        # Should use default ignore if only sharded pytorch available
        False,
        [PYTORCH_WEIGHTS_INDEX_NAME],
        DEFAULT_IGNORE_PATTERNS,
    ],
    [   # Ignore pytorch if safetensors are preferred
        True,
        [PYTORCH_WEIGHTS_NAME, SAFE_WEIGHTS_NAME],
        DEFAULT_IGNORE_PATTERNS + [PYTORCH_WEIGHTS_PATTERN],
    ],
    [   # Ignore safetensors if pytorch is preferred
        False,
        [PYTORCH_WEIGHTS_NAME, SAFE_WEIGHTS_NAME],
        DEFAULT_IGNORE_PATTERNS + [SAFE_WEIGHTS_PATTERN],
    ],
    [   # Ignore pytorch if safetensors are preferred
        True,
        [PYTORCH_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_INDEX_NAME],
        DEFAULT_IGNORE_PATTERNS + [PYTORCH_WEIGHTS_PATTERN],
    ],
    [   # Ignore safetensors if pytorch is preferred
        False,
        [PYTORCH_WEIGHTS_NAME, SAFE_WEIGHTS_NAME],
        DEFAULT_IGNORE_PATTERNS + [SAFE_WEIGHTS_PATTERN],
    ],
])
@mock.patch('huggingface_hub.snapshot_download')
@mock.patch('huggingface_hub.list_repo_files')
def test_download_from_hf_hub_weights_pref(
    mock_list_repo_files: MagicMock,
    mock_snapshot_download: MagicMock,
    prefer_safetensors: bool,
    repo_files: List[str],
    expected_ignore_patterns: List[str]
):
    test_repo_id = 'test_repo_id'
    mock_list_repo_files.return_value = repo_files

    download_from_hf_hub(test_repo_id, prefer_safetensors=prefer_safetensors)
    mock_snapshot_download.assert_called_once_with(
        test_repo_id,
        cache_dir=None,
        ignore_patterns=expected_ignore_patterns,
        token=None,
    )


@mock.patch('huggingface_hub.snapshot_download')
@mock.patch('huggingface_hub.list_repo_files')
def test_download_from_hf_hub_no_weights(
    mock_list_repo_files: MagicMock,
    mock_snapshot_download: MagicMock,
):
    test_repo_id = 'test_repo_id'
    mock_list_repo_files.return_value = []

    with pytest.raises(ValueError):
        download_from_hf_hub(test_repo_id)

    mock_snapshot_download.assert_not_called()
