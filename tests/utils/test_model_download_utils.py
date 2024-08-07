# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import os
import unittest.mock as mock
from http import HTTPStatus
from typing import Any, Dict, List
from unittest.mock import MagicMock
from urllib.parse import urljoin

import pytest
import requests
import tenacity
from huggingface_hub.utils import RepositoryNotFoundError
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME
from transformers.utils import WEIGHTS_INDEX_NAME as PYTORCH_WEIGHTS_INDEX_NAME
from transformers.utils import WEIGHTS_NAME as PYTORCH_WEIGHTS_NAME

from llmfoundry.utils.model_download_utils import (
    DEFAULT_IGNORE_PATTERNS,
    PYTORCH_WEIGHTS_PATTERN,
    SAFE_WEIGHTS_PATTERN,
    download_from_hf_hub,
    download_from_http_fileserver,
)

# ======================== download_from_hf_hub tests ========================


@pytest.mark.parametrize(
    ['prefer_safetensors', 'repo_files', 'expected_ignore_patterns'],
    [
        [  # Should use default ignore if only safetensors available
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
        [  # Should use default ignore if only sharded safetensors available
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
        [  # Ignore pytorch if safetensors are preferred
            True,
            [PYTORCH_WEIGHTS_NAME, SAFE_WEIGHTS_NAME],
            DEFAULT_IGNORE_PATTERNS + [PYTORCH_WEIGHTS_PATTERN],
        ],
        [  # Ignore safetensors if pytorch is preferred
            False,
            [PYTORCH_WEIGHTS_NAME, SAFE_WEIGHTS_NAME],
            DEFAULT_IGNORE_PATTERNS + [SAFE_WEIGHTS_PATTERN],
        ],
        [  # Ignore pytorch if safetensors are preferred
            True,
            [PYTORCH_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_INDEX_NAME],
            DEFAULT_IGNORE_PATTERNS + [PYTORCH_WEIGHTS_PATTERN],
        ],
        [  # Ignore safetensors if pytorch is preferred
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
    expected_ignore_patterns: List[str],
):
    test_repo_id = 'test_repo_id'
    save_dir = 'save_dir'
    mock_list_repo_files.return_value = repo_files

    download_from_hf_hub(
        test_repo_id,
        save_dir=save_dir,
        prefer_safetensors=prefer_safetensors,
    )
    mock_snapshot_download.assert_called_once_with(
        test_repo_id,
        local_dir=save_dir,
        local_dir_use_symlinks=False,
        allow_patterns=None,
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
    save_dir = 'save_dir'
    mock_list_repo_files.return_value = []

    with pytest.raises(ValueError):
        download_from_hf_hub(test_repo_id, save_dir)

    mock_snapshot_download.assert_not_called()


@pytest.mark.parametrize(['exception', 'expected_attempts'], [
    [requests.exceptions.RequestException(), 3],
    [RepositoryNotFoundError(''), 1],
    [ValueError(), 1],
])
@mock.patch('tenacity.nap.time.sleep')
@mock.patch('huggingface_hub.snapshot_download')
@mock.patch('huggingface_hub.list_repo_files')
def test_download_from_hf_hub_retry(
    mock_list_repo_files: MagicMock,
    mock_snapshot_download: MagicMock,
    mock_sleep: MagicMock,  # so the retry wait doesn't actually wait
    exception: BaseException,
    expected_attempts: int,
):
    mock_list_repo_files.return_value = [SAFE_WEIGHTS_INDEX_NAME]
    mock_snapshot_download.side_effect = exception

    with pytest.raises((tenacity.RetryError, exception.__class__)):
        download_from_hf_hub('test_repo_id', 'save_dir')

    assert mock_snapshot_download.call_count == expected_attempts


# ======================== download_from_http_fileserver tests ========================

ROOT_HTML = b"""
<!DOCTYPE html>
<html>
<body>
    <ul>
        <li><a href="file1">file1</a></li>
        <li><a href="folder/">folder/</a></li>
    </ul>
</body>
</html>
"""

SUBFOLDER_HTML = b"""
<!DOCTYPE html>
<html>
<body>
    <ul>
        <li><a href="file2">file2</a></li>
    </ul>
</body>
</html>
"""


@mock.patch.object(requests.Session, 'get')
@mock.patch('os.makedirs')
@mock.patch('builtins.open')
def test_download_from_http_fileserver(
    mock_open: MagicMock,
    mock_makedirs: MagicMock,
    mock_get: MagicMock,
):
    model_url = f'https://cache.com/models/model/'
    save_dir = 'save_dir/'

    mock_open.return_value = MagicMock()

    def _server_response(url: str, **kwargs: Dict[str, Any]):
        if url == model_url:
            return MagicMock(status_code=HTTPStatus.OK, content=ROOT_HTML)
        if url == urljoin(model_url, 'file1'):
            return MagicMock(status_code=HTTPStatus.OK)
        elif url == urljoin(model_url, 'folder/'):
            return MagicMock(status_code=HTTPStatus.OK, content=SUBFOLDER_HTML)
        elif url == urljoin(model_url, 'folder/file2'):
            return MagicMock(status_code=HTTPStatus.OK)
        else:
            return MagicMock(status_code=HTTPStatus.NOT_FOUND)

    mock_get.side_effect = _server_response
    download_from_http_fileserver(model_url, save_dir)

    mock_open.assert_has_calls(
        [
            mock.call(os.path.join(save_dir, 'file1'), 'wb'),
            mock.call(os.path.join(save_dir, 'folder/file2'), 'wb'),
        ],
        any_order=True,
    )


@mock.patch.object(requests.Session, 'get')
def test_download_from_http_fileserver_unauthorized(mock_get: MagicMock):
    model_name = 'model'
    cache_url = f'https://cache.com/models--{model_name}/blobs/'
    save_dir = 'save_dir/'

    mock_get.return_value = MagicMock(status_code=HTTPStatus.UNAUTHORIZED)
    with pytest.raises(PermissionError):
        download_from_http_fileserver(cache_url, save_dir)


@pytest.mark.parametrize(['exception', 'expected_attempts'], [
    [requests.exceptions.RequestException(), 3],
    [PermissionError(), 1],
    [ValueError(), 1],
])
@mock.patch('tenacity.nap.time.sleep')
@mock.patch('llmfoundry.utils.model_download_utils._recursive_download')
def test_download_from_http_fileserver_retry(
    mock_recursive_download: MagicMock,
    mock_sleep: MagicMock,  # so the retry wait doesn't actually wait
    exception: BaseException,
    expected_attempts: int,
):
    mock_recursive_download.side_effect = exception

    with pytest.raises((tenacity.RetryError, exception.__class__)):
        download_from_http_fileserver('cache_url', 'save_dir')
