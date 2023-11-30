# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for downloading models."""
import copy
import logging
import os
import time
import warnings
from http import HTTPStatus
from typing import Optional
from urllib.parse import urljoin

import huggingface_hub as hf_hub
import requests
import tenacity
from bs4 import BeautifulSoup
from requests.packages.urllib3.exceptions import InsecureRequestWarning
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME
from transformers.utils import WEIGHTS_INDEX_NAME as PYTORCH_WEIGHTS_INDEX_NAME
from transformers.utils import WEIGHTS_NAME as PYTORCH_WEIGHTS_NAME

DEFAULT_IGNORE_PATTERNS = [
    '*.ckpt',
    '*.h5',
    '*.msgpack',
]
PYTORCH_WEIGHTS_PATTERN = 'pytorch_model*.bin*'
SAFE_WEIGHTS_PATTERN = 'model*.safetensors*'

log = logging.getLogger(__name__)


@tenacity.retry(retry=tenacity.retry_if_not_exception_type(
    (ValueError, hf_hub.utils.RepositoryNotFoundError)),
                stop=tenacity.stop_after_attempt(3),
                wait=tenacity.wait_exponential(min=1, max=10))
def download_from_hf_hub(
    repo_id: str,
    save_dir: Optional[str] = None,
    prefer_safetensors: bool = True,
    token: Optional[str] = None,
):
    """Downloads model files from a Hugging Face Hub model repo.

    Only supports models stored in Safetensors and PyTorch formats for now. If both formats are available, only the
    Safetensors weights will be downloaded unless `prefer_safetensors` is set to False.

    Args:
        repo_id (str): The Hugging Face Hub repo ID.
        save_dir (str, optional): The path to the directory where the model files will be downloaded. If `None`, reads
            from the `HUGGINGFACE_HUB_CACHE` environment variable or uses the default Hugging Face Hub cache directory.
        prefer_safetensors (bool): Whether to prefer Safetensors weights over PyTorch weights if both are
            available. Defaults to True.
        token (str, optional): The HuggingFace API token. If not provided, the token will be read from the
            `HUGGING_FACE_HUB_TOKEN` environment variable.

    Raises:
        RepositoryNotFoundError: If the model repo doesn't exist or the token is unauthorized.
        ValueError: If the model repo doesn't contain any supported model weights.
    """
    repo_files = set(hf_hub.list_repo_files(repo_id))

    # Ignore TensorFlow, TensorFlow 2, and Flax weights as they are not supported by Composer.
    ignore_patterns = copy.deepcopy(DEFAULT_IGNORE_PATTERNS)

    safetensors_available = (SAFE_WEIGHTS_NAME in repo_files or
                             SAFE_WEIGHTS_INDEX_NAME in repo_files)
    pytorch_available = (PYTORCH_WEIGHTS_NAME in repo_files or
                         PYTORCH_WEIGHTS_INDEX_NAME in repo_files)

    if safetensors_available and pytorch_available:
        if prefer_safetensors:
            log.info(
                'Safetensors available and preferred. Excluding pytorch weights.'
            )
            ignore_patterns.append(PYTORCH_WEIGHTS_PATTERN)
        else:
            log.info(
                'Pytorch available and preferred. Excluding safetensors weights.'
            )
            ignore_patterns.append(SAFE_WEIGHTS_PATTERN)
    elif safetensors_available:
        log.info('Only safetensors available. Ignoring weights preference.')
    elif pytorch_available:
        log.info('Only pytorch available. Ignoring weights preference.')
    else:
        raise ValueError(
            f'No supported model weights found in repo {repo_id}.' +
            ' Please make sure the repo contains either safetensors or pytorch weights.'
        )

    download_start = time.time()
    hf_hub.snapshot_download(repo_id,
                             cache_dir=save_dir,
                             ignore_patterns=ignore_patterns,
                             token=token)
    download_duration = time.time() - download_start
    log.info(
        f'Downloaded model {repo_id} from Hugging Face Hub in {download_duration} seconds'
    )


def _extract_links_from_html(html: str):
    """Extracts links from HTML content.

    Args:
        html (str): The HTML content

    Returns:
        list[str]: A list of links to download.
    """
    soup = BeautifulSoup(html, 'html.parser')
    links = [a['href'] for a in soup.find_all('a')]
    return links


def _recursive_download(
    session: requests.Session,
    base_url: str,
    path: str,
    save_dir: str,
    ignore_cert: bool = False,
):
    """Downloads all files/subdirectories from a directory on a remote server.

    Args:
        session: A requests.Session through which to make requests to the remote server.
        url (str): The base URL where the files are located.
        path (str): The path from the base URL to the files to download. The full URL for the download is equal to
            '<base_url>/<path>'.
        save_dir (str): The directory to save downloaded files to.
        ignore_cert (bool): Whether or not to ignore the validity of the SSL certificate of the remote server.
            Defaults to False.
            WARNING: Setting this to true is *not* secure, as no certificate verification will be performed.

    Raises:
        PermissionError: If the remote server returns a 401 Unauthorized status code.
        ValueError: If the remote server returns a 404 Not Found status code.
        RuntimeError: If the remote server returns a status code other than 200 OK or 401 Unauthorized.
    """
    url = urljoin(base_url, path)
    response = session.get(url, verify=(not ignore_cert))

    if response.status_code == HTTPStatus.UNAUTHORIZED:
        raise PermissionError(
            f'Not authorized to download file from {url}. Received status code {response.status_code}. '
        )
    elif response.status_code == HTTPStatus.NOT_FOUND:
        raise ValueError(
            f'Could not find file at {url}. Received status code {response.status_code}'
        )
    elif response.status_code != HTTPStatus.OK:
        raise RuntimeError(
            f'Could not download file from {url}. Received unexpected status code {response.status_code}'
        )

    # Assume that the URL points to a file if it does not end with a slash.
    if not path.endswith('/'):
        save_path = os.path.join(save_dir, path)
        parent_dir = os.path.dirname(save_path)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)

        with open(save_path, 'wb') as f:
            f.write(response.content)

            log.info(f'Downloaded file {save_path}')
            return

    # If the URL is a directory, the response should be an HTML directory listing that we can parse for additional links
    # to download.
    child_links = _extract_links_from_html(response.content.decode())
    for child_link in child_links:
        _recursive_download(session,
                            base_url,
                            urljoin(path, child_link),
                            save_dir,
                            ignore_cert=ignore_cert)


@tenacity.retry(retry=tenacity.retry_if_not_exception_type(
    (PermissionError, ValueError)),
                stop=tenacity.stop_after_attempt(3),
                wait=tenacity.wait_exponential(min=1, max=10))
def download_from_cache_server(
    model_name: str,
    cache_base_url: str,
    save_dir: str,
    token: Optional[str] = None,
    ignore_cert: bool = False,
):
    """Downloads Hugging Face models from a mirror file server.

    The file server is expected to store the files in the same structure as the Hugging Face cache
    structure. See https://huggingface.co/docs/huggingface_hub/guides/manage-cache.

    Args:
        model_name: The name of the model to download. This should be the same as the repository ID in the Hugging Face
            Hub.
        cache_base_url: The base URL of the cache file server. This function will attempt to download all of the blob
            files from `<cache_base_url>/<formatted_model_name>/blobs/`, where `formatted_model_name` is equal to
            `models/<model_name>` with all slashes replaced with `--`.
        save_dir: The directory to save the downloaded files to.
        token: The Hugging Face API token. If not provided, the token will be read from the `HUGGING_FACE_HUB_TOKEN`
            environment variable.
        ignore_cert: Whether or not to ignore the validity of the SSL certificate of the remote server. Defaults to
            False.
            WARNING: Setting this to true is *not* secure, as no certificate verification will be performed.
    """
    formatted_model_name = f'models/{model_name}'.replace('/', '--')
    with requests.Session() as session:
        session.headers.update({'Authorization': f'Bearer {token}'})

        download_start = time.time()

        # Temporarily suppress noisy SSL certificate verification warnings if ignore_cert is set to True
        with warnings.catch_warnings():
            if ignore_cert:
                warnings.simplefilter('ignore', category=InsecureRequestWarning)

            # Only downloads the blobs in order to avoid downloading model files twice due to the
            # symlnks in the Hugging Face cache structure:
            _recursive_download(
                session,
                cache_base_url,
                # Trailing slash to indicate directory
                f'{formatted_model_name}/blobs/',
                save_dir,
                ignore_cert=ignore_cert,
            )
        download_duration = time.time() - download_start
        log.info(
            f'Downloaded model {model_name} from cache server in {download_duration} seconds'
        )
