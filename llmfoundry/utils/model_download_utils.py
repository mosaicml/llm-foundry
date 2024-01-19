# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for downloading models."""
import copy
import logging
import os
import shutil
import subprocess
import time
import warnings
from http import HTTPStatus
from typing import Optional
from urllib.parse import urljoin

import huggingface_hub as hf_hub
import requests
import tenacity
import yaml
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

ORAS_PASSWD_PLACEHOLDER = '<placeholder_for_passwd>'
ORAS_CLI = 'oras'

log = logging.getLogger(__name__)


@tenacity.retry(retry=tenacity.retry_if_not_exception_type(
    (ValueError, hf_hub.utils.RepositoryNotFoundError)),
                stop=tenacity.stop_after_attempt(3),
                wait=tenacity.wait_exponential(min=1, max=10))
def download_from_hf_hub(
    model: str,
    save_dir: str,
    prefer_safetensors: bool = True,
    token: Optional[str] = None,
):
    """Downloads model files from a Hugging Face Hub model repo.

    Only supports models stored in Safetensors and PyTorch formats for now. If both formats are available, only the
    Safetensors weights will be downloaded unless `prefer_safetensors` is set to False.

    Args:
        repo_id (str): The Hugging Face Hub repo ID.
        save_dir (str, optional): The local path to the directory where the model files will be downloaded.
        prefer_safetensors (bool): Whether to prefer Safetensors weights over PyTorch weights if both are
            available. Defaults to True.
        token (str, optional): The HuggingFace API token. If not provided, the token will be read from the
            `HUGGING_FACE_HUB_TOKEN` environment variable.

    Raises:
        RepositoryNotFoundError: If the model repo doesn't exist or the token is unauthorized.
        ValueError: If the model repo doesn't contain any supported model weights.
    """
    repo_files = set(hf_hub.list_repo_files(model))

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
            f'No supported model weights found in repo {model}.' +
            ' Please make sure the repo contains either safetensors or pytorch weights.'
        )

    download_start = time.time()
    hf_hub.snapshot_download(model,
                             local_dir=save_dir,
                             ignore_patterns=ignore_patterns,
                             token=token)
    download_duration = time.time() - download_start
    log.info(
        f'Downloaded model {model} from Hugging Face Hub in {download_duration} seconds'
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
    print(url)
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
    if not url.endswith('/'):
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
    print(child_links)
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
def download_from_http_fileserver(
    url: str,
    save_dir: str,
    ignore_cert: bool = False,
):
    """Downloads files from a remote HTTP file server.

    Args:
        url (str): The base URL where the files are located.
        save_dir (str): The directory to save downloaded files to.
        ignore_cert (bool): Whether or not to ignore the validity of the SSL certificate of the remote server.
            Defaults to False.
            WARNING: Setting this to true is *not* secure, as no certificate verification will be performed.
    """
    with requests.Session() as session:
        # Temporarily suppress noisy SSL certificate verification warnings if ignore_cert is set to True
        with warnings.catch_warnings():
            if ignore_cert:
                warnings.simplefilter('ignore', category=InsecureRequestWarning)

            _recursive_download(session,
                                url,
                                '',
                                save_dir,
                                ignore_cert=ignore_cert)


def download_from_oras(model: str,
                       config_file: str,
                       credentials_dir: str,
                       save_dir: str,
                       concurrency: int = 10):
    """Download from an OCI-compliant registry using oras.

    Args:
        model: The name of the model to download.
        config_file: Path to a YAML config file that maps model names to registry paths.
        credentials_dir: Path to a directory containing credentials for the registry. It is expected to contain three
            files: `username`, `password`, and `registry`, each of which contains the corresponding credential.
        save_dir: Path to the directory where files will be downloaded.
        concurrency: The number of concurrent downloads to run.
    """
    if shutil.which(ORAS_CLI) is None:
        raise Exception(
            f'oras cli command `{ORAS_CLI}` is not found. Please install oras: https://oras.land/docs/installation '
        )

    def _read_secrets_file(secret_file_path: str,):
        try:
            with open(secret_file_path, encoding='utf-8') as f:
                return f.read().strip()
        except Exception as error:
            raise ValueError(
                f'secrets file {secret_file_path} failed to be read') from error

    secrets = {}
    for secret in ['username', 'password', 'registry']:
        secrets[secret] = _read_secrets_file(
            os.path.join(credentials_dir, secret))

    with open(config_file, 'r', encoding='utf-8') as f:
        configs = yaml.safe_load(f.read())

    path = configs['models'][model]
    registry = secrets['registry']

    def get_oras_cmd(username: Optional[str] = None,
                     password: Optional[str] = None):
        cmd = [
            ORAS_CLI,
            'pull',
            f'{registry}/{path}',
            '-o',
            save_dir,
            '--verbose',
            '--concurrency',
            str(concurrency),
        ]
        if username is not None:
            cmd.extend(['--username', username])
        if password is not None:
            cmd.extend(['--password', password])

        return cmd

    cmd_without_creds = get_oras_cmd()
    log.info(f'CMD for oras cli to run: {" ".join(cmd_without_creds)}')
    cmd_to_run = get_oras_cmd(username=secrets['username'],
                              password=secrets['password'])
    try:
        subprocess.run(cmd_to_run, check=True)
    except subprocess.CalledProcessError as e:
        # Intercept the error and replace the cmd, which may have sensitive info.
        raise subprocess.CalledProcessError(e.returncode, cmd_without_creds,
                                            e.output, e.stderr)
