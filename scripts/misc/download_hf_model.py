"""Download functions for Hugging Face models.

TODO(jerry): Add better logging throughout
TODO(jerry): Track download metrics
"""
import os
import sys
import time

import argparse
from typing import Optional
from http import HTTPStatus
from bs4 import BeautifulSoup

import huggingface_hub as hf_hub
import requests

# Constants derived from https://github.com/huggingface/transformers/blob/main/src/transformers/utils/__init__.py
PYTORCH_WEIGHTS_NAME = "pytorch_model.bin"
PYTORCH_WEIGHTS_INDEX_NAME = "pytorch_model.bin.index.json"
PYTORCH_WEIGHTS_PATTERN = "pytorch_model*.bin*"

SAFETENSORS_WEIGHTS_NAME = "model.safetensors"
SAFETENSORS_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
SAFETENSORS_EXCLUDE_PATTERN = "model*.safetensors*"

DEFAULT_HF_HUB_CACHE_DIR = os.getenv(
    "HUGGINGFACE_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub")
)


def download_from_hf_hub(
    repo_id: str,
    save_dir: Optional[str] = None,
    prefer_safetensors: bool = True,
    token: Optional[str] = None,
):
    """Downloads the model weights and suppporting files/metadata from a Hugging Face Hub model repo.

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
    """
    repo_files = set(hf_hub.list_repo_files(repo_id))

    # Ignore TensorFlow, TensorFlow 2, and Flax weights as they are not supported by Composer.
    ignore_patterns = [
        "*.ckpt",
        "*.h5",
        "*.msgpack",
    ]

    if (
        SAFETENSORS_WEIGHTS_NAME in repo_files
        or SAFETENSORS_WEIGHTS_INDEX_NAME in repo_files
    ) and prefer_safetensors:
        print("Safetensors found and preferred. Excluding pytorch files")
        ignore_patterns.append(PYTORCH_WEIGHTS_PATTERN)
    elif PYTORCH_WEIGHTS_NAME in repo_files or PYTORCH_WEIGHTS_INDEX_NAME in repo_files:
        print(
            "Safetensors not found or prefer_safetensors is False. Excluding safetensors files"
        )
        ignore_patterns.append(SAFETENSORS_EXCLUDE_PATTERN)
    else:
        raise ValueError(
            f"No supported model weights found in repo {repo_id}."
            "Please make sure the repo contains either safetensors or pytorch weights."
        )

    download_start = time.time()
    hf_hub.snapshot_download(
        repo_id, cache_dir=save_dir, ignore_patterns=ignore_patterns, token=token
    )
    download_duration = time.time() - download_start
    print(
        f"Downloaded model {repo_id} from Hugging Face Hub in {download_duration} seconds"
    )


def _extract_links_from_html(html: str):
    """Extracts links from HTML content.

    Args:
        html (str): The HTML content

    Returns:
        list[str]: A list of links to download.
    """
    soup = BeautifulSoup(html, "html.parser")
    links = [a["href"] for a in soup.find_all("a")]
    return links


def _recursive_download(
    session: requests.Session,
    base_url: str,
    path: str,
    save_dir: str,
    ignore_cert: bool = False,
):
    """Downloads all files from a directory on a remote server, including subdirectories.

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
        RuntimeError: If the remote server returns a status code other than 200 OK or 401 Unauthorized.
    """
    url = f"{base_url}/{path}"
    response = session.get(url, verify=(not ignore_cert))

    if response.status_code == HTTPStatus.UNAUTHORIZED:
        raise PermissionError(
            f"Could not download file from {url}. Received status code {response.status_code}. "
            + "Please check that your token is valid and try again."
        )
    elif response.status_code != HTTPStatus.OK:
        raise RuntimeError(
            f"Could not download file from {url}. Received status code {response.status_code}"
        )

    # Assume that the URL points to a file if it does not end with a slash.
    if not path.endswith("/"):
        save_path = f"{save_dir}/{path}"
        parent_dir = os.path.dirname(save_path)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)

        with open(save_path, "wb") as f:
            f.write(response.content)

            print(f"Downloaded file {save_path}")
            return

    # If the URL is a directory, the response should be an HTML directory listing that we can parse for additional links
    # to download.
    child_links = _extract_links_from_html(response.content.decode())
    for child_link in child_links:
        _recursive_download(
            session, base_url, f"{path}/{child_link}", save_dir, ignore_cert=ignore_cert
        )


def download_from_cache_server(
    model_name: str,
    cache_base_url: str,
    save_dir: str,
    token: Optional[str] = None,
    ignore_cert: bool = False,
):
    """Downloads Hugging Face model files from a file server that mirrors the Hugging Face Hub cache structure.

    This function will attempt to download all of the blob files from `<cache_base_url>/<formatted_model_name>/blobs/`
    on the remote cache file server, where `formatted_model_name` is equal to `models/<model_name>` with all slashes
    replaced with `--`.

    It only downloads the blobs in order to avoid downloading model files twice due to the symlink structure of the
    Hugging Face cache.
    For details on the cache structure: https://huggingface.co/docs/huggingface_hub/guides/manage-cache

    Args:
        model_name: The name of the model to download. This should be the same as the repository ID in the Hugging Face
            Hub.
        cache_base_url: The base URL of the cache file server.
        save_dir: The directory to save the downloaded files to.
        token: The HuggingFace API token. If not provided, the token will be read from the `HUGGING_FACE_HUB_TOKEN`
            environment variable.
        ignore_cert: Whether or not to ignore the validity of the SSL certificate of the remote server. Defaults to
            False.
            WARNING: Setting this to true is *not* secure, as no certificate verification will be performed.
    """
    formatted_model_name = f"models/{model_name}".replace("/", "--")
    with requests.Session() as session:
        session.headers.update({"Authorization": f"Bearer {token}"})

        download_start = time.time()
        _recursive_download(
            session,
            cache_base_url,
            f"{formatted_model_name}/blobs/",  # Trailing slash to indicate directory
            save_dir,
            ignore_cert=ignore_cert,
        )
        download_duration = time.time() - download_start
        print(
            f"Downloaded model {model_name} from cache server in {download_duration} seconds"
        )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", type=str, required=True)
    argparser.add_argument(
        "--download-from", type=str, choices=["hf", "cache"], default="hf"
    )
    argparser.add_argument("--save-dir", type=str, default=DEFAULT_HF_HUB_CACHE_DIR)
    argparser.add_argument("--cache-url", type=str, default=None)
    argparser.add_argument("--token", type=str, default=None)
    argparser.add_argument("--ignore-cert", action="store_true", default=False)
    argparser.add_argument(
        "--fallback",
        action="store_true",
        default=False,
        help="Whether to fallback to downloading from Hugging Face if download from cache fails",
    )

    args = argparser.parse_args(sys.argv[1:])
    if args.download_from == "hf":
        download_from_hf_hub(args.model, save_dir=args.save_dir, token=args.token)
    else:
        try:
            download_from_cache_server(
                args.model,
                args.cache_url,
                args.save_dir,
                token=args.token,
                ignore_cert=args.ignore_cert,
            )
        except PermissionError:
            print(f"Not authorized to download {args.model}.")
        except Exception as e:
            if args.fallback:
                print(
                    f"Failed to download from cache server. Falling back to Hugging Face Hub. Error: {e}"
                )
                download_from_hf_hub(
                    args.model, save_dir=args.save_dir, token=args.token
                )
            else:
                raise e
