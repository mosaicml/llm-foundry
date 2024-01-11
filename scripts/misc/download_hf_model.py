# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Script to download model weights from Hugging Face Hub or a cache server."""
import argparse
import logging
import os
import shutil
import subprocess

import yaml
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE

from llmfoundry.utils.model_download_utils import (download_from_cache_server,
                                                   download_from_hf_hub)

HF_TOKEN_ENV_VAR = 'HUGGING_FACE_HUB_TOKEN'

logging.basicConfig(format=f'%(asctime)s: %(levelname)s: %(name)s: %(message)s',
                    level=logging.INFO)
log = logging.getLogger(__name__)
ORAS_PASSWD_PLACEHOLDER = '<placeholder_for_passwd>'
ORAS_CLI = 'oras'


def download_from_oras(model: str, save_dir: str, credentials_dirpath: str,
                       config_file: str, concurrency: int):
    """Download from an OCI-compliant registry using oras."""
    if shutil.which(ORAS_CLI) is None:
        raise Exception(
            f'oras cli command `{ORAS_CLI}` is not found. Please install oras: https://oras.land/docs/installation '
        )

    def validate_and_add_from_secret_file(
        secrets: dict[str, str],
        secret_name: str,
        secret_file_path: str,
    ):
        try:
            with open(secret_file_path, encoding='utf-8') as f:
                secrets[secret_name] = f.read()
        except Exception as error:
            raise ValueError(
                f'secret_file {secret_file_path} is failed to be read; but got error'
            ) from error

    secrets = {}
    validate_and_add_from_secret_file(
        secrets, 'username', os.path.join(credentials_dirpath, 'username'))
    validate_and_add_from_secret_file(
        secrets, 'password', os.path.join(credentials_dirpath, 'password'))
    validate_and_add_from_secret_file(
        secrets, 'registry', os.path.join(credentials_dirpath, 'registry'))

    with open(config_file, 'r', encoding='utf-8') as f:
        configs = yaml.safe_load(f.read())

    path = configs[model]
    hostname = secrets['registry']

    def get_oras_cmd_to_run(password: str):
        return [
            ORAS_CLI, 'pull', '-o', save_dir, '--verbose', '--concurrency',
            str(concurrency), '-u', secrets['username'], '-p', password,
            f'{hostname}/{path}'
        ]

    cmd_to_run = get_oras_cmd_to_run(ORAS_PASSWD_PLACEHOLDER)
    log.info(f'CMD for oras cli to run: {cmd_to_run}')
    cmd_to_run = get_oras_cmd_to_run(secrets['password'])
    subprocess.run(cmd_to_run, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Add shared args
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--download-from',
                        type=str,
                        choices=['hf', 'cache', 'oras'],
                        default='hf')
    parser.add_argument('--save-dir', type=str, default=HUGGINGFACE_HUB_CACHE)

    # Add HF args
    parser.add_argument('--token',
                        type=str,
                        default=os.getenv(HF_TOKEN_ENV_VAR))

    # Add cache args
    parser.add_argument('--cache-url', type=str, default=None)
    parser.add_argument('--ignore-cert', action='store_true', default=False)
    parser.add_argument(
        '--fallback',
        type=str,
        choices=['hf', 'oras', None],
        default=None,
        help='Fallback target to download from if download from cache fails',
    )

    # Add oras args
    parser.add_argument('--credentials-dirpath', type=str, default=None)
    parser.add_argument('--oras-config-file', type=str, default=None)
    parser.add_argument('--concurrency', type=int, default=10)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.download_from != 'cache' and args.fallback is not None:
        raise ValueError(
            f'Downloading from {args.download_from}, but fallback cannot be specified unless downloading from cache.'
        )

    if args.download_from == 'hf':
        download_from_hf_hub(args.model,
                             save_dir=args.save_dir,
                             token=args.token)
    elif args.download_from == 'oras':
        download_from_oras(args.model, args.save_dir, args.credentials_dirpath,
                           args.oras_config_file, args.concurrency)
    else:
        try:
            download_from_cache_server(
                args.model,
                args.cache_url,
                args.save_dir,
                token=args.token,
                ignore_cert=args.ignore_cert,
            )

            if args.fallback == 'hf':
                # A little hacky: run the Hugging Face download just to repair the symlinks in the HF cache file structure.
                # This shouldn't actually download any files if the cache server download was successful, but should address
                # a non-deterministic bug where the symlinks aren't repaired properly by the time the model is initialized.
                log.info('Repairing Hugging Face cache symlinks')

                # Hide some noisy logs that aren't important for just the symlink repair.
                old_level = logging.getLogger().level
                logging.getLogger().setLevel(logging.ERROR)
                download_from_hf_hub(args.model,
                                     save_dir=args.save_dir,
                                     token=args.token)
                logging.getLogger().setLevel(old_level)

        except PermissionError:
            log.error(f'Not authorized to download {args.model}.')
        except Exception as e:
            log.warning(
                f'Failed to download {args.model} from cache server. Falling back to {args.fallback}. Error: {e}'
            )
            if args.fallback == 'oras':
                # save_dir, token, model, url, username, password, and config file
                download_from_oras(args.model, args.save_dir,
                                   args.credentials_dirpath,
                                   args.oras_config_file, args.concurrency)
            elif args.fallback == 'hf':
                # save_dir, token, model
                download_from_hf_hub(args.model,
                                     save_dir=args.save_dir,
                                     token=args.token)
            else:
                raise e
