# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Script to download model weights from Hugging Face Hub or a cache server.

Download from Hugging Face Hub:
    python download_model.py hf --model mosaicml/mpt-7b --save-dir <save_dir> --token <token>

Download from ORAS registry:
    python download_model.py oras --registry <registry> --path mosaicml/mpt-7b --save-dir <save_dir>

Download from an HTTP file server:
    python download_model.py http --host https://server.com --path mosaicml/mpt-7b --save-dir <save_dir>
"""
import argparse
import logging
import os

from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE

from llmfoundry.utils.model_download_utils import (
    download_from_hf_hub, download_from_http_fileserver, download_from_oras)

HF_TOKEN_ENV_VAR = 'HUGGING_FACE_HUB_TOKEN'

logging.basicConfig(format=f'%(asctime)s: %(levelname)s: %(name)s: %(message)s',
                    level=logging.INFO)
log = logging.getLogger(__name__)


def add_hf_subparser(subparsers: argparse._SubParsersAction) -> None:
    hf_parser = subparsers.add_parser('hf')
    hf_parser.add_argument('--model', type=str, required=True)
    hf_parser.add_argument('--save-dir',
                           type=str,
                           default=HUGGINGFACE_HUB_CACHE)
    hf_parser.add_argument('--prefer-safetensors', type=bool, default=True)
    hf_parser.add_argument('--token',
                           type=str,
                           default=os.getenv(HF_TOKEN_ENV_VAR))


def add_oras_subparser(subparsers: argparse._SubParsersAction) -> None:
    oras_parser = subparsers.add_parser('oras')
    oras_parser.add_argument('--registry', type=str, required=True)
    oras_parser.add_argument('--path', type=str, required=True)
    oras_parser.add_argument('--save-dir', type=str, required=True)
    oras_parser.add_argument('--username', type=str, default='')
    oras_parser.add_argument('--password', type=str, default='')
    oras_parser.add_argument('--concurrency', type=int, default=10)


def add_http_subparser(subparsers: argparse._SubParsersAction) -> None:
    http_parser = subparsers.add_parser('http')
    http_parser.add_argument('--host', type=str, required=True)
    http_parser.add_argument('--path', type=str, required=True)
    http_parser.add_argument('--save-dir', type=str, default=None)
    http_parser.add_argument('--ignore-cert',
                             action='store_true',
                             default=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='download_from', required=True)
    add_hf_subparser(subparsers)
    add_oras_subparser(subparsers)
    add_http_subparser(subparsers)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    download_from = args.download_from

    if download_from == 'http':
        try:
            download_from_http_fileserver(args.host, args.path, args.save_dir,
                                          args.ignore_cert)
            # if args.fallback == 'hf':
            #     # A little hacky: run the Hugging Face download just to repair the symlinks in the HF cache file structure.
            #     # This shouldn't actually download any files if the cache server download was successful, but should address
            #     # a non-deterministic bug where the symlinks aren't repaired properly by the time the model is initialized.
            #     log.info('Repairing Hugging Face cache symlinks')
            #     # Hide some noisy logs that aren't important for just the symlink repair.
            #     old_level = logging.getLogger().level
            #     logging.getLogger().setLevel(logging.ERROR)
            #     download_from_hf_hub(args.model,
            #                          save_dir=args.save_dir,
            #                          token=args.token)
            #     logging.getLogger().setLevel(old_level)

        except PermissionError as e:
            log.error(f'Not authorized to download {args.model}. Error: {e}')
        except Exception as e:
            raise e

            # if args.fallback:
            #     log.warning(
            #         f'Failed to download {args.model} from cache server. Falling back to {args.fallback}. Error: {e}'
            #     )

            #     # Update download_from so the fallback happens below in the normal code path.
            #     download_from = args.fallback
            # else:
            # raise e
    elif download_from == 'hf':
        download_from_hf_hub(args.model,
                             save_dir=args.save_dir,
                             token=args.token,
                             prefer_safetensors=args.prefer_safetensors)
    elif download_from == 'oras':
        download_from_oras(args.registry, args.path, args.save_dir,
                           args.username, args.password, args.concurrency)

    # if args.download_from != 'cache' and args.fallback is not None:
    #     raise ValueError(
    #         f'Downloading from {args.download_from}, but fallback cannot be specified unless downloading from cache.'
    #     )

    # if args.download_from == 'hf':
    #     download_from_hf_hub(args.model,
    #                          save_dir=args.save_dir,
    #                          token=args.token)
    # elif args.download_from == 'oras':
    #     download_from_oras(args.model, args.save_dir, args.credentials_dirpath,
    #                        args.oras_config_file, args.concurrency)
    # else:
    #     try:
    #         download_from_cache_server(
    #             args.model,
    #             args.cache_url,
    #             args.save_dir,
    #             token=args.token,
    #             ignore_cert=args.ignore_cert,
    #         )

    #         if args.fallback == 'hf':
    #             # A little hacky: run the Hugging Face download just to repair the symlinks in the HF cache file structure.
    #             # This shouldn't actually download any files if the cache server download was successful, but should address
    #             # a non-deterministic bug where the symlinks aren't repaired properly by the time the model is initialized.
    #             log.info('Repairing Hugging Face cache symlinks')

    #             # Hide some noisy logs that aren't important for just the symlink repair.
    #             old_level = logging.getLogger().level
    #             logging.getLogger().setLevel(logging.ERROR)
    #             download_from_hf_hub(args.model,
    #                                  save_dir=args.save_dir,
    #                                  token=args.token)
    #             logging.getLogger().setLevel(old_level)

    #     except PermissionError:
    #         log.error(f'Not authorized to download {args.model}.')
    #     except Exception as e:
    #         log.warning(
    #             f'Failed to download {args.model} from cache server. Falling back to {args.fallback}. Error: {e}'
    #         )
    #         if args.fallback == 'oras':
    #             # save_dir, token, model, url, username, password, and config file
    #             download_from_oras(args.model, args.save_dir,
    #                                args.credentials_dirpath,
    #                                args.oras_config_file, args.concurrency)
    #         elif args.fallback == 'hf':
    #             # save_dir, token, model
    #             download_from_hf_hub(args.model,
    #                                  save_dir=args.save_dir,
    #                                  token=args.token)
    #         else:
    #             raise e
