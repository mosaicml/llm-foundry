# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Script to download model weights from Hugging Face Hub or a cache server.

Download from Hugging Face Hub:
    python download_model.py hf --model mosaicml/mpt-7b --save-dir <save_dir> --token <token>

Download from ORAS registry:
    python download_model.py oras --model mosaicml/mpt-7b --config-file <config_file> \
        --credentials-dir <credentials_dir> --save-dir <save_dir>

Download from an HTTP file server:
    python download_model.py http --url https://server.com/models/mosaicml/mpt-7b/ --save-dir <save_dir>

Download from an HTTP file server with fallback to Hugging Face Hub:
    python download_model.py http --host https://server.com --path mosaicml/mpt-7b --save-dir <save_dir> \
        fallback-hf --model mosaicml/mpt-7b --token hf_token
"""
import argparse
import logging
import os

from llmfoundry.utils.model_download_utils import (
    download_from_hf_hub, download_from_http_fileserver, download_from_oras)

HF_TOKEN_ENV_VAR = 'HUGGING_FACE_HUB_TOKEN'

logging.basicConfig(format=f'%(asctime)s: %(levelname)s: %(name)s: %(message)s',
                    level=logging.INFO)
log = logging.getLogger(__name__)


def add_hf_parser_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--prefer-safetensors', type=bool, default=True)
    parser.add_argument('--token',
                        type=str,
                        default=os.getenv(HF_TOKEN_ENV_VAR))


def add_oras_parser_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--config-file', type=str, required=True)
    parser.add_argument('--credentials-dir', type=str, required=True)
    parser.add_argument('--concurrency', type=int, default=10)


def add_http_parser_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--url', type=str, required=True)
    parser.add_argument('--ignore-cert', action='store_true', default=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='download_from', required=True)

    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument('--save-dir', type=str, required=True)
    base_parser.add_argument('--tokenizer-only',
                             default=False,
                             action='store_true')

    # Add subparser for downloading from Hugging Face Hub.
    hf_parser = subparsers.add_parser('hf', parents=[base_parser])
    add_hf_parser_arguments(hf_parser)

    # Add subparser for downloading from ORAS registry.
    oras_parser = subparsers.add_parser('oras', parents=[base_parser])
    add_oras_parser_arguments(oras_parser)

    # Add subparser for downloading from an HTTP file server.
    http_parser = subparsers.add_parser('http', parents=[base_parser])
    add_http_parser_arguments(http_parser)

    # Add fallbacks for HTTP
    fallback_subparsers = http_parser.add_subparsers(dest='fallback')
    hf_fallback_parser = fallback_subparsers.add_parser('fallback-hf')
    add_hf_parser_arguments(hf_fallback_parser)

    oras_fallback_parser = fallback_subparsers.add_parser('fallback-oras')
    add_oras_parser_arguments(oras_fallback_parser)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    download_from = args.download_from

    if download_from == 'http':
        if args.tokenizer_only:
            log.warning(
                'tokenizer-only is not currently supported for http. Downloading all files instead.'
            )
        try:
            download_from_http_fileserver(args.url, args.save_dir,
                                          args.ignore_cert)
        except PermissionError as e:
            log.error(f'Not authorized to download {args.model}.')
            raise e
        except Exception as e:
            log.warning(f'Failed to download from HTTP server with error: {e}')
            if args.fallback:
                log.warning(f'Falling back to provided fallback destination.')
                if args.fallback == 'fallback-hf':
                    download_from = 'hf'
                elif args.fallback == 'fallback-oras':
                    download_from = 'oras'
                else:
                    raise ValueError(
                        f'Invalid fallback destination {args.fallback}.')
            else:
                raise e

    if download_from == 'hf':
        download_from_hf_hub(args.model,
                             save_dir=args.save_dir,
                             token=args.token,
                             tokenizer_only=args.tokenizer_only,
                             prefer_safetensors=args.prefer_safetensors)
    elif download_from == 'oras':
        download_from_oras(args.model,
                           args.config_file,
                           args.credentials_dir,
                           args.save_dir,
                           tokenizer_only=args.tokenizer_only,
                           concurrency=args.concurrency)
