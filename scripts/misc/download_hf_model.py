# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Script to download model weights from Hugging Face Hub or a cache server."""
import argparse
import logging
import os
import sys

from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE

from llmfoundry.utils.model_download_utils import (download_from_cache_server,
                                                   download_from_hf_hub)

HF_TOKEN_ENV_VAR = 'HUGGING_FACE_HUB_TOKEN'

logging.basicConfig(format=f'%(asctime)s: %(levelname)s: %(name)s: %(message)s',
                    level=logging.INFO)
log = logging.getLogger(__name__)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model', type=str, required=True)
    argparser.add_argument('--download-from',
                           type=str,
                           choices=['hf', 'cache'],
                           default='hf')
    argparser.add_argument('--token',
                           type=str,
                           default=os.getenv(HF_TOKEN_ENV_VAR))
    argparser.add_argument('--save-dir',
                           type=str,
                           default=HUGGINGFACE_HUB_CACHE)
    argparser.add_argument('--cache-url', type=str, default=None)
    argparser.add_argument('--ignore-cert', action='store_true', default=False)
    argparser.add_argument(
        '--fallback',
        action='store_true',
        default=True,
        help=
        'Whether to fallback to downloading from Hugging Face if download from cache fails',
    )

    args = argparser.parse_args(sys.argv[1:])
    if args.download_from == 'hf':
        download_from_hf_hub(args.model,
                             save_dir=args.save_dir,
                             token=args.token)
    else:
        try:
            download_from_cache_server(
                args.model,
                args.cache_url,
                args.save_dir,
                token=args.token,
                ignore_cert=args.ignore_cert,
            )

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
            if args.fallback:
                log.warning(
                    f'Failed to download {args.model} from cache server. Falling back to Hugging Face Hub. Error: {e}'
                )
                download_from_hf_hub(args.model,
                                     save_dir=args.save_dir,
                                     token=args.token)
            else:
                raise e
