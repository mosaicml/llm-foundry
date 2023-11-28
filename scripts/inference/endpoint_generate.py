# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
import asyncio
import copy
import logging
import math
import os
import time
from argparse import ArgumentParser, Namespace
from typing import List, cast

import pandas as pd
import requests
from composer.utils import (ObjectStore, maybe_create_object_store_from_uri,
                            parse_uri)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

ENDPOINT_API_KEY_ENV: str = 'ENDPOINT_API_KEY'
ENDPOINT_URL_ENV: str = 'ENDPOINT_URL'

PROMPT_DELIMITER = '\n'


def parse_args() -> Namespace:
    """Parse commandline arguments."""
    parser = ArgumentParser(
        description='Call prompts against a text completions endpoint')

    #####
    # Path Parameters
    parser.add_argument(
        '-p',
        '--prompts',
        nargs='+',
        help='Generation prompts. Use syntax "file::/path/to/prompt.txt" to load a ' +\
             'prompt contained in a txt file.'
        )

    now = time.strftime('%Y%m%d-%H%M%S')
    default_local_folder = f'/tmp/output/{now}'
    parser.add_argument('-l',
                        '--local-folder',
                        type=str,
                        default=default_local_folder,
                        help='Local folder to save the output')

    parser.add_argument('-o',
                        '--output-folder',
                        help='Remote folder to save the output')

    #####
    # Generation Parameters
    parser.add_argument(
        '--rate-limit',
        type=int,
        default=5,
        help='Max number of calls to make to the endpoint in a second')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=5,
        help='Max number of calls to make to the endpoint in a single request')

    #####
    # Endpoint Parameters
    parser.add_argument(
        '-e',
        '--endpoint',
        type=str,
        help=
        f'OpenAI-compatible text completions endpoint to query on. If not set, will read from {ENDPOINT_URL_ENV}'
    )

    parser.add_argument('--max-tokens', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--top-p', type=float, default=1.0)
    return parser.parse_args()


def load_prompts_from_file(prompt_path_str: str) -> List[str]:
    if not prompt_path_str.startswith('file::'):
        raise ValueError('prompt_path_str must start with "file::".')
    _, prompt_file_path = prompt_path_str.split('file::', maxsplit=1)
    prompt_file_path = os.path.expanduser(prompt_file_path)
    if not os.path.isfile(prompt_file_path):
        raise FileNotFoundError(
            f'{prompt_file_path=} does not match any existing files.')
    with open(prompt_file_path, 'r') as f:
        prompt_string = f.read()
    return prompt_string.split(PROMPT_DELIMITER)


async def main(args: Namespace) -> None:
    # This is mildly experimental, so for now imports are not added as part of llm-foundry
    try:
        import aiohttp
    except ImportError as e:
        raise ImportError('Please install aiohttp') from e

    try:
        from ratelimit import limits, sleep_and_retry
    except ImportError as e:
        raise ImportError('Please install ratelimit') from e

    if args.batch_size > args.rate_limit:
        raise ValueError(
            f'Batch size is {args.batch_size} but rate limit is set to { args.rate_limit} / s'
        )

    url = args.endpoint if args.endpoint else os.environ.get(ENDPOINT_URL_ENV)
    if not url:
        raise ValueError(
            f'URL must be provided via --endpoint or {ENDPOINT_URL_ENV}')

    log.info(f'Using endpoint {url}')

    api_key = os.environ.get(ENDPOINT_API_KEY_ENV, '')
    if not api_key:
        log.warning(f'API key not set in {ENDPOINT_API_KEY_ENV}')

    # Load prompts
    prompt_strings = []
    for prompt in args.prompts:
        if prompt.startswith('file::'):
            prompt = load_prompts_from_file(prompt)
        prompt_strings.append(prompt)

    cols = ['batch', 'prompt', 'output']
    param_data = {
        'max_tokens': args.max_tokens,
        'temperature': args.temperature,
        'top_k': args.top_k,
        'top_p': args.top_p,
    }

    @sleep_and_retry
    @limits(calls=args.rate_limit // args.batch_size, period=1)  # type: ignore
    async def generate(session: aiohttp.ClientSession, batch: int,
                       prompts: list):
        data = copy.copy(param_data)
        data['prompt'] = prompts
        headers = {'Authorization': api_key, 'Content-Type': 'application/json'}

        req_start = time.time()
        async with session.post(url, headers=headers, json=data) as resp:
            if resp.ok:
                try:
                    response = await resp.json()
                except requests.JSONDecodeError:
                    raise Exception(
                        f'Bad response: {resp.status_code} {resp.reason}'  # type: ignore
                    )
            else:
                raise Exception(
                    f'Bad response: {resp.status_code} {resp.content.decode().strip()}'  # type: ignore
                )

        req_end = time.time()
        n_compl = response['usage']['completion_tokens']
        n_prompt = response['usage']['prompt_tokens']
        req_latency = (req_end - req_start)
        log.info(f'Completed batch {batch}: {n_compl:,} completion' +
                 f' tokens using {n_prompt:,} prompt tokens in {req_latency}s')

        res = pd.DataFrame(columns=cols)

        for r in response['choices']:
            index = r['index']
            res.loc[len(res)] = [batch, prompts[index], r['text']]
        return res

    res = pd.DataFrame(columns=cols)
    batch = 0

    total_batches = math.ceil(len(prompt_strings) / args.batch_size)
    log.info(
        f'Generating {len(prompt_strings)} prompts in {total_batches} batches')

    async with aiohttp.ClientSession() as session:
        tasks = []

        for i in range(total_batches):
            prompts = prompt_strings[i * args.batch_size:min(
                (i + 1) * args.batch_size, len(prompt_strings))]

            tasks.append(generate(session, batch, prompts))
            batch += 1

        results = await asyncio.gather(*tasks)
        res = pd.concat(results)

    res.reset_index(drop=True, inplace=True)
    log.info(f'Generated {len(res)} prompts, example data:')
    log.info(res.head())

    # save res to local output folder
    os.makedirs(args.local_folder, exist_ok=True)
    local_path = os.path.join(args.local_folder, 'output.csv')
    res.to_csv(os.path.join(args.local_folder, 'output.csv'), index=False)
    log.info(f'Saved results in {local_path}')

    if args.output_folder:
        # Upload the local output to the remote location
        output_object_store = cast(
            ObjectStore, maybe_create_object_store_from_uri(args.output_folder))
        _, _, output_folder_prefix = parse_uri(args.output_folder)
        files_to_upload = os.listdir(args.local_folder)

        for file in files_to_upload:
            assert not os.path.isdir(file)
            local_path = os.path.join(args.local_folder, file)
            remote_path = os.path.join(output_folder_prefix, file)
            output_object_store.upload_object(remote_path, local_path)
            log.info(f'Uploaded {local_path} to {args.output_folder}/{file}')


if __name__ == '__main__':
    asyncio.run(main(parse_args()))
