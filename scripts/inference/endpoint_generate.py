# Copyright 2022-2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Batch generate text completion results from an endpoint.

Warning: This script is experimental and could change or be removed at any time
"""

import asyncio
import copy
import logging
import math
import os
import tempfile
import time
from argparse import ArgumentParser, Namespace

import pandas as pd
import requests
from composer.utils import (get_file, maybe_create_object_store_from_uri,
                            parse_uri)

from llmfoundry.utils import prompt_files as utils

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
        '-i',
        '--inputs',
        nargs='+',
        help=f'List of strings, local datafiles (starting with {utils.PROMPTFILE_PREFIX}),' +\
             ' and/or remote object stores'
        )
    parser.add_argument(
        '--prompt-delimiter',
        default='\n',
        help=
        'Prompt delimiter for txt files. By default, a file is a single prompt')

    parser.add_argument('-o',
                        '--output-folder',
                        required=True,
                        help='Remote folder to save the output')

    #####
    # Generation Parameters
    parser.add_argument(
        '--rate-limit',
        type=int,
        default=75,
        help='Max number of calls to make to the endpoint in a second')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
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
            f'Batch size is {args.batch_size} but rate limit is set to {args.rate_limit} / s'
        )

    url = args.endpoint if args.endpoint else os.environ.get(ENDPOINT_URL_ENV)
    if not url:
        raise ValueError(
            f'URL must be provided via --endpoint or {ENDPOINT_URL_ENV}')

    log.info(f'Using endpoint {url}')

    api_key = os.environ.get(ENDPOINT_API_KEY_ENV, '')
    if not api_key:
        log.warning(f'API key not set in {ENDPOINT_API_KEY_ENV}')

    new_inputs = []
    for prompt in args.inputs:
        if prompt.startswith(utils.PROMPTFILE_PREFIX):
            new_inputs.append(prompt)
            continue

        input_object_store = maybe_create_object_store_from_uri(prompt)
        if input_object_store is not None:
            local_output_path = tempfile.TemporaryDirectory().name
            get_file(prompt, str(local_output_path))
            log.info(f'Downloaded {prompt} to {local_output_path}')
            prompt = f'{utils.PROMPTFILE_PREFIX}{local_output_path}'

        new_inputs.append(prompt)

    prompt_strings = utils.load_prompts(new_inputs, args.prompt_delimiter)

    cols = ['batch', 'prompt', 'output']
    param_data = {
        'max_tokens': args.max_tokens,
        'temperature': args.temperature,
        'top_k': args.top_k,
        'top_p': args.top_p,
    }

    total_batches = math.ceil(len(prompt_strings) / args.batch_size)
    log.info(
        f'Generating {len(prompt_strings)} prompts in {total_batches} batches')

    @sleep_and_retry
    @limits(calls=total_batches, period=1)  # type: ignore
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
                        f'Bad response: {resp.status} {resp.reason}')
            else:
                raise Exception(f'Bad response: {resp.status} {resp.reason}')

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

    gen_start = time.time()
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

    gen_end = time.time()
    gen_latency = (gen_end - gen_start)
    log.info(f'Generated {len(res)} prompts in {gen_latency}s, example data:')
    log.info(res.head())

    with tempfile.TemporaryDirectory() as tmp_dir:
        file = 'output.csv'
        local_path = os.path.join(tmp_dir, file)
        res.to_csv(local_path, index=False)

        output_object_store = maybe_create_object_store_from_uri(
            args.output_folder)
        if output_object_store is not None:
            _, _, output_folder_prefix = parse_uri(args.output_folder)
            remote_path = os.path.join(output_folder_prefix, file)
            output_object_store.upload_object(remote_path, local_path)
            output_object_store.download_object
            log.info(f'Uploaded results to {args.output_folder}/{file}')
        else:
            output_dir, _ = os.path.split(args.output_folder)
            os.makedirs(output_dir, exist_ok=True)
            os.rename(local_path, args.output_folder)
            log.info(f'Saved results to {args.output_folder}')


if __name__ == '__main__':
    asyncio.run(main(parse_args()))
