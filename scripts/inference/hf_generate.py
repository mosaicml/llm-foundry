# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import itertools
import os
import random
import time
import warnings
from argparse import ArgumentParser, ArgumentTypeError, Namespace
from contextlib import nullcontext

import torch
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          pipeline)


def get_dtype(dtype):
    if dtype == 'fp32':
        return torch.float32
    elif dtype == 'fp16':
        return torch.float16
    elif dtype == 'bf16':
        return torch.bfloat16
    else:
        raise NotImplementedError(
            f'dtype {dtype} is not supported. ' +\
            f'We only support fp32, fp16, and bf16 currently')


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def str_or_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        return v


def parse_args() -> Namespace:
    """Parse commandline arguments."""
    parser = ArgumentParser(
        description='Load a HF CausalLM Model and use it to generate text.')
    parser.add_argument('-n', '--name_or_path', type=str, required=True)
    parser.add_argument(
        '-p',
        '--prompts',
        nargs='+',
        default=[
            'My name is',
            'This is an explanation of deep learning to a five year old. Deep learning is',
        ],
        help='Generation prompts. Use syntax "file::/path/to/prompt.txt" to load a ' +\
             'prompt contained in a txt file.'
        )
    parser.add_argument('--max_seq_len', type=int, default=None)
    parser.add_argument('--max_new_tokens', type=int, default=100)
    parser.add_argument('--max_batch_size', type=int, default=None)
    #####
    # Note: Generation config defaults are set to match Hugging Face defaults
    parser.add_argument('--temperature', type=float, nargs='+', default=[1.0])
    parser.add_argument('--top_k', type=int, nargs='+', default=[50])
    parser.add_argument('--top_p', type=float, nargs='+', default=[1.0])
    parser.add_argument('--repetition_penalty',
                        type=float,
                        nargs='+',
                        default=[1.0])
    parser.add_argument('--no_repeat_ngram_size',
                        type=int,
                        nargs='+',
                        default=[0])
    #####
    parser.add_argument('--seed', type=int, nargs='+', default=[42])
    parser.add_argument('--do_sample',
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=True)
    parser.add_argument('--use_cache',
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=True)
    parser.add_argument('--eos_token_id', type=int, default=None)
    parser.add_argument('--pad_token_id', type=int, default=None)
    parser.add_argument('--model_dtype',
                        type=str,
                        choices=['fp32', 'fp16', 'bf16'],
                        default=None)
    parser.add_argument('--autocast_dtype',
                        type=str,
                        choices=['fp32', 'fp16', 'bf16'],
                        default=None)
    parser.add_argument('--warmup',
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=True)
    parser.add_argument('--trust_remote_code',
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=True)
    parser.add_argument('--use_auth_token',
                        type=str_or_bool,
                        nargs='?',
                        const=True,
                        default=None)
    parser.add_argument('--revision', type=str, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--device_map', type=str, default=None)
    parser.add_argument('--attn_impl', type=str, default=None)
    return parser.parse_args()


def load_prompt_string_from_file(prompt_path_str: str):
    if not prompt_path_str.startswith('file::'):
        raise ValueError('prompt_path_str must start with "file::".')
    _, prompt_file_path = prompt_path_str.split('file::', maxsplit=1)
    prompt_file_path = os.path.expanduser(prompt_file_path)
    if not os.path.isfile(prompt_file_path):
        raise FileNotFoundError(
            f'{prompt_file_path=} does not match any existing files.')
    with open(prompt_file_path, 'r') as f:
        prompt_string = ''.join(f.readlines())
    return prompt_string


def maybe_synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main(args: Namespace) -> None:
    # Set device or device_map
    if args.device and args.device_map:
        raise ValueError('You can only set one of `device` and `device_map`.')
    if args.device is not None:
        device = args.device
        device_map = None
    else:
        device = None
        device_map = args.device_map or 'auto'
    print(f'Using {device=} and {device_map=}')

    # Set model_dtype
    if args.model_dtype is not None:
        model_dtype = get_dtype(args.model_dtype)
    else:
        model_dtype = torch.float32
    print(f'Using {model_dtype=}')

    # Load prompts
    prompt_strings = []
    for prompt in args.prompts:
        if prompt.startswith('file::'):
            prompt = load_prompt_string_from_file(prompt)
        prompt_strings.append(prompt)

    # Grab config first
    print(f'Loading HF Config...')
    from_pretrained_kwargs = {
        'use_auth_token': args.use_auth_token,
        'trust_remote_code': args.trust_remote_code,
        'revision': args.revision,
    }
    try:
        config = AutoConfig.from_pretrained(args.name_or_path,
                                            **from_pretrained_kwargs)
        if hasattr(config, 'init_device') and device is not None:
            config.init_device = device
        if args.attn_impl is not None and hasattr(config, 'attn_config'):
            config.attn_config['attn_impl'] = args.attn_impl
        if args.max_seq_len is not None and hasattr(config, 'max_seq_len'):
            config.max_seq_len = args.max_seq_len

    except Exception as e:
        raise RuntimeError(
            'If you are having auth problems, try logging in via `huggingface-cli login` ' +\
            'or by setting the environment variable `export HUGGING_FACE_HUB_TOKEN=... ' +\
            'using your access token from https://huggingface.co/settings/tokens.'
        ) from e

    # Build tokenizer
    print('\nLoading HF tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(args.name_or_path,
                                              **from_pretrained_kwargs)
    if tokenizer.pad_token_id is None:
        warnings.warn(
            'pad_token_id is not set for the tokenizer. Using eos_token_id as pad_token_id.'
        )
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    # Load HF Model
    print(f'Loading HF model with dtype={model_dtype}...')
    try:
        model = AutoModelForCausalLM.from_pretrained(args.name_or_path,
                                                     config=config,
                                                     torch_dtype=model_dtype,
                                                     device_map=device_map,
                                                     **from_pretrained_kwargs)
        model.eval()
        print(f'n_params={sum(p.numel() for p in model.parameters())}')
        if device is not None:
            print(f'Placing model on {device=}...')
            model.to(device)
    except Exception as e:
        raise RuntimeError(
            'Unable to load HF model. '
            'If you are having auth problems, try logging in via `huggingface-cli login` ' +\
            'or by setting the environment variable `export HUGGING_FACE_HUB_TOKEN=... ' +\
            'using your access token from https://huggingface.co/settings/tokens.'
        ) from e

    # Autocast
    if args.autocast_dtype is not None:
        autocast_dtype = get_dtype(args.autocast_dtype)
        autocast_context = torch.autocast(model.device.type, autocast_dtype)
        print(f'Using autocast with dtype={autocast_dtype}...')
    else:
        autocast_context = nullcontext()
        print('NOT using autocast...')

    done_warmup = False

    for temp, topp, topk, repp, nrnz, seed in itertools.product(
            args.temperature, args.top_p, args.top_k, args.repetition_penalty,
            args.no_repeat_ngram_size, args.seed):

        # Seed randomness
        random.seed(seed)
        torch.manual_seed(seed)
        print(f'\nGenerate seed:\n{seed}')

        generate_kwargs = {
            'max_new_tokens': args.max_new_tokens,
            'temperature': temp,
            'top_p': topp,
            'top_k': topk,
            'repetition_penalty': repp,
            'no_repeat_ngram_size': nrnz,
            'use_cache': args.use_cache,
            'do_sample': False if temp == 0 else args.do_sample,
            'eos_token_id': args.eos_token_id or tokenizer.eos_token_id,
            'pad_token_id': args.pad_token_id or tokenizer.pad_token_id,
        }
        print(f'\nGenerate kwargs:\n{generate_kwargs}')

        # Generate function with correct context managers
        def _generate(encoded_inp):
            with torch.no_grad():
                with autocast_context:
                    return model.generate(
                        input_ids=encoded_inp['input_ids'],
                        attention_mask=encoded_inp['attention_mask'],
                        **generate_kwargs,
                    )

        # Split into prompt batches
        batches = []
        if args.max_batch_size:
            bs = args.max_batch_size
            batches = [
                prompt_strings[i:i + bs]
                for i in range(0, len(prompt_strings), bs)
            ]

        else:
            batches = [prompt_strings]

        for batch in batches:
            print(f'\nTokenizing prompts...')
            maybe_synchronize()
            encode_start = time.time()
            encoded_inp = tokenizer(batch, return_tensors='pt', padding=True)
            for key, value in encoded_inp.items():
                encoded_inp[key] = value.to(model.device)
            maybe_synchronize()
            encode_end = time.time()
            input_tokens = torch.sum(
                encoded_inp['input_ids'] != tokenizer.pad_token_id,
                axis=1).numpy(force=True)  # type: ignore

            # Warmup
            if args.warmup and (not done_warmup):
                print('Warming up...')
                _ = _generate(encoded_inp)
                done_warmup = True

            # Run HF generate
            print('Generating responses...')
            maybe_synchronize()
            gen_start = time.time()
            encoded_gen = _generate(encoded_inp)
            maybe_synchronize()
            gen_end = time.time()

            decode_start = time.time()
            decoded_gen = tokenizer.batch_decode(encoded_gen,
                                                 skip_special_tokens=True)
            maybe_synchronize()
            decode_end = time.time()
            gen_tokens = torch.sum(encoded_gen != tokenizer.pad_token_id,
                                   axis=1).numpy(force=True)  # type: ignore

            # Print generations
            delimiter = '#' * 100
            for prompt, gen in zip(batch, decoded_gen):
                continuation = gen[len(prompt):]
                print(delimiter)
                print('\033[92m' + prompt + '\033[0m' + continuation)
            print(delimiter)

            # Print timing info
            bs = len(batch)
            output_tokens = gen_tokens - input_tokens
            total_input_tokens = input_tokens.sum()
            total_output_tokens = output_tokens.sum()
            encode_latency = 1000 * (encode_end - encode_start)
            gen_latency = 1000 * (gen_end - gen_start)
            decode_latency = 1000 * (decode_end - decode_start)
            total_latency = encode_latency + gen_latency + decode_latency

            latency_per_output_token = total_latency / total_output_tokens
            output_tok_per_sec = 1000 / latency_per_output_token
            print(f'{bs=}, {input_tokens=}, {output_tokens=}')
            print(f'{total_input_tokens=}, {total_output_tokens=}')
            print(
                f'{encode_latency=:.2f}ms, {gen_latency=:.2f}ms, {decode_latency=:.2f}ms, {total_latency=:.2f}ms'
            )
            print(f'{latency_per_output_token=:.2f}ms/tok')
            print(f'{output_tok_per_sec=:.2f}tok/sec')


if __name__ == '__main__':
    main(parse_args())
