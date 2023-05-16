# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import time
import warnings
from argparse import ArgumentParser, ArgumentTypeError, Namespace
from contextlib import nullcontext
from typing import Any, Dict, Tuple, Union

import torch
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedTokenizer, PreTrainedTokenizerFast)

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


def get_dtype(dtype):
    if dtype == 'fp32':
        return torch.float32
    elif dtype == 'fp16':
        return torch.float16
    elif dtype == 'bf16':
        return torch.bfloat16
    else:
        raise NotImplementedError(
            f'dtype {dtype} is not supported. '
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


SYSTEM_PROMPT = """<|im_start|>system
    - You are a helpful assistant chatbot trained by MosaicML.
    - You answer questions.
    - You are excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
    - You are more than just an information source, you are also able to write poetry, short stories, and make jokes.<|im_end|>\n"""
USER_MSG_FMT = '<|im_start|>user {}<|im_end|>\n'
ASSISTANT_MSG_FMT = '<|im_start|>assistant {}<|im_end|>\n'


def parse_args() -> Namespace:
    """Parse commandline arguments."""
    parser = ArgumentParser(
        description='Load a HF CausalLM Model and use it to generate text.')
    parser.add_argument('-n', '--name_or_path', type=str, required=True)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--max_seq_len', type=int, default=None)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--top_p', type=float, default=1.0)
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
    parser.add_argument('--eos_token_id', type=str, default=None)
    parser.add_argument('--pad_token_id', type=str, default=None)
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
    parser.add_argument('--attn_impl', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--system_prompt', type=str, default=SYSTEM_PROMPT)
    parser.add_argument('--user_msg_fmt', type=str, default=USER_MSG_FMT)
    parser.add_argument('--assistant_msg_fmt',
                        type=str,
                        default=ASSISTANT_MSG_FMT)
    return parser.parse_args()


def maybe_synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def conversation(model, tokenizer: Tokenizer, user_inp: str, history: str,
                 **generate_kwargs: Dict[str, Any]) -> Tuple[str, str, float]:
    if history != '':
        user_inp = USER_MSG_FMT.format(user_inp)
        conversation = history + user_inp
    else:
        conversation = SYSTEM_PROMPT + USER_MSG_FMT.format(user_inp)
    input_ids = tokenizer(conversation, return_tensors='pt').input_ids
    input_ids = input_ids.to(model.device)
    maybe_synchronize()
    start = time.time()
    with torch.no_grad():
        output_ids = model.generate(input_ids, **generate_kwargs)
    maybe_synchronize()
    end = time.time()
    # Slice the output_ids tensor to get only new tokens
    new_tokens = output_ids[0, len(input_ids[0]):]
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    conversation = conversation + ASSISTANT_MSG_FMT.format(output_text)
    return output_text, conversation, end - start


def have_conversation(model, tokenizer: Tokenizer,
                      **generate_kwargs: Dict[str, Any]) -> None:
    history = ''
    while True:
        print(
            "Enter your message below.\n- Type 'EOF' on a new line to send input to the model\n"
            +
            "- Type 'clear' to restart the conversation\n- Type 'history' to see the conversation\n"
            + "- Type 'quit' to end:")
        user_inp_lines = []
        while True:
            line = input()
            if line.strip() == 'EOF':
                break
            user_inp_lines.append(line)
        user_inp = '\n'.join(user_inp_lines)
        if user_inp.lower() == 'quit':
            break
        elif user_inp.lower() == 'clear':
            history = ''
            continue
        elif user_inp == 'history':
            print(f'history: {history}\n')
            continue
        assistant_resp, history, time_taken = conversation(
            model, tokenizer, user_inp, history, **generate_kwargs)
        print(f'Assistant: {assistant_resp} ({time_taken:.3f}s)\n')


def main(args: Namespace) -> None:
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
        if args.attn_impl is not None and hasattr(config, 'attn_config'):
            config.attn_config['attn_impl'] = args.attn_impl
        if args.max_seq_len is not None and hasattr(config, 'max_seq_len'):
            config.max_seq_len = args.max_seq_len

    except Exception as e:
        raise RuntimeError(
            'If you are having auth problems, try logging in via `huggingface-cli login` '
            'or by setting the environment variable `export HUGGING_FACE_HUB_TOKEN=... '
            'using your access token from https://huggingface.co/settings/tokens.'
        ) from e

    # Set device and model_dtype
    if args.device is not None:
        device = args.device
    else:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if args.model_dtype is not None:
        model_dtype = get_dtype(args.model_dtype)
    else:
        model_dtype = config.torch_dtype or torch.float32

    # Load HF Model
    print(f'Loading HF model to device={device} and dtype={model_dtype}...')
    try:
        model = AutoModelForCausalLM.from_pretrained(args.name_or_path,
                                                     config=config,
                                                     torch_dtype=model_dtype,
                                                     **from_pretrained_kwargs)
        model.to(device)
        model.eval()
        print(f'n_params={sum(p.numel() for p in model.parameters())}')
    except Exception as e:
        raise RuntimeError(
            'If you are having auth problems, try logging in via `huggingface-cli login` '
            'or by setting the environment variable `export HUGGING_FACE_HUB_TOKEN=... '
            'using your access token from https://huggingface.co/settings/tokens.'
        ) from e

    print('\nLoading HF tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(args.name_or_path,
                                              **from_pretrained_kwargs)
    if tokenizer.pad_token_id is None:
        warnings.warn(
            'pad_token_id is not set for the tokenizer. Using eos_token_id as pad_token_id.'
        )
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    generate_kwargs = {
        'max_new_tokens': args.max_new_tokens,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'top_k': args.top_k,
        'use_cache': args.use_cache,
        'do_sample': args.do_sample,
        'eos_token_id': args.eos_token_id or tokenizer.eos_token_id,
        'pad_token_id': args.pad_token_id or tokenizer.eos_token_id,
    }
    # Autocast
    if args.autocast_dtype is not None:
        autocast_dtype = get_dtype(args.autocast_dtype)
        autocast_context = torch.autocast(device, autocast_dtype)
        print(f'Using autocast with dtype={autocast_dtype}...')
    else:
        autocast_context = nullcontext()
        print('NOT using autocast...')

    # Warmup
    if args.warmup:
        print('Warming up...')
        with autocast_context:
            conversation(model, tokenizer, 'hello', '', **generate_kwargs)

    print('Starting conversation...')
    with autocast_context:
        have_conversation(model, tokenizer, **generate_kwargs)


if __name__ == '__main__':
    main(parse_args())
