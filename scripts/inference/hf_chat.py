# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import time
import warnings
from argparse import ArgumentParser, ArgumentTypeError, Namespace
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedModel, PreTrainedTokenizerBase,
                          StoppingCriteria, StoppingCriteriaList, TextStreamer)


class ChatFormatter:
    """A class for formatting the chat history.

    Args:
        system: The system prompt. If None, a default ChatML-formatted prompt is used.
        user: The user prompt. If None, a default ChatML value is used.
        assistant: The assistant prompt. If None, a default ChatML value is used.

    Attributes:
        system: The system prompt.
        user: The user prompt.
        assistant: The assistant prompt.
        response_prefix: The response prefix (anything before {} in the assistant format string)
    """

    def __init__(self, system: str, user: str, assistant: str) -> None:
        self.system = system if system else '<|im_start|>system\nA conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.<|im_end|>\n'
        self.user = user if user else '<|im_start|>user\n{}<|im_end|>\n'
        self.assistant = assistant if assistant else '<|im_start|>assistant\n{}<|im_end|>\n'
        self.response_prefix = self.assistant.split('{}')[0]


class Conversation:
    """A class for interacting with a chat-tuned LLM.

    Args:
        model: The model to use for inference.
        tokenizer: The tokenizer to use for inference.
        chat_format: The chat format to use for the conversation.
        generate_kwargs: The keyword arguments to pass to `model.generate`.
        stop_tokens: The tokens to stop generation on.

    Attributes:
        model: The model to use for inference.
        tokenizer: The tokenizer to use for inference.
        chat_format: The chat format to use for the conversation.
        streamer: The streamer to use for inference.
        generate_kwargs: The keyword arguments to pass to `model.generate`.
        history: The conversation history.
        cli_instructions: The instructions to display to the user.
    """

    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizerBase,
                 chat_format: ChatFormatter,
                 generate_kwargs: Dict[str, Any],
                 stop_tokens: Optional[List[str]] = None) -> None:
        if stop_tokens is None:
            stop_tokens = ['<|endoftext|>', '<|im_end|>']
        self.model = model
        self.tokenizer = tokenizer
        self.chat_format = chat_format

        stop_token_ids = self.tokenizer.convert_tokens_to_ids(stop_tokens)
        if len(stop_token_ids) != len(stop_tokens):
            warnings.warn(
                f'Not all stop tokens were found in the tokenizer vocabulary: {stop_tokens}\n'
                + 'Generation may stop or continue unexpectedly.')

        class StopOnTokens(StoppingCriteria):

            def __call__(self, input_ids: torch.LongTensor,
                         scores: torch.FloatTensor, **kwargs: Any) -> bool:
                del kwargs  # unused
                for stop_id in stop_token_ids:
                    if input_ids[0][-1] == stop_id:
                        return True
                return False

        self.streamer = TextStreamer(tokenizer,
                                     skip_prompt=True,
                                     skip_special_tokens=True)
        self.generate_kwargs = {
            **generate_kwargs,
            'stopping_criteria':
                StoppingCriteriaList([StopOnTokens()]),
            'streamer':
                self.streamer,
        }
        self.history = []
        self.cli_instructions = (
            'Enter your message below.\n- Hit return twice to send input to the model\n'
            +
            "- Type 'clear' to restart the conversation\n- Type 'history' to see the conversation\n"
            +
            "- Type 'quit' to end\n- Type 'system' to change the system prompt\n"
        )

    def _history_as_formatted_str(self) -> str:
        text = self.chat_format.system + ''.join([
            '\n'.join([
                self.chat_format.user.format(item[0]),
                self.chat_format.assistant.format(item[1]),
            ]) for item in self.history[:-1]
        ])
        text += self.chat_format.user.format(self.history[-1][0])
        text += self.chat_format.response_prefix
        return text

    def turn(self, user_inp: str) -> None:
        self.history.append([user_inp, ''])
        conversation = self._history_as_formatted_str()
        input_ids = self.tokenizer(conversation, return_tensors='pt').input_ids
        input_ids = input_ids.to(self.model.device)
        # also stream to stdout
        maybe_synchronize()
        start = time.time()
        print('Assistant:')
        gkwargs = {**self.generate_kwargs, 'input_ids': input_ids}
        # this will stream to stdout, but we need to keep track of the output_ids for saving history
        output_ids = self.model.generate(**gkwargs)
        maybe_synchronize()
        end = time.time()
        print(f'took {end - start:.2f} seconds')
        new_tokens = output_ids[0, len(input_ids[0]):]
        assistant_response = self.tokenizer.decode(new_tokens,
                                                   skip_special_tokens=True)
        self.history[-1][-1] = assistant_response

    def __call__(self) -> None:
        print(self.cli_instructions)
        while True:
            print('User:')
            user_inp_lines = []
            while True:
                line = input()
                if line.strip() == '':
                    break
                user_inp_lines.append(line)
            user_inp = '\n'.join(user_inp_lines)
            if user_inp.lower() == 'quit':
                break
            elif user_inp.lower() == 'clear':
                self.history = []
                continue
            elif user_inp == 'history':
                print(f'history: {self.history}')
                continue
            elif user_inp == 'history_fmt':
                print(f'history: {self._history_as_formatted_str()}')
                continue
            elif user_inp == 'system':
                print('Enter a new system prompt:')
                new_system = input()
                sys = f'<|im_start|>system\n{new_system.strip()}.<|im_end|>\n'
                self.chat_format.system = sys
                continue
            self.turn(user_inp)


def get_dtype(dtype: str):
    if dtype == 'fp32':
        return torch.float32
    elif dtype == 'fp16':
        return torch.float16
    elif dtype == 'bf16':
        return torch.bfloat16
    else:
        raise NotImplementedError(
            f'dtype {dtype} is not supported. ' +
            'We only support fp32, fp16, and bf16 currently')


def str2bool(v: Union[str, bool]):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def str_or_bool(v: Union[str, bool]):
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
    parser.add_argument('--device_map', type=str, default=None)
    parser.add_argument('--attn_impl', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--system_prompt', type=str, default=None)
    parser.add_argument('--user_msg_fmt', type=str, default=None)
    parser.add_argument('--assistant_msg_fmt', type=str, default=None)
    parser.add_argument(
        '--stop_tokens',
        type=str,
        default='<|endoftext|> <|im_end|>',
        help='A string of tokens to stop generation on; will be split on spaces.'
    )
    return parser.parse_args()


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
        if hasattr(config, 'init_device') and device is not None:
            config.init_device = device
        if args.max_seq_len is not None and hasattr(config, 'max_seq_len'):
            config.max_seq_len = args.max_seq_len

    except Exception as e:
        raise RuntimeError(
            'If you are having auth problems, try logging in via `huggingface-cli login` '
            +
            'or by setting the environment variable `export HUGGING_FACE_HUB_TOKEN=... '
            +
            'using your access token from https://huggingface.co/settings/tokens.'
        ) from e

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
            'Unable to load HF model. ' +
            'If you are having auth problems, try logging in via `huggingface-cli login` '
            +
            'or by setting the environment variable `export HUGGING_FACE_HUB_TOKEN=... '
            +
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
        autocast_context = torch.autocast(model.device.type, autocast_dtype)
        print(f'Using autocast with dtype={autocast_dtype}...')
    else:
        autocast_context = nullcontext()
        print('NOT using autocast...')

    chat_format = ChatFormatter(system=args.system_prompt,
                                user=args.user_msg_fmt,
                                assistant=args.assistant_msg_fmt)

    conversation = Conversation(model=model,
                                tokenizer=tokenizer,
                                chat_format=chat_format,
                                generate_kwargs=generate_kwargs,
                                stop_tokens=args.stop_tokens.split())

    # Warmup
    if args.warmup:
        print('Warming up...')
        with autocast_context:
            conversation.turn('Write a welcome message to the user.')
            conversation.history = []

    print('Starting conversation...')
    with autocast_context:
        conversation()


if __name__ == '__main__':
    main(parse_args())
