# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run MPT model with FT.

This script is a modified version of
https://github.com/NVIDIA/FasterTransformer/blob/main/examples/pytorch/gpt/multi_gpu_gpt_example.py
"""

import argparse
import configparser
import os
import sys
import timeit

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, '../../..'))
from examples.pytorch.gpt.utils import comm, gpt_decoder
from examples.pytorch.gpt.utils.parallel_gpt import ParallelGPT


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer_num',
                        type=int,
                        default=32,
                        help='number of layers')
    parser.add_argument('--input_len',
                        type=int,
                        default=128,
                        help='input sequence length to generate.')
    parser.add_argument('--output_len',
                        type=int,
                        default=64,
                        help='output sequence length to generate.')
    parser.add_argument('--head_num', type=int, default=32, help='head number')
    parser.add_argument('--size_per_head',
                        type=int,
                        default=128,
                        help='size per head')
    parser.add_argument('--vocab_size',
                        type=int,
                        default=50432,
                        help='vocab size')
    parser.add_argument(
        '--beam_width',
        type=int,
        default=1,
        help='beam width for beam search. Using sampling when beam width is 1.')
    parser.add_argument('--top_k',
                        type=int,
                        default=1,
                        help='top k candidate num')
    parser.add_argument('--top_p',
                        type=float,
                        default=0.95,
                        help='top p probability threshold')
    parser.add_argument('--temperature',
                        type=float,
                        default=0.8,
                        help='temperature')
    parser.add_argument('--len_penalty',
                        type=float,
                        default=0.,
                        help='len_penalty')
    parser.add_argument('--beam_search_diversity_rate',
                        type=float,
                        default=0.,
                        help='beam_search_diversity_rate')
    parser.add_argument('--tensor_para_size',
                        type=int,
                        default=1,
                        help='tensor parallel size')
    parser.add_argument('--pipeline_para_size',
                        type=int,
                        default=1,
                        help='pipeline parallel size')
    parser.add_argument('--ckpt_path',
                        type=str,
                        default='mpt-ft-7b/1-gpu',
                        help='path to the FT checkpoint file.')
    parser.add_argument(
        '--tokenizer_name_or_path',
        type=str,
        default='EleutherAI/gpt-neox-20b',
        help=
        'Name of the tokenizer or the directory where the tokenizer file is located.'
    )
    parser.add_argument(
        '--lib_path',
        type=str,
        help=
        'path to the libth_transformer dynamic lib file(.e.g., build/lib/libth_transformer.so.'
    )
    parser.add_argument('--start_id',
                        type=int,
                        default=0,
                        help='start token id.')
    parser.add_argument('--end_id', type=int, default=0, help='end token id.')
    parser.add_argument(
        '--max_batch_size',
        type=int,
        default=8,
        help=
        'Max batch size. If sample_input_file is given, it is truncated to this max_batch_size, otherwise, this value is used as batch size.'
    )
    parser.add_argument('--repetition_penalty',
                        type=float,
                        default=5.,
                        help='repetition penalty')
    parser.add_argument(
        '--presence_penalty',
        type=float,
        default=0.,
        help=
        'presence penalty. Similar to repetition, but additive rather than multiplicative.'
    )
    parser.add_argument('--min_length',
                        type=int,
                        default=0,
                        help='A minimum number of tokens to generate')
    parser.add_argument(
        '--max_seq_len',
        type=int,
        default=2048,
        help='max sequence length for position embedding table.')
    parser.add_argument('--inference_data_type',
                        '--data_type',
                        type=str,
                        choices=['fp32', 'fp16', 'bf16'],
                        default='bf16')
    parser.add_argument('--time',
                        action='store_true',
                        help='whether or not to measure time elapsed.')
    parser.add_argument(
        '--sample_input_file',
        type=str,
        default=None,
        help=
        'path to sample input file. If not set, it runs with no context inputs.'
    )
    parser.add_argument('--sample_output_file',
                        type=str,
                        default=None,
                        help='path to sample output file.')
    parser.add_argument(
        '--disable_random_seed',
        dest='random_seed',
        action='store_false',
        help='Disable the use of random seed for sentences in a batch.')
    parser.add_argument('--skip_end_tokens',
                        dest='skip_end_tokens',
                        action='store_false',
                        help='Whether to remove or not end tokens in outputs.')
    parser.add_argument('--no_detokenize',
                        dest='detokenize',
                        action='store_false',
                        help='Skip detokenizing output token ids.')
    parser.add_argument(
        '--int8_mode',
        type=int,
        default=0,
        choices=[0, 1],
        help='The level of quantization to perform.' +
        ' 0: No quantization. All computation in data_type' +
        ' 1: Quantize weights to int8, all compute occurs in fp16/bf16. Not supported when data_type is fp32'
    )
    parser.add_argument(
        '--weights_data_type',
        type=str,
        default='fp32',
        choices=['fp32', 'fp16'],
        help='Data type of FT checkpoint weights',
    )
    parser.add_argument(
        '--return_cum_log_probs',
        type=int,
        default=0,
        choices=[0, 1, 2],
        help='Whether to compute the cumulative log probability of sentences.' +
        ' 0: do not return the cumulative log probs' +
        ' 1: return the cumulative log probs of generated sequences' +
        ' 2: return the cumulative log probs of sequences')
    parser.add_argument('--shared_contexts_ratio',
                        type=float,
                        default=0.0,
                        help='Triggers the shared context optimization when ' +
                        'compact_size <= shared_contexts_ratio * batch_size ' +
                        'A value of 0.0 deactivate the optimization')
    parser.add_argument(
        '--use_gpt_decoder_ops',
        action='store_true',
        help='Use separate decoder FT operators instead of end-to-end model op.'
    )
    parser.add_argument(
        '--no-alibi',
        dest='alibi',
        action='store_false',
        help='Do not use ALiBi (aka use_attention_linear_bias).')
    parser.add_argument(
        '--layernorm_eps',
        type=float,
        default=1e-5,
        help='layernorm eps in PyTorch, by default, is 1e-5 and 1e-6 in FT.')
    args = parser.parse_args()

    ckpt_config = configparser.ConfigParser()
    ckpt_config_path = os.path.join(args.ckpt_path, 'config.ini')
    if os.path.isfile(ckpt_config_path):
        ckpt_config.read(ckpt_config_path)
    if 'gpt' in ckpt_config.keys():
        for args_key, config_key, func in [
            ('layer_num', 'num_layer', ckpt_config.getint),
            ('max_seq_len', 'max_pos_seq_len', ckpt_config.getint),
            ('weights_data_type', 'weight_data_type', ckpt_config.get),
            ('layernorm_eps', 'layernorm_eps', ckpt_config.getfloat),
            ('alibi', 'use_attention_linear_bias', ckpt_config.getboolean),
        ]:
            if config_key in ckpt_config['gpt'].keys():
                prev_val = args.__dict__[args_key]
                args.__dict__[args_key] = func('gpt', config_key)
                print(
                    'Loading {} from config.ini,    previous: {},    current: {}'
                    .format(args_key, prev_val, args.__dict__[args_key]))
            else:
                print('Not loading {} from config.ini'.format(args_key))
        for key in ['head_num', 'size_per_head', 'tensor_para_size']:
            if key in args.__dict__:
                prev_val = args.__dict__[key]
                args.__dict__[key] = ckpt_config.getint('gpt', key)
                print(
                    'Loading {} from config.ini,    previous: {},    current: {}'
                    .format(key, prev_val, args.__dict__[key]))
            else:
                print('Not loading {} from config.ini'.format(key))

    layer_num = args.layer_num
    output_len = args.output_len
    head_num = args.head_num
    size_per_head = args.size_per_head
    vocab_size = args.vocab_size
    beam_width = args.beam_width
    top_k = args.top_k
    top_p = args.top_p
    temperature = args.temperature
    len_penalty = args.len_penalty
    beam_search_diversity_rate = args.beam_search_diversity_rate
    tensor_para_size = args.tensor_para_size
    pipeline_para_size = args.pipeline_para_size
    start_id = args.start_id
    end_id = args.end_id
    max_batch_size = args.max_batch_size
    max_seq_len = args.max_seq_len
    repetition_penalty = args.repetition_penalty
    presence_penalty = args.presence_penalty
    min_length = args.min_length
    weights_data_type = args.weights_data_type
    return_cum_log_probs = args.return_cum_log_probs
    return_output_length = return_cum_log_probs > 0
    shared_contexts_ratio = args.shared_contexts_ratio
    layernorm_eps = args.layernorm_eps
    use_attention_linear_bias = args.alibi
    has_positional_encoding = not args.alibi

    print('\n=================== Arguments ===================')
    for k, v in vars(args).items():
        print(f'{k.ljust(30, ".")}: {v}')
    print('=================================================\n')

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    torch.manual_seed(0)

    comm.initialize_model_parallel(args.tensor_para_size,
                                   args.pipeline_para_size)
    rank = comm.get_rank()
    device = comm.get_device()

    # Inputs
    contexts = []
    if args.sample_input_file:
        with open(args.sample_input_file, 'r') as f:
            contexts = f.read().splitlines()
            batch_size = min(len(contexts), max_batch_size)
        contexts = contexts[:batch_size]
        start_ids = [
            torch.tensor(tokenizer.encode(c), dtype=torch.int32, device=device)
            for c in contexts
        ]
    else:
        batch_size = max_batch_size
        contexts = ['<|endoftext|>'] * batch_size
        start_ids = [torch.IntTensor([end_id for _ in range(args.input_len)])
                    ] * batch_size

    start_lengths = [len(ids) for ids in start_ids]

    start_ids = pad_sequence(start_ids, batch_first=True, padding_value=end_id)
    start_lengths = torch.IntTensor(start_lengths)

    # Prepare model.
    if not args.use_gpt_decoder_ops:
        gpt = ParallelGPT(head_num,
                          size_per_head,
                          vocab_size,
                          start_id,
                          end_id,
                          layer_num,
                          max_seq_len,
                          tensor_para_size,
                          pipeline_para_size,
                          lib_path=args.lib_path,
                          inference_data_type=args.inference_data_type,
                          int8_mode=args.int8_mode,
                          weights_data_type=weights_data_type,
                          layernorm_eps=layernorm_eps,
                          use_attention_linear_bias=use_attention_linear_bias,
                          has_positional_encoding=has_positional_encoding,
                          shared_contexts_ratio=shared_contexts_ratio)
        if not gpt.load(ckpt_path=args.ckpt_path):
            print(
                '[WARNING] Checkpoint file not found. Model loading is skipped.'
            )
    else:
        gpt = gpt_decoder.Gpt(num_heads=head_num,
                              size_per_head=size_per_head,
                              num_layers=layer_num,
                              vocab_size=vocab_size,
                              start_id=start_id,
                              end_id=end_id,
                              tensor_para_size=tensor_para_size,
                              pipeline_para_size=pipeline_para_size,
                              lib_path=args.lib_path,
                              max_seq_len=max_seq_len,
                              int8_mode=args.int8_mode,
                              weights_data_type=args.weights_data_type)
        gpt.load(args.ckpt_path, args.inference_data_type)

    if args.random_seed:
        random_seed_tensor = torch.randint(0,
                                           10000,
                                           size=[batch_size],
                                           dtype=torch.int64)
    else:
        random_seed_tensor = torch.zeros([batch_size], dtype=torch.int64)

    repetition_penalty_vec = None if repetition_penalty == 1. else repetition_penalty * torch.ones(
        batch_size, dtype=torch.float32)
    presence_penalty_vec = None if presence_penalty == 0. else presence_penalty * torch.ones(
        batch_size, dtype=torch.float32)

    infer_decode_args = {
        'beam_width':
            beam_width,
        'top_k':
            top_k * torch.ones(batch_size, dtype=torch.int32),
        'top_p':
            top_p * torch.ones(batch_size, dtype=torch.float32),
        'temperature':
            temperature * torch.ones(batch_size, dtype=torch.float32),
        'repetition_penalty':
            repetition_penalty_vec,
        'presence_penalty':
            presence_penalty_vec,
        'beam_search_diversity_rate':
            beam_search_diversity_rate *
            torch.ones(batch_size, dtype=torch.float32),
        'len_penalty':
            len_penalty * torch.ones(size=[batch_size], dtype=torch.float32),
        'bad_words_list':
            None,
        'min_length':
            min_length * torch.ones(size=[batch_size], dtype=torch.int32),
        'random_seed':
            random_seed_tensor
    }

    if not args.use_gpt_decoder_ops:

        def gpt_generate_fn():
            tokens_batch = gpt(start_ids,
                               start_lengths,
                               output_len,
                               return_output_length=return_output_length,
                               return_cum_log_probs=return_cum_log_probs,
                               **infer_decode_args)
            return tokens_batch
    else:

        def gpt_generate_fn():
            output_dict = gpt.generate(
                input_token_ids=start_ids,
                input_lengths=start_lengths,
                gen_length=output_len,
                eos_token_id=end_id,
                return_output_length=return_output_length,
                return_log_probs=return_cum_log_probs,
                **infer_decode_args)
            return output_dict

    # Generate tokens.
    gen_outputs = gpt_generate_fn()

    if rank == 0:
        if not args.use_gpt_decoder_ops:
            if return_cum_log_probs > 0:
                tokens_batch, _, cum_log_probs = gen_outputs
            else:
                tokens_batch, cum_log_probs = gen_outputs, None
        else:
            tokens_batch = gen_outputs['output_token_ids']
            cum_log_probs = gen_outputs[
                'cum_log_probs'] if return_cum_log_probs > 0 else None
        if cum_log_probs is not None:
            print('[INFO] Log probs of sentences:', cum_log_probs)

        outputs = []
        tokens_batch = tokens_batch.cpu().numpy()
        for i, (context, tokens) in enumerate(zip(contexts, tokens_batch)):
            for beam_id in range(beam_width):
                token = tokens[beam_id][
                    start_lengths[i]:]  # exclude context input from the output
                if args.skip_end_tokens:
                    token = token[token != end_id]
                output = tokenizer.decode(
                    token) if args.detokenize else ' '.join(
                        str(t) for t in token.tolist())
                outputs.append(output)
                print(
                    f'[INFO] batch {i}, beam {beam_id}:\n[Context]\n{context}\n\n[Output]\n{output}\n'
                )

        if args.sample_output_file:
            with open(args.sample_output_file, 'w+') as f:
                outputs = [o.replace('\n', '\\n') for o in outputs]
                f.writelines('\n'.join(outputs))

    # Measure inference time.
    if args.time:
        warmup_iterations = 10
        for _ in range(warmup_iterations):
            gpt_generate_fn()
        torch.cuda.synchronize()
        measurement_iterations = 10
        time = timeit.default_timer()
        for _ in range(measurement_iterations):
            gpt_generate_fn()
        torch.cuda.synchronize()
        time_elapsed = timeit.default_timer() - time
        if rank == 0:
            print(f'[INFO] MPT time costs:')
            print(
                'model_name, gpu_type, gpu_count, batch_size, input_tokens, output_tokens, latency_ms'
            )
            print(
                f'{ckpt_config.get("gpt", "model_name")}, {torch.cuda.get_device_name().replace(" ", "-")}, {torch.cuda.device_count()}, {batch_size}, {args.input_len}, {args.output_len}, {time_elapsed * 1000 / measurement_iterations:.2f}'
            )


if __name__ == '__main__':
    main()
