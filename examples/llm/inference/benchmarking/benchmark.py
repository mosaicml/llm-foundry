# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import sys
import time

import numpy as np
import torch
# You can use this to load the model weights
from omegaconf import OmegaConf as om

from examples.llm.src import COMPOSER_MODEL_REGISTRY


def get_precision(precision):
    if precision == 'fp32':
        return torch.float32
    elif precision == 'fp16':
        return torch.float16
    elif precision == 'bf16':
        return torch.bfloat16
    else:
        raise NotImplementedError(
            f'Precision of type {precision} is not supported. '
            f'We only support fp32, amp_fp16, and amp_bf16 currently')


def compare_precision(precision, param_dtype):
    if precision != param_dtype:
        raise ValueError(
            f'Precision type is: {precision} but model dtype is: {param_dtype}. '
            f"The expected precision and model precision don't match.")


def main(config):
    model_dtype = get_precision(config.model_dtype)
    autocast_precision = None
    if config.autocast_precision is not None:
        autocast_precision = get_precision(config.autocast_precision)

    inference_config = {
        'replace_with_kernel_inject': True,
        'dtype': model_dtype,
        'replace_method': 'auto',
        'enable_cuda_graph': False,
        'tensor_parallel': {
            'tp_size': 0
        },
    }

    composer_model = COMPOSER_MODEL_REGISTRY[config.model.name](
        config.model, config.tokenizer)

    model = composer_model.model

    model.eval()

    if config.use_deepspeed:
        import deepspeed  # type: ignore
        model = deepspeed.init_inference(model, config=inference_config)

        # Checking if deepspeed casts dtypes correctly
        for _, p in model.named_parameters():
            compare_precision(model_dtype, p.dtype)
            break
    else:
        model.to(torch.cuda.current_device())
        model.to(model_dtype)

    n_params = sum(p.numel() for p in model.parameters())
    print('n_params is: ', n_params)

    print('name, latency (s), tokens / s, output token time (ms)')
    print('=' * 75)

    stats = []
    for batch_size in config.batch_sizes:
        for input_length in config.input_lengths:
            for output_length in config.output_lengths:
                times = []

                batch = torch.randint(
                    0,
                    config.model.vocab_size - 1,
                    size=(
                        batch_size,
                        input_length)).to(f'cuda:{torch.cuda.current_device()}')

                # We're just going to have generate eos, padding tokens be
                # ignored by HF generate
                batch = batch.to(torch.long)
                attention_mask = torch.ones_like(batch)

                torch.cuda.synchronize()

                for i in range(config.num_runs + 1):
                    start_time = time.time()
                    with torch.no_grad():
                        precision_context = contextlib.nullcontext()
                        if autocast_precision is not None and autocast_precision in [
                                'fp16', 'bf16'
                        ]:
                            precision_context = torch.cuda.amp.autocast(
                                True, dtype=autocast_precision)

                        with precision_context:
                            model.generate(batch,
                                           max_new_tokens=output_length,
                                           use_cache=True,
                                           attention_mask=attention_mask,
                                           eos_token_id=None,
                                           pad_token_id=None)

                    torch.cuda.synchronize()

                    # We noticed there sometimes might be a small bit of startup time
                    # so we only start to benchmark after some number of batches
                    if i >= config.num_warmup_batches:
                        times.append(time.time() - start_time)

                num_output_tokens = output_length * batch_size
                mean_time = np.mean(times)
                tokens_per_second = num_output_tokens / float(mean_time)
                ms_per_seq_output_token = float(
                    mean_time) * 1000 / num_output_tokens

                result = (
                    f'{config.benchmark_name}_{batch_size}_{input_length}_{output_length}',
                    f'{mean_time:.3f}', f'{tokens_per_second:.3f}',
                    f'{ms_per_seq_output_token:.3f}')

                run_name, latency, tokens_per_second, ms_per_seq_output_token = result

                print(
                    f'{run_name}, {latency}, {tokens_per_second}, {ms_per_seq_output_token}'
                )

                stats.append(result)

    print('=' * 75)
    print('name, latency (s), tokens / s, output token time (ms)')
    for val in stats:
        run_name, latency, tokens_per_second, ms_per_seq_output_token = val
        print(
            f'{run_name}, latency (s) {latency}, tokens per second {tokens_per_second}, output token time (ms) {ms_per_seq_output_token}'
        )


if __name__ == '__main__':
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open(yaml_path) as f:
        yaml_config = om.load(f)
    cli_config = om.from_cli(args_list)
    config = om.merge(yaml_config, cli_config)
    main(config)
