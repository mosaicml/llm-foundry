# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import sys
import time
from contextlib import nullcontext

import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf as om

from llmfoundry.utils.builders import build_composer_model, build_tokenizer


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
            f'We only support fp32, fp16, and bf16 currently')


def compare_dtype(dtype: torch.dtype, param_dtype: torch.dtype):
    if dtype != param_dtype:
        raise ValueError(
            f'dtype type is: {dtype} but model dtype is: {param_dtype}. ' +
            f"The expected dtype and model dtype don't match.")


def main(config: DictConfig):
    if config.device is not None:
        device = config.device
    else:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model_dtype = get_dtype(config.model_dtype)
    print(f'Using device={device} and dtype={model_dtype}...')

    if config.autocast_dtype is not None:
        autocast_dtype = get_dtype(config.autocast_dtype)
        autocast_context = torch.autocast(device, autocast_dtype)
        print(f'Using autocast with dtype={autocast_dtype}...')
    else:
        autocast_context = nullcontext()
        print('NOT using autocast...')

    inference_config = {
        'replace_with_kernel_inject': True,
        'dtype': model_dtype,
        'replace_method': 'auto',
        'enable_cuda_graph': False,
        'tensor_parallel': {
            'tp_size': 0
        },
    }

    tokenizer_name = config.tokenizer['name']
    tokenizer_kwargs = config.tokenizer.get('kwargs', {})
    tokenizer = build_tokenizer(
        tokenizer_name=tokenizer_name,
        tokenizer_kwargs=tokenizer_kwargs,
    )
    name = config.model.pop('name')
    composer_model = build_composer_model(
        name=name,
        tokenizer=tokenizer,
        cfg=config.model,
    )
    model = composer_model.model
    model.eval()

    if config.use_deepspeed:
        import deepspeed  # type: ignore
        model = deepspeed.init_inference(model, config=inference_config)

        # Checking if deepspeed casts dtypes correctly
        for _, p in model.named_parameters():
            compare_dtype(model_dtype, p.dtype)
            break
    else:
        model.to(device=device, dtype=model_dtype)

    n_params = sum(p.numel() for p in model.parameters())
    print('n_params is: ', n_params)

    print(
        'name, latency (s), throughput (tokens/s), latency_per_sequence_output_token (ms)'
    )
    print('=' * 75)

    for batch_size in config.batch_sizes:
        for input_length in config.input_lengths:
            for output_length in config.output_lengths:
                batch = torch.randint(0,
                                      config.model.vocab_size - 1,
                                      size=(batch_size,
                                            input_length)).to(device)

                # We're just going to have generate eos, padding tokens be
                # ignored by HF generate
                batch = batch.to(torch.long)
                attention_mask = torch.ones_like(batch)

                start_time = 0
                for i in range(config.num_batches + config.num_warmup_batches):
                    if i == config.num_warmup_batches:
                        torch.cuda.synchronize()
                        start_time = time.time()
                    with torch.no_grad():
                        with autocast_context:
                            model.generate(batch,
                                           max_new_tokens=output_length,
                                           use_cache=config.use_cache,
                                           attention_mask=attention_mask,
                                           eos_token_id=None,
                                           pad_token_id=None)

                torch.cuda.synchronize()
                mean_time = (time.time() - start_time) / config.num_batches

                num_output_tokens = output_length * batch_size
                tokens_per_second = num_output_tokens / mean_time
                ms_per_seq_output_token = mean_time * 1000 / output_length

                run_name = f'{config.benchmark_name}_{batch_size}_{input_length}_{output_length}'
                print(
                    f'{run_name}, {mean_time:.3f}, {tokens_per_second:.3f}, {ms_per_seq_output_token:.3f}'
                )


if __name__ == '__main__':
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open(yaml_path) as f:
        yaml_config = om.load(f)
    cli_config = om.from_cli(args_list)
    config = om.merge(yaml_config, cli_config)
    assert isinstance(config, DictConfig)
    main(config)
