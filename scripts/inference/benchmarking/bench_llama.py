# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import sys
import time
import copy
import warnings

import deepspeed
import numpy as np
import torch
# You can use this to load the model weights
from composer.core import get_precision_context
from composer.utils import get_device
from omegaconf import OmegaConf as om
from transformers import AutoTokenizer

from llmfoundry.models.layers.blocks import MPTBlock
import os
from transformers import AutoModel, AutoConfig, LlamaForCausalLM, LlamaConfig, LlamaTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaAttention, LlamaMLP

def main(tp=False):
    batch_sizes = [1, 2, 4, 8]
    input_lengths = [512]
    output_lengths = [8]
    num_runs = 3
    num_warmup_runs = 1
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    
    dtype = torch.float16

    inf_config = {
        'replace_with_kernel_inject': True,
        'dtype': dtype,
        'enable_cuda_graph': True,
        'replace_method': 'auto',
        'tensor_parallel': {
            'tp_size': 0
        },
    }
    if tp:
        assert world_size > 1, "Trying to run Tensor Parallelism with World Size=1"
        print("***** Tensor Parallelism with World Size:", world_size)
        # Must disable replace_with_kernel_inject for tensor parallelism
        inf_config = {
            'dtype': dtype, 
            'tensor_parallel': {
                'tp_size': world_size
            },            
            #'injection_policy': {LlamaDecoderLayer: ('mlp.down_proj')}
        }

    print("Inference Config:", inf_config)
    
    # config.model.init_device = 'cpu'

    hf_model_name = '/mnt/workdisk/rishab/meta-llama/Llama-2-70b-hf'

    with deepspeed.OnDevice(dtype=dtype, device="meta"):
        
        model = LlamaForCausalLM.from_pretrained(hf_model_name, low_cpu_mem_usage=True)
        hf_config = LlamaConfig.from_pretrained(hf_model_name, low_cpu_mem_usage=True)
    
        print("Loaded model!")

        model.eval()
    
        print("HF model:", model)

        tokenizer = LlamaTokenizer.from_pretrained(hf_model_name, low_cpu_mem_usage=True)
    
        # Deepspeed's init_inference takes in a huggingface model, which is the .model
        # object of our ComposerModel object.
        ds_engine = deepspeed.init_inference(model, config=inf_config)
        model = ds_engine.module

        print("Deepspeed model for inference:", model)


    stats = []
    print('Run Name\tLatency\tTokens per Second')
    for batch_size in batch_sizes:
        for input_length in input_lengths:
            for output_length in output_lengths:
                times = []
                eos_token = tokenizer.eos_token
                # Make sure we are not generating a fake batch with a EOS token
                while True:
                    batch = torch.randint(
                        0,
                        hf_config.vocab_size - 1,
                        size=(batch_size, input_length
                             )).to(f'cuda:{torch.cuda.current_device()}')
                    if tokenizer.convert_tokens_to_ids(eos_token) not in batch:
                        break
                batch = batch.to(torch.long)
                torch.cuda.synchronize()
                for i in range(num_runs + num_warmup_runs):
                    start_time = time.time()
                    with torch.no_grad():
                        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True) as prof:
                        model.generate(
                            batch,
                            max_new_tokens=output_length,
                            use_cache=True)
                    torch.cuda.synchronize()
                    # We noticed there sometimes might be a small bit of startup time
                    # so we only start to benchmark after the first iteration
                    if i >= num_warmup_runs:
                        times.append(time.time() - start_time)

                #print(prof.key_averages(group_by_stack_n=10).table(sort_by="cpu_time_total", row_limit=250))
                #prof.export_chrome_trace('trace-deepspeed-256.json')
                num_output_tokens = output_length * batch_size
                mean_time = np.mean(times)
                tokens_per_second = num_output_tokens / float(mean_time)

                resu = (
                    f'{hf_config._name_or_path}_{batch_size}_{input_length}_{output_length}',
                    f'{mean_time:.3f}', f'{tokens_per_second:.3f}')

                run_name, latency, tokens_per_second = resu

                print(f'{run_name}\t\t{latency}\t\t{tokens_per_second}')

                stats.append(resu)

    print('=' * 75)
    print('name, latency (s), tokens / s')
    for val in stats:
        print(val)


if __name__ == '__main__':
    enable_tensor_parallelism = True

    #yaml_path, args_list = sys.argv[1], sys.argv[2:]    
    #with open(yaml_path) as f:
    #    yaml_config = om.load(f)
    #cli_config = om.from_cli(args_list)
    # config = om.merge(yaml_config, cli_config)
    #print(config)
    main(tp=enable_tensor_parallelism)

