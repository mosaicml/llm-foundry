# Copyright 2022-2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import os

# Define the arguments to sweep over

base_args = [
    '--project tput',
    '--image <insert_image_name>',
    '--git_branch main',
    '--precisions bf16',
    '--fsdp_config_mixed_precision PURE',
    '--fsdp_config_limit_all_gathers true',
    '--fsdp_config_forward_prefetch true',
    '--fsdp_config_backward_prefetch BACKWARD_PRE',
    '--activation_cpu_offload false',
    '--seq_len_exp 11 11',
    '--accum 1',
    '--clusters <insert_cluster_name>',
    '--gpu_types h100_80gb',
    '--data_remote <insert_data_remote_location>',
    '--wandb true',
    '--priority lowest',
    '--RUN true',
]

num_gpu_args_list = [
    [
        '--gpu_nums 128',
    ],
    [
        '--gpu_nums 256',
    ],
    [
        '--gpu_nums 512',
    ],
]

model_args_list = [
    [
        '--model_yamls 1b.yaml',
        '--fsdp_config_activation_checkpointing false',
        '--fsdp_config_shard_strategy SHARD_GRAD_OP',
        '--microbatch_size 12',
        '--attn_impl flash',
    ],
    [
        '--model_yamls 3b.yaml',
        '--fsdp_config_activation_checkpointing false',
        '--fsdp_config_shard_strategy SHARD_GRAD_OP',
        '--microbatch_size 8',
        '--attn_impl flash',
    ],
    [
        '--model_yamls 7b.yaml',
        '--fsdp_config_activation_checkpointing false',
        '--fsdp_config_shard_strategy FULL_SHARD',
        '--microbatch_size 6',
        '--attn_impl flash',
    ],
    [
        '--model_yamls 13b.yaml',
        '--fsdp_config_activation_checkpointing true',
        '--fsdp_config_shard_strategy FULL_SHARD',
        '--microbatch_size 16',
        '--attn_impl triton',
    ],
    [
        '--model_yamls 30b.yaml',
        '--fsdp_config_activation_checkpointing true',
        '--fsdp_config_shard_strategy FULL_SHARD',
        '--microbatch_size 8',
        '--attn_impl triton',
    ],
    [
        '--model_yamls 70b.yaml',
        '--fsdp_config_activation_checkpointing true',
        '--fsdp_config_shard_strategy FULL_SHARD',
        '--microbatch_size 8',
        '--attn_impl flash',
    ],
]

# Iterate over the arguments and call submit_benchmarks.py
for num_gpu_args in num_gpu_args_list:
    for model_args in model_args_list:
        command = ['python submit_benchmarks.py'
                  ] + base_args + num_gpu_args + model_args
        command = ' '.join(command)
        os.system(command)
