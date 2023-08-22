#!/bin/bash

PROJECT="attn"
TORCH_2_IMAGE="mosaicml/pytorch:2.0.1_cu118-python3.10-ubuntu20.04"
CLUSTER_80GB=r9z1
GIT_COMMIT=v0.2.0
GIT_BRANCH=main
# 30b test Torch Runs
python submit_benchmarks.py --project $PROJECT -m 30b.yaml -g 8 --microbatch_size  3 --accum  21 --image $TORCH_2_IMAGE --git_branch $GIT_BRANCH --gpu_type h100_80gb -t bf16 --cluster $CLUSTER_80GB -s 11 11 --RUN --fsdp_config_activation_checkpointing true
python submit_benchmarks.py --project $PROJECT -m 30b.yaml -g 8 --microbatch_size  3 --accum  21 --image $TORCH_2_IMAGE --git_branch $GIT_BRANCH --gpu_type h100_80gb -t bf16 --cluster $CLUSTER_80GB -s 11 11 --RUN --fsdp_config_activation_checkpointing true --attn_impl xformers

#13b test Torch runs -- seperate Torch1.13 and torch2
python submit_benchmarks.py --project $PROJECT -m 13b.yaml -g 8 --microbatch_size 2 --accum  2 --image $TORCH_2_IMAGE --git_branch $GIT_BRANCH --gpu_type h100_80gb -t bf16 --cluster $CLUSTER_80GB -s 11 11 --RUN --fsdp_config_activation_checkpointing false
python submit_benchmarks.py --project $PROJECT -m 13b.yaml -g 8 --microbatch_size  5 --accum  3 --image $TORCH_2_IMAGE --git_branch $GIT_BRANCH --gpu_type h100_80gb -t bf16 --cluster $CLUSTER_80GB -s 11 11 --RUN --fsdp_config_activation_checkpointing false --attn_impl xformers
#7b test torch runs
python submit_benchmarks.py --project $PROJECT -m 7b.yaml -g 8 --microbatch_size  4 --accum  2 --image $TORCH_2_IMAGE --git_branch $GIT_BRANCH --gpu_type h100_80gb -t bf16 --cluster $CLUSTER_80GB -s 11 11 --RUN --fsdp_config_activation_checkpointing false
python submit_benchmarks.py --project $PROJECT -m 7b.yaml -g 8 --microbatch_size  4 --accum  2 --image $TORCH_2_IMAGE --git_branch $GIT_BRANCH --gpu_type h100_80gb -t bf16 --cluster $CLUSTER_80GB -s 11 11 --RUN --fsdp_config_activation_checkpointing false --attn_impl xformers
# #1b test torch runs
python submit_benchmarks.py --project $PROJECT -m 1b.yaml -g 8 --microbatch_size  8 --accum  2 --image $TORCH_2_IMAGE --git_branch $GIT_BRANCH --gpu_type h100_80gb -t bf16 --cluster $CLUSTER_80GB -s 11 11 --RUN --fsdp_config_activation_checkpointing false
python submit_benchmarks.py --project $PROJECT -m 1b.yaml -g 8 --microbatch_size  8 --accum  2 --image $TORCH_2_IMAGE --git_branch $GIT_BRANCH --gpu_type h100_80gb -t bf16 --cluster $CLUSTER_80GB -s 11 11 --RUN --fsdp_config_activation_checkpointing false --attn_impl xformers