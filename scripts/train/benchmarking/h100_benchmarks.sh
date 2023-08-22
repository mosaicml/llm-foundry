#!/bin/bash

PROJECT="h100setup"
TORCH_2_IMAGE="mosaicml/pytorch:2.0.1_cu118-python3.10-ubuntu20.04"
CLUSTER_80GB=r9z1
GIT_COMMIT=v0.2.0
GIT_BRANCH=main
# 30b test Torch Runs
# python submit_benchmarks.py --project $PROJECT -m 30b.yaml -g 8 --microbatch_size  1 --accum  21 --image $TORCH_2_IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb -t fp8 --cluster $CLUSTER_80GB -s 12 13 --RUN --fsdp_config_activation_checkpointing true
python submit_benchmarks.py --project $PROJECT -m 350m.yaml -g 8 --microbatch_size  3 --accum  21 --image $TORCH_2_IMAGE --git_branch $GIT_BRANCH --gpu_type h100_80gb -t fp8 --cluster $CLUSTER_80GB -s 11 11 --RUN --fsdp_config_activation_checkpointing true
# python submit_benchmarks.py --project $PROJECT -m 30b.yaml -g 8 --microbatch_size  6 --accum  21 --image $TORCH_2_IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb  -t fp16 --cluster $CLUSTER_80GB -s 10 10 --RUN --fsdp_config_activation_checkpointing true
# python submit_benchmarks.py --project $PROJECT -m 30b.yaml -g 8 --microbatch_size  12 --accum  21 --image $TORCH_2_IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb -t fp16 --cluster $CLUSTER_80GB -s 9 9 --RUN --fsdp_config_activation_checkpointing true

# #13b test Torch runs -- seperate Torch1.13 and torch2
# python submit_benchmarks.py --project $PROJECT -m 13b.yaml -g 8 --microbatch_size  1 --accum  3 --image $TORCH_2_IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb -t fp16 --cluster $CLUSTER_80GB -s 15 15 --RUN --fsdp_config_activation_checkpointing true
# python submit_benchmarks.py --project $PROJECT -m 13b.yaml -g 8 --microbatch_size  5 --accum  3 --image $TORCH_2_IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb -t fp16 --cluster $CLUSTER_80GB -s 13 13 --RUN --fsdp_config_activation_checkpointing true
# python submit_benchmarks.py --project $PROJECT -m 13b.yaml -g 8 --microbatch_size  20 --accum  3 --image $TORCH_2_IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb -t fp16 --cluster $CLUSTER_80GB -s 11 11 --RUN --fsdp_config_activation_checkpointing true
# python submit_benchmarks.py --project $PROJECT -m 13b.yaml -g 8 --microbatch_size  80 --accum  3 --image $TORCH_2_IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb -t fp16 --cluster $CLUSTER_80GB -s 9 9 --RUN --fsdp_config_activation_checkpointing true

# #7b test torch runs
# python submit_benchmarks.py --project $PROJECT -m 7b.yaml -g 8 --microbatch_size  2 --accum  2 --image $TORCH_2_IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb -t fp16 --cluster $CLUSTER_80GB -s 15 15 --RUN --fsdp_config_activation_checkpointing true
# python submit_benchmarks.py --project $PROJECT -m 7b.yaml -g 8 --microbatch_size  8 --accum  2 --image $TORCH_2_IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb -t fp16 --cluster $CLUSTER_80GB -s 13 13 --RUN --fsdp_config_activation_checkpointing true
# python submit_benchmarks.py --project $PROJECT -m 7b.yaml -g 8 --microbatch_size  32 --accum  2 --image $TORCH_2_IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb -t fp16 --cluster $CLUSTER_80GB -s 11 11 --RUN --fsdp_config_activation_checkpointing true
# python submit_benchmarks.py --project $PROJECT -m 7b.yaml -g 8 --microbatch_size  128 --accum  2 --image $TORCH_2_IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb -t fp16 --cluster $CLUSTER_80GB -s 9 9 --RUN --fsdp_config_activation_checkpointing true

# # #3b test torch runs
# python submit_benchmarks.py --project $PROJECT -m 3b.yaml -g 8 --microbatch_size  3 --accum  6 --image $TORCH_2_IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb -t fp16 --cluster $CLUSTER_80GB -s 15 15 --RUN --fsdp_config_activation_checkpointing true
# python submit_benchmarks.py --project $PROJECT -m 3b.yaml -g 8 --microbatch_size  3 --accum  6 --image $TORCH_2_IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb -t fp16 --cluster $CLUSTER_80GB -s 13 13 --RUN --fsdp_config_activation_checkpointing false
# python submit_benchmarks.py --project $PROJECT -m 3b.yaml -g 8 --microbatch_size  10 --accum  6 --image $TORCH_2_IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb -t fp16 --cluster $CLUSTER_80GB -s 11 11 --RUN --fsdp_config_activation_checkpointing false
# python submit_benchmarks.py --project $PROJECT -m 3b.yaml -g 8 --microbatch_size  40 --accum  6 --image $TORCH_2_IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb -t fp16 --cluster $CLUSTER_80GB -s 9 9 --RUN --fsdp_config_activation_checkpointing false

# #1b test torch runs
# python submit_benchmarks.py --project $PROJECT -m 1b.yaml -g 8 --microbatch_size  1 --accum  4 --image $TORCH_2_IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb -t fp16 --cluster $CLUSTER_80GB -s 15 15 --RUN --fsdp_config_activation_checkpointing false
# python submit_benchmarks.py --project $PROJECT -m 1b.yaml -g 8 --microbatch_size  2 --accum  4 --image $TORCH_2_IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb -t fp16 --cluster $CLUSTER_80GB -s 13 13 --RUN --fsdp_config_activation_checkpointing false
# python submit_benchmarks.py --project $PROJECT -m 1b.yaml -g 8 --microbatch_size  56 --accum  4 --image $TORCH_2_IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb -t fp16 --cluster $CLUSTER_80GB -s 9 9 --RUN --fsdp_config_activation_checkpointing false

# #abbreviate it, 350m
# python submit_benchmarks.py --project $PROJECT -m 350m.yaml -g 8 --microbatch_size  1 --accum  4 --image $TORCH_2_IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb -t fp16 --cluster $CLUSTER_80GB -s 15 15 --RUN --fsdp_config_activation_checkpointing false
# python submit_benchmarks.py --project $PROJECT -m 350m.yaml -g 8 --microbatch_size  2 --accum  4 --image $TORCH_2_IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb -t fp16 --cluster $CLUSTER_80GB -s 14 14 --RUN --fsdp_config_activation_checkpointing false
# python submit_benchmarks.py --project $PROJECT -m 350m.yaml -g 8 --microbatch_size  56 --accum  4 --image $TORCH_2_IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb -t fp16 --cluster $CLUSTER_80GB -s 9 9 --RUN --fsdp_config_activation_checkpointing false
