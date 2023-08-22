#!/bin/bash

PROJECT="ackckpt"
TORCH_2_IMAGE="mosaicml/pytorch:2.0.1_cu118-python3.10-ubuntu20.04"
CLUSTER_80GB=r1z1
GIT_COMMIT=v0.2.0

for MB_SIZE in 1 2 4 8
do
    for GATH_LMT in true false
    do
        for CPU_OFFLOAD in true false
        do
            python submit_benchmarks.py --project $PROJECT -m 3b.yaml -g 8 --microbatch_size  $MB_SIZE --accum  2 --image $TORCH_2_IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb -t bf16 --cluster $CLUSTER_80GB -s 11 11 --RUN --fsdp_config_limit_all_gathers $GATH_LMT --activation_cpu_offload $CPU_OFFLOAD --fsdp_config_activation_checkpointing false
            python submit_benchmarks.py --project $PROJECT -m 7b.yaml -g 8 --microbatch_size  $MB_SIZE --accum 2 --image $TORCH_2_IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb -t bf16 --cluster $CLUSTER_80GB -s 11 11 --RUN --fsdp_config_limit_all_gathers $GATH_LMT --activation_cpu_offload $CPU_OFFLOAD --fsdp_config_activation_checkpointing false
            python submit_benchmarks.py --project $PROJECT -m 13b.yaml -g 8 --microbatch_size  $MB_SIZE --accum  2 --image $TORCH_2_IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb -t bf16 --cluster $CLUSTER_80GB -s 11 11 --RUN --fsdp_config_limit_all_gathers $GATH_LMT --activation_cpu_offload $CPU_OFFLOAD --fsdp_config_activation_checkpointing false
        done
    done
done