#!/bin/bash

PROJECT="torchcompile"
TORCH_2_IMAGE="mosaicml/pytorch:2.0.1_cu118-python3.10-ubuntu20.04"
CLUSTER_80GB=r1z1
GIT_COMMIT=v0.2.0

for FULLGRAPH in false
do
    for DYNAMIC in true false 
    do
        for MODE in default reduce-overhead max-autotune
        do 
            python submit_benchmarks.py --project $PROJECT -m 125m.yaml -g 8 --microbatch_size 60 --accum 32 --image $TORCH_2_IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb -t bf16 --cluster $CLUSTER_80GB -s 11 11 --RUN \
                                        --torch_compile_fullgraph $FULLGRAPH --torch_compile_dynamic $DYNAMIC --torch_compile_mode $MODE

            python submit_benchmarks.py --project $PROJECT -m 760m.yaml -g 8 --microbatch_size 24 --accum  2 --image $TORCH_2_IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb -t bf16 --cluster $CLUSTER_80GB -s 11 11 --RUN \
                                        --torch_compile_fullgraph $FULLGRAPH --torch_compile_dynamic $DYNAMIC --torch_compile_mode $MODE

            python submit_benchmarks.py --project $PROJECT -m 13b.yaml -g 8 --microbatch_size  20 --accum  3 --image $TORCH_2_IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb -t bf16 --cluster $CLUSTER_80GB -s 11 11 --RUN \
                                        --torch_compile_fullgraph $FULLGRAPH --torch_compile_dynamic $DYNAMIC --torch_compile_mode $MODE --fsdp_config_activation_checkpointing true

        done
    done
done