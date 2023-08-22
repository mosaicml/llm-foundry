#!/bin/bash

PROJECT="shard_grad_op"
TORCH_2_IMAGE="mosaicml/pytorch:2.0.1_cu118-python3.10-ubuntu20.04"
CLUSTER_80GB=r1z1
GIT_COMMIT=v0.2.0

# python submit_benchmarks.py --project $PROJECT -m 760m.yaml -g 8 --microbatch_size  2 --accum  2 --image $TORCH_2_IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb -t bf16 --cluster $CLUSTER_80GB -s 9 9 --RUN --fsdp_config_activation_checkpointing true --fsdp_config_shard_strategy "SHARD_GRAD_OP" --fsdp_config_limit_all_gathers true --fsdp_config_forward_prefetch false --fsdp_config_backward_prefetch BACKWARD_PRE
for SHARD_STRAT in "FULL_SHARD" "SHARD_GRAD_OP" "NO_SHARD"
do
    for FWD_PREFTCH in true false
    do
        for GATH_LMT in true false
        do
            for BACK_PREFTCH in BACKWARD_PRE BACKWARD_POST 
            do 
                for CPU_OFFLOAD in true false
                do
                    python submit_benchmarks.py --project $PROJECT -m 7b.yaml -g 8 16 --microbatch_size  2 --accum  2 --image $TORCH_2_IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb -t bf16 --cluster $CLUSTER_80GB -s 15 15 --RUN --fsdp_config_activation_checkpointing true --fsdp_config_shard_strategy $SHARD_STRAT --fsdp_config_limit_all_gathers $GATH_LMT --fsdp_config_forward_prefetch $FWD_PREFTCH --fsdp_config_backward_prefetch $BACK_PREFTCH --activation_cpu_offload $CPU_OFFLOAD
                    python submit_benchmarks.py --project $PROJECT -m 7b.yaml -g 8 16 --microbatch_size  32 --accum  2 --image $TORCH_2_IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb -t bf16 --cluster $CLUSTER_80GB -s 11 11 --RUN --fsdp_config_activation_checkpointing true --fsdp_config_shard_strategy $SHARD_STRAT --fsdp_config_limit_all_gathers $GATH_LMT --fsdp_config_forward_prefetch $FWD_PREFTCH --fsdp_config_backward_prefetch $BACK_PREFTCH --activation_cpu_offload $CPU_OFFLOAD
                    #3b test torch runs
                    python submit_benchmarks.py --project $PROJECT -m 3b.yaml -g 8 16 --microbatch_size  3 --accum  6 --image $TORCH_2_IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb -t bf16 --cluster $CLUSTER_80GB -s 15 15 --RUN --fsdp_config_activation_checkpointing true --fsdp_config_shard_strategy $SHARD_STRAT --fsdp_config_limit_all_gathers $GATH_LMT --fsdp_config_forward_prefetch $FWD_PREFTCH --fsdp_config_backward_prefetch $BACK_PREFTCH --activation_cpu_offload $CPU_OFFLOAD
                    python submit_benchmarks.py --project $PROJECT -m 3b.yaml -g 8 16 --microbatch_size  32 --accum  12 --image $TORCH_2_IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb -t bf16 --cluster $CLUSTER_80GB -s 11 11 --RUN --fsdp_config_activation_checkpointing false --fsdp_config_shard_strategy $SHARD_STRAT --fsdp_config_limit_all_gathers $GATH_LMT --fsdp_config_forward_prefetch $FWD_PREFTCH --fsdp_config_backward_prefetch $BACK_PREFTCH --activation_cpu_offload $CPU_OFFLOAD
                
                    #Try smaller models
                    python submit_benchmarks.py --project $PROJECT -m 350m.yaml -g 8 16 --microbatch_size  3 --accum  6 --image $TORCH_2_IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb -t bf16 --cluster $CLUSTER_80GB -s 15 15 --RUN --fsdp_config_activation_checkpointing true --fsdp_config_shard_strategy $SHARD_STRAT --fsdp_config_limit_all_gathers $GATH_LMT --fsdp_config_forward_prefetch $FWD_PREFTCH --fsdp_config_backward_prefetch $BACK_PREFTCH --activation_cpu_offload $CPU_OFFLOAD
                    python submit_benchmarks.py --project $PROJECT -m 350m.yaml -g 8 16 --microbatch_size  32 --accum  12 --image $TORCH_2_IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb -t bf16 --cluster $CLUSTER_80GB -s 11 11 --RUN --fsdp_config_activation_checkpointing false --fsdp_config_shard_strategy $SHARD_STRAT --fsdp_config_limit_all_gathers $GATH_LMT --fsdp_config_forward_prefetch $FWD_PREFTCH --fsdp_config_backward_prefetch $BACK_PREFTCH --activation_cpu_offload $CPU_OFFLOAD
                done
            done
        done
    done
done