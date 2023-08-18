#!/bin/bash

PROJECT="interntorch2"
GIT_COMMIT="v0.0.4"
IMAGE="mosaicml/pytorch:2.0.1_cu118-python3.10-ubuntu20.04"
CLUSTER_80GB=r1z1
CLUSTER_40GB=r8z3

# A100 40GB

# seqlen 2048
# python submit_benchmarks.py --project $PROJECT -m 125m.yaml -g 8 --microbatch_size  24 --accum  3 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 11 11 --RUN --data_remote oci://mosaicml-internal-dataset-c4/preconcat-gpt_neox/
# python submit_benchmarks.py --project $PROJECT -m 350m.yaml -g 8 --microbatch_size  16 --accum  4 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 11 11 --RUN --data_remote oci://mosaicml-internal-dataset-c4/preconcat-gpt_neox/ --fsdp_config_activation_checkpointing true
# python submit_benchmarks.py --project $PROJECT -m 760m.yaml -g 8 --microbatch_size  12 --accum  4 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 11 11 --RUN --data_remote oci://mosaicml-internal-dataset-c4/preconcat-gpt_neox/ --fsdp_config_activation_checkpointing true
# python submit_benchmarks.py --project $PROJECT -m   1b.yaml -g 8 --microbatch_size   8 --accum  8 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 11 11 --RUN --data_remote oci://mosaicml-internal-dataset-c4/preconcat-gpt_neox/ --fsdp_config_activation_checkpointing true

# python submit_benchmarks.py --project $PROJECT -m 125m.yaml -g 8 --microbatch_size  12 --accum  3 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 11 11 --RUN --data_remote oci://mosaicml-internal-dataset-c4/preconcat-gpt_neox/ --fsdp_config_activation_checkpointing false

# python submit_benchmarks.py --project $PROJECT -m   3b.yaml -g 8 --microbatch_size   5 --accum  8 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 11 11 --RUN --fsdp_config_activation_checkpointing false
# python submit_benchmarks.py --project $PROJECT -m   3b.yaml -g 8 --microbatch_size   8 --accum  8 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 11 11 --RUN --fsdp_config_activation_checkpointing false
# python submit_benchmarks.py --project $PROJECT -m   3b.yaml -g 8 --microbatch_size   12 --accum  8 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 11 11 --RUN --fsdp_config_activation_checkpointing false
# python submit_benchmarks.py --project $PROJECT -m   3b.yaml -g 8 --microbatch_size   16 --accum  8 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 11 11 --RUN --fsdp_config_activation_checkpointing false

# python submit_benchmarks.py --project $PROJECT -m   3b.yaml -g 8 --microbatch_size   5 --accum  8 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 11 11 --RUN --fsdp_config_activation_checkpointing true
# python submit_benchmarks.py --project $PROJECT -m   3b.yaml -g 8 --microbatch_size   8 --accum  8 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 11 11 --RUN --fsdp_config_activation_checkpointing true
# python submit_benchmarks.py --project $PROJECT -m   3b.yaml -g 8 --microbatch_size   12 --accum  8 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 11 11 --RUN --fsdp_config_activation_checkpointing true
# python submit_benchmarks.py --project $PROJECT -m   3b.yaml -g 8 --microbatch_size   16 --accum  8 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 11 11 --RUN --fsdp_config_activation_checkpointing true

# python submit_benchmarks.py --project $PROJECT -m   7b.yaml -g 8 --microbatch_size  16 --accum  4 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 11 11 --RUN --data_remote oci://mosaicml-internal-dataset-c4/preconcat-gpt_neox/ --fsdp_config_activation_checkpointing false
# python submit_benchmarks.py --project $PROJECT -m   7b.yaml -g 8 --microbatch_size  12 --accum  4 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 11 11 --RUN --data_remote oci://mosaicml-internal-dataset-c4/preconcat-gpt_neox/

# Replicate/understand diffs using streaming data loader
# python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g 8 --microbatch_size 1 --accum 16 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 13 13 --RUN --data_remote oci://mosaicml-internal-dataset-c4/preconcat-gpt_neox/ --fsdp_config_activation_checkpointing true
# python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g 8 --microbatch_size 2 --accum 16 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 12 12 --RUN --data_remote oci://mosaicml-internal-dataset-c4/preconcat-gpt_neox/ --fsdp_config_activation_checkpointing true
# python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g 8 --microbatch_size 4 --accum 16 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 11 11 --RUN --data_remote oci://mosaicml-internal-dataset-c4/preconcat-gpt_neox/ --fsdp_config_activation_checkpointing true
# python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g 8 --microbatch_size 8 --accum 16 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 10 10 --RUN --data_remote oci://mosaicml-internal-dataset-c4/preconcat-gpt_neox/ --fsdp_config_activation_checkpointing true
# python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g 8 --microbatch_size 16 --accum 16 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 9 9 --RUN --data_remote oci://mosaicml-internal-dataset-c4/preconcat-gpt_neox/ --fsdp_config_activation_checkpointing true
# python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g 16 --microbatch_size 10 --accum 16 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 11 11 --RUN --data_remote oci://mosaicml-internal-dataset-c4/preconcat-gpt_neox/ --fsdp_config_activation_checkpointing true

# Test ack_ckpt differences
# python submit_benchmarks.py --project $PROJECT -m  13b.yaml --seq_len 8192 -g 8 --microbatch_size 1 --accum 16 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 11 11 --RUN --data_remote oci://mosaicml-internal-dataset-c4/preconcat-gpt_neox/ --fsdp_config_activation_checkpointing false
# python submit_benchmarks.py --project $PROJECT -m  13b.yaml --seq_len 4096 -g 8 --microbatch_size 2 --accum 16 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 11 11 --RUN --data_remote oci://mosaicml-internal-dataset-c4/preconcat-gpt_neox/ --fsdp_config_activation_checkpointing false
# python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g 8 --microbatch_size 4 --accum 16 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 11 11 --RUN --data_remote oci://mosaicml-internal-dataset-c4/preconcat-gpt_neox/ --fsdp_config_activation_checkpointing false
# python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g 8 --microbatch_size 3 --accum 16 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 11 11 --RUN --data_remote oci://mosaicml-internal-dataset-c4/preconcat-gpt_neox/ --fsdp_config_activation_checkpointing false
# python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g 8 --microbatch_size 2 --accum 16 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 11 11 --RUN --data_remote oci://mosaicml-internal-dataset-c4/preconcat-gpt_neox/ --fsdp_config_activation_checkpointing false
# python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g 8 --microbatch_size 1 --accum 16 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 11 11 --RUN --data_remote oci://mosaicml-internal-dataset-c4/preconcat-gpt_neox/ --fsdp_config_activation_checkpointing false

# python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g 8 --microbatch_size 8 --accum 16 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 10 10 --RUN --data_remote oci://mosaicml-internal-dataset-c4/preconcat-gpt_neox/ --fsdp_config_activation_checkpointing false
# python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g 8 --microbatch_size 6 --accum 16 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 10 10 --RUN --data_remote oci://mosaicml-internal-dataset-c4/preconcat-gpt_neox/ --fsdp_config_activation_checkpointing false
# python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g 8 --microbatch_size 4 --accum 16 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 10 10 --RUN --data_remote oci://mosaicml-internal-dataset-c4/preconcat-gpt_neox/ --fsdp_config_activation_checkpointing false
# python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g 8 --microbatch_size 2 --accum 16 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 10 10 --RUN --data_remote oci://mosaicml-internal-dataset-c4/preconcat-gpt_neox/ --fsdp_config_activation_checkpointing false
# python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g 8 --microbatch_size 1 --accum 16 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 10 10 --RUN --data_remote oci://mosaicml-internal-dataset-c4/preconcat-gpt_neox/ --fsdp_config_activation_checkpointing false # PASSED
# python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g 8 --microbatch_size 1 --accum 16 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 10 10 --RUN --data_remote oci://mosaicml-internal-dataset-c4/preconcat-gpt_neox/  --fsdp_config_activation_checkpointing false # PASSED
# python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g 8 --microbatch_size 1 --accum 4 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 10 10 --RUN --data_remote oci://mosaicml-internal-dataset-c4/preconcat-gpt_neox/ --fsdp_config_activation_checkpointing false # PASSED
# python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g 8 --microbatch_size 2 --accum 4 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 10 10 --RUN --data_remote oci://mosaicml-internal-dataset-c4/preconcat-gpt_neox/ --fsdp_config_activation_checkpointing false
# python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g 8 --microbatch_size 4 --accum 4 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 10 10 --RUN --data_remote oci://mosaicml-internal-dataset-c4/preconcat-gpt_neox/ --fsdp_config_activation_checkpointing false


# python submit_benchmarks.py --project $PROJECT -m  7b.yaml -g 8 --microbatch_size 1 --accum 4 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 10 10 --RUN --data_remote oci://mosaicml-internal-dataset-c4/preconcat-gpt_neox/ --fsdp_config_activation_checkpointing false
# python submit_benchmarks.py --project $PROJECT -m  7b.yaml -g 8 --microbatch_size 2 --accum 4 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 10 10 --RUN --data_remote oci://mosaicml-internal-dataset-c4/preconcat-gpt_neox/ --fsdp_config_activation_checkpointing false
# python submit_benchmarks.py --project $PROJECT -m  7b.yaml -g 8 --microbatch_size 4 --accum 4 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 10 10 --RUN --data_remote oci://mosaicml-internal-dataset-c4/preconcat-gpt_neox/ --fsdp_config_activation_checkpointing false
# python submit_benchmarks.py --project $PROJECT -m  7b.yaml -g 8 --microbatch_size 8 --accum 4 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 10 10 --RUN --data_remote oci://mosaicml-internal-dataset-c4/preconcat-gpt_neox/ --fsdp_config_activation_checkpointing false

# python submit_benchmarks.py --project $PROJECT -m  7b.yaml -g 8 --microbatch_size 1 --accum 16 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 10 10 --RUN --data_remote oci://mosaicml-internal-dataset-c4/preconcat-gpt_neox/ --fsdp_config_activation_checkpointing false
# python submit_benchmarks.py --project $PROJECT -m  7b.yaml -g 8 --microbatch_size 2 --accum 16 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 10 10 --RUN --data_remote oci://mosaicml-internal-dataset-c4/preconcat-gpt_neox/ --fsdp_config_activation_checkpointing false
# python submit_benchmarks.py --project $PROJECT -m  7b.yaml -g 8 --microbatch_size 4 --accum 16 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 10 10 --RUN --data_remote oci://mosaicml-internal-dataset-c4/preconcat-gpt_neox/ --fsdp_config_activation_checkpointing false
# python submit_benchmarks.py --project $PROJECT -m  7b.yaml -g 8 --microbatch_size 8 --accum 16 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 10 10 --RUN --data_remote oci://mosaicml-internal-dataset-c4/preconcat-gpt_neox/ --fsdp_config_activation_checkpointing false

# NOTE: Tried the commented ones last night, OOM'd
# python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g 8 --microbatch_size 14 --accum 16 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 9 9 --RUN --data_remote oci://mosaicml-internal-dataset-c4/preconcat-gpt_neox/ --fsdp_config_activation_checkpointing false
# python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g 8 --microbatch_size 12 --accum 16 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 9 9 --RUN --data_remote oci://mosaicml-internal-dataset-c4/preconcat-gpt_neox/ --fsdp_config_activation_checkpointing false
# python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g 8 --microbatch_size 10 --accum 16 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 9 9 --RUN --data_remote oci://mosaicml-internal-dataset-c4/preconcat-gpt_neox/ --fsdp_config_activation_checkpointing false
# python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g 8 --microbatch_size 8 --accum 16 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 9 9 --RUN --data_remote oci://mosaicml-internal-dataset-c4/preconcat-gpt_neox/ --fsdp_config_activation_checkpointing false
# python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g 8 --microbatch_size 4 --accum 16 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 9 9 --RUN --data_remote oci://mosaicml-internal-dataset-c4/preconcat-gpt_neox/ --fsdp_config_activation_checkpointing false
# python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g 8 --microbatch_size 4 --accum 8 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 9 9 --RUN --data_remote oci://mosaicml-internal-dataset-c4/preconcat-gpt_neox/ --fsdp_config_activation_checkpointing false
# python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g 8 --microbatch_size 4 --accum 4 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 9 9 --RUN --data_remote oci://mosaicml-internal-dataset-c4/preconcat-gpt_neox/ --fsdp_config_activation_checkpointing false
# python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g 8 --microbatch_size 2 --accum 16 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 9 9 --RUN --data_remote oci://mosaicml-internal-dataset-c4/preconcat-gpt_neox/ --fsdp_config_activation_checkpointing false # PASSED
# python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g 8 --microbatch_size 8 --accum 16 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 7 7 --RUN --data_remote oci://mosaicml-internal-dataset-c4/preconcat-gpt_neox/ --fsdp_config_activation_checkpointing false #PASSED

# Test torch.compile
# python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g 8 --microbatch_size 1 --accum 16 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 13 12 --RUN --data_remote oci://mosaicml-internal-dataset-c4/preconcat-gpt_neox/ --torch_compile true
# python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g 8 --microbatch_size 2 --accum 16 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 12 12 --RUN --data_remote oci://mosaicml-internal-dataset-c4/preconcat-gpt_neox/
# python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g 8 --microbatch_size 4 --accum 16 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 11 11 --RUN --data_remote oci://mosaicml-internal-dataset-c4/preconcat-gpt_neox/
# python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g 8 --microbatch_size 8 --accum 16 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 10 10 --RUN --data_remote oci://mosaicml-internal-dataset-c4/preconcat-gpt_neox/
# python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g 8 --microbatch_size 16 --accum 16 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 9 9 --RUN --data_remote oci://mosaicml-internal-dataset-c4/preconcat-gpt_neox/
# python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g 16 --microbatch_size 10 --accum 16 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 11 11 --RUN --data_remote oci://mosaicml-internal-dataset-c4/preconcat-gpt_neox/

# # INCREASE GPU COUNT
# python submit_benchmarks.py --project $PROJECT -m 125m.yaml -g 16 32 64 128 --microbatch_size  26 --accum  1 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 11 11 --RUN
# python submit_benchmarks.py --project $PROJECT -m 350m.yaml -g 16 32 64 128 --microbatch_size  18 --accum  1 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 11 11 --RUN
# python submit_benchmarks.py --project $PROJECT -m 760m.yaml -g 16 32 64 128 --microbatch_size  12 --accum  1 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 11 11 --RUN
# python submit_benchmarks.py --project $PROJECT -m   1b.yaml -g 16           --microbatch_size   8 --accum  1 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 11 11 --RUN
# python submit_benchmarks.py --project $PROJECT -m   3b.yaml -g 16           --microbatch_size   5 --accum  1 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 11 11 --RUN --fsdp_config_activation_checkpointing false
# python submit_benchmarks.py --project $PROJECT -m   7b.yaml -g 16           --microbatch_size  16 --accum  1 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 11 11 --RUN
# python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g 16           --microbatch_size  10 --accum  1 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 11 11 --RUN
# python submit_benchmarks.py --project $PROJECT -m   1b.yaml -g    32 64 128 --microbatch_size  10 --accum  1 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 11 11 --RUN
# python submit_benchmarks.py --project $PROJECT -m   3b.yaml -g    32 64 128 --microbatch_size   6 --accum  1 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 11 11 --RUN --fsdp_config_activation_checkpointing false
# python submit_benchmarks.py --project $PROJECT -m   7b.yaml -g    32 64 128 --microbatch_size  18 --accum  1 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 11 11 --RUN
# python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g    32        --microbatch_size  14 --accum  1 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 11 11 --RUN
# python submit_benchmarks.py --project $PROJECT -m  30b.yaml -g    32        --microbatch_size   4 --accum  1 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 11 11 --RUN
# python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g       64 128 --microbatch_size  16 --accum  1 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 11 11 --RUN
# python submit_benchmarks.py --project $PROJECT -m  70b.yaml -g       64     --microbatch_size   2 --accum  1 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 11 11 --RUN
# python submit_benchmarks.py --project $PROJECT -m  30b.yaml -g          128 --microbatch_size   6 --accum  1 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 11 11 --RUN
# python submit_benchmarks.py --project $PROJECT -m  70b.yaml -g          128 --microbatch_size   4 --accum  1 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 11 11 --RUN

# # SCALE SEQUENCE LENGTH
# # seqlen 512
# python submit_benchmarks.py --project $PROJECT -m 125m.yaml -g 8 --microbatch_size 104 --accum  3 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s  9  9 --RUN
# python submit_benchmarks.py --project $PROJECT -m 350m.yaml -g 8 --microbatch_size  64 --accum  4 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s  9  9 --RUN
# python submit_benchmarks.py --project $PROJECT -m 760m.yaml -g 8 --microbatch_size  48 --accum  4 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s  9  9 --RUN
# python submit_benchmarks.py --project $PROJECT -m   1b.yaml -g 8 --microbatch_size  32 --accum  8 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s  9  9 --RUN
# python submit_benchmarks.py --project $PROJECT -m   3b.yaml -g 8 --microbatch_size  20 --accum  8 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s  9  9 --RUN --fsdp_config_activation_checkpointing false
# python submit_benchmarks.py --project $PROJECT -m   7b.yaml -g 8 --microbatch_size  56 --accum  4 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s  9  9 --RUN
# python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g 8 --microbatch_size  16 --accum 16 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s  9  9 --RUN
# # seqlen 1024
# python submit_benchmarks.py --project $PROJECT -m 125m.yaml -g 8 --microbatch_size  52 --accum  3 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 10 10 --RUN
# python submit_benchmarks.py --project $PROJECT -m 350m.yaml -g 8 --microbatch_size  32 --accum  4 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 10 10 --RUN
# python submit_benchmarks.py --project $PROJECT -m 760m.yaml -g 8 --microbatch_size  24 --accum  4 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 10 10 --RUN
# python submit_benchmarks.py --project $PROJECT -m   1b.yaml -g 8 --microbatch_size  16 --accum  8 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 10 10 --RUN
# python submit_benchmarks.py --project $PROJECT -m   3b.yaml -g 8 --microbatch_size  10 --accum  8 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 10 10 --RUN --fsdp_config_activation_checkpointing false
# python submit_benchmarks.py --project $PROJECT -m   7b.yaml -g 8 --microbatch_size  28 --accum  4 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 10 10 --RUN
# python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g 8 --microbatch_size   8 --accum 16 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 10 10 --RUN
# # seqlen 4096
# python submit_benchmarks.py --project $PROJECT -m 125m.yaml -g 8 --microbatch_size  13 --accum  3 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 12 12 --RUN
# python submit_benchmarks.py --project $PROJECT -m 350m.yaml -g 8 --microbatch_size   8 --accum  4 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 12 12 --RUN
# python submit_benchmarks.py --project $PROJECT -m 760m.yaml -g 8 --microbatch_size   6 --accum  4 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 12 12 --RUN
# python submit_benchmarks.py --project $PROJECT -m   1b.yaml -g 8 --microbatch_size   4 --accum  8 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 12 12 --RUN
# python submit_benchmarks.py --project $PROJECT -m   3b.yaml -g 8 --microbatch_size   2 --accum  8 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 12 12 --RUN --fsdp_config_activation_checkpointing false
# python submit_benchmarks.py --project $PROJECT -m   7b.yaml -g 8 --microbatch_size   8 --accum  4 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 12 12 --RUN
# python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g 8 --microbatch_size   2 --accum 16 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 12 12 --RUN
# # seqlen 8192
# python submit_benchmarks.py --project $PROJECT -m 125m.yaml -g 8 --microbatch_size   5 --accum  3 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 13 13 --RUN
# python submit_benchmarks.py --project $PROJECT -m 350m.yaml -g 8 --microbatch_size   4 --accum  4 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 13 13 --RUN
# python submit_benchmarks.py --project $PROJECT -m 760m.yaml -g 8 --microbatch_size   3 --accum  4 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 13 13 --RUN
# python submit_benchmarks.py --project $PROJECT -m   1b.yaml -g 8 --microbatch_size   2 --accum  8 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 13 13 --RUN
# python submit_benchmarks.py --project $PROJECT -m   3b.yaml -g 8 --microbatch_size   1 --accum  8 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 13 13 --RUN --fsdp_config_activation_checkpointing false
# python submit_benchmarks.py --project $PROJECT -m   7b.yaml -g 8 --microbatch_size   3 --accum  4 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 13 13 --RUN
# python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g 8 --microbatch_size   1 --accum 16 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 13 13 --RUN
# # seqlen 16384
# python submit_benchmarks.py --project $PROJECT -m 125m.yaml -g 8 --microbatch_size   2 --accum  3 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 14 14 --RUN
# python submit_benchmarks.py --project $PROJECT -m 350m.yaml -g 8 --microbatch_size   2 --accum  4 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 14 14 --RUN
# python submit_benchmarks.py --project $PROJECT -m 760m.yaml -g 8 --microbatch_size   1 --accum  4 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 14 14 --RUN
# python submit_benchmarks.py --project $PROJECT -m   1b.yaml -g 8 --microbatch_size   1 --accum  8 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 14 14 --RUN
# python submit_benchmarks.py --project $PROJECT -m   3b.yaml -g 8 --microbatch_size   2 --accum  8 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 14 14 --RUN
# python submit_benchmarks.py --project $PROJECT -m   7b.yaml -g 8 --microbatch_size   1 --accum  4 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 14 14 --RUN
# # seqlen 32768
# python submit_benchmarks.py --project $PROJECT -m 125m.yaml -g 8 --microbatch_size   1 --accum  4 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 15 15 --RUN
# python submit_benchmarks.py --project $PROJECT -m 350m.yaml -g 8 --microbatch_size   1 --accum  4 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 15 15 --RUN
# python submit_benchmarks.py --project $PROJECT -m 760m.yaml -g 8 --microbatch_size   1 --accum  4 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 15 15 --RUN --fsdp_config_activation_checkpointing true
# python submit_benchmarks.py --project $PROJECT -m   1b.yaml -g 8 --microbatch_size   1 --accum  4 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 15 15 --RUN --fsdp_config_activation_checkpointing true
# python submit_benchmarks.py --project $PROJECT -m   3b.yaml -g 8 --microbatch_size   1 --accum  4 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 15 15 --RUN
