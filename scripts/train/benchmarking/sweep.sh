#!/bin/bash

PROJECT="tput"
GIT_COMMIT="v0.0.4"
IMAGE="mosaicml/pytorch:2.1.0_cu121-python3.10-ubuntu20.04"
CLUSTER_40GB= # TODO

for PRECISION in fp8 bf16
do

    # H100 80GB
    python submit_benchmarks.py --project $PROJECT -m 125m.yaml -g 8 --microbatch_size  40 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 11 11 --RUN -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m 350m.yaml -g 8 --microbatch_size  32 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 11 11 --RUN -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m 760m.yaml -g 8 --microbatch_size  24 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 11 11 --RUN -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m   1b.yaml -g 8 --microbatch_size  14 --accum  1 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 11 11 --RUN -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m   3b.yaml -g 8 --microbatch_size  10 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 11 11 --RUN --fsdp_config_activation_checkpointing false -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m   7b.yaml -g 8 --microbatch_size  6 --accum  1 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 11 11 --RUN --fsdp_config_activation_checkpointing false -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g 8 --microbatch_size  2 --accum  1 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 11 11 --RUN --fsdp_config_activation_checkpointing false -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m  30b.yaml -g 8 --microbatch_size   3 --accum 21 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 11 11 --RUN -t ${PRECISION}

    python submit_benchmarks.py --project $PROJECT -m   7b.yaml -g 8 --microbatch_size  7 --accum  1 --image $IMAGE1 --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 11 11 --RUN --fsdp_config_activation_checkpointing false -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m   7b.yaml -g 8 --microbatch_size  7 --accum  1 --image $IMAGE0 --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 11 11 --RUN --fsdp_config_activation_checkpointing false -t ${PRECISION}

    python submit_benchmarks.py --project $PROJECT -m   7b.yaml -g 8 --microbatch_size  6 --accum  1 --image $IMAGE1 --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 11 11 --RUN --fsdp_config_activation_checkpointing false -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m   7b.yaml -g 8 --microbatch_size  6 --accum  1 --image $IMAGE0 --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 11 11 --RUN --fsdp_config_activation_checkpointing false -t ${PRECISION}

    # INCREASE GPU COUNT
    for GPU_NUM in 16 32 64
    do
        python submit_benchmarks.py --project $PROJECT -m 125m.yaml -g $GPU_NUM --microbatch_size 32 --accum  1 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 11 11 --RUN -t ${PRECISION}
        python submit_benchmarks.py --project $PROJECT -m 350m.yaml -g $GPU_NUM --microbatch_size 32 --accum  1 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 11 11 --RUN -t ${PRECISION}
        python submit_benchmarks.py --project $PROJECT -m 760m.yaml -g $GPU_NUM --microbatch_size 24 --accum  1 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 11 11 --RUN -t ${PRECISION}
        python submit_benchmarks.py --project $PROJECT -m   1b.yaml -g $GPU_NUM --microbatch_size 20 --accum  1 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 11 11 --RUN -t ${PRECISION}
        python submit_benchmarks.py --project $PROJECT -m   7b.yaml -g $GPU_NUM --microbatch_size 32 --accum  1 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 11 11 --RUN --fsdp_config_activation_checkpointing false -t ${PRECISION}
    done

    python submit_benchmarks.py --project $PROJECT -m   3b.yaml -g 16       --microbatch_size 10 --accum  3 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 11 11 --RUN --fsdp_config_activation_checkpointing false -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g 16       --microbatch_size 2 --accum  1 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 11 11 --RUN --fsdp_config_activation_checkpointing false -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m  30b.yaml -g 16       --microbatch_size 10 --accum  3 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 11 11 --RUN -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m   3b.yaml -g    32    --microbatch_size 6 --accum  3 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 11 11 --RUN --fsdp_config_activation_checkpointing false -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g    32    --microbatch_size 2 --accum  1 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 11 11 --RUN --fsdp_config_activation_checkpointing false -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m   3b.yaml -g       64 --microbatch_size 6 --accum  3 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 11 11 --RUN --fsdp_config_activation_checkpointing false -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g       64 --microbatch_size 2 --accum  1 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 11 11 --RUN --fsdp_config_activation_checkpointing false -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m  30b.yaml -g    32    --microbatch_size 14 --accum  3 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 11 11 --RUN -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m  70b.yaml -g    32    --microbatch_size  2 --accum 16 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 11 11 --RUN -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m  30b.yaml -g       64 --microbatch_size 16 --accum  3 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 11 11 --RUN -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m  70b.yaml -g       64 --microbatch_size  8 --accum  4 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 11 11 --RUN -t ${PRECISION}

    # SCALE SEQUENCE LENGTH
    # seqlen 512
    python submit_benchmarks.py --project $PROJECT -m 125m.yaml -g 8 --microbatch_size 128 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --precision fp8 --gpu_type h100_80gb --cluster $CLUSTER_H100 -s  9  9 --RUN -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m 350m.yaml -g 8 --microbatch_size 128 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s  9  9 --RUN -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m 760m.yaml -g 8 --microbatch_size  96 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s  9  9 --RUN -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m   1b.yaml -g 8 --microbatch_size  56 --accum  4 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s  9  9 --RUN -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m   3b.yaml -g 8 --microbatch_size  40 --accum  6 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s  9  9 --RUN --fsdp_config_activation_checkpointing false -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m   7b.yaml -g 8 --microbatch_size 64 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s  9  9 --RUN --fsdp_config_activation_checkpointing false -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g 8 --microbatch_size  20 --accum  3 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s  9  9 --RUN --fsdp_config_activation_checkpointing false -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m  30b.yaml -g 8 --microbatch_size  12 --accum 21 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s  9  9 --RUN -t ${PRECISION}
    # seqlen 1024
    python submit_benchmarks.py --project $PROJECT -m 125m.yaml -g 8 --microbatch_size  64 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 10 10 --RUN -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m 350m.yaml -g 8 --microbatch_size  64 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 10 10 --RUN -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m 760m.yaml -g 8 --microbatch_size  48 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 10 10 --RUN -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m   1b.yaml -g 8 --microbatch_size  18 --accum  4 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 10 10 --RUN -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m   3b.yaml -g 8 --microbatch_size  20 --accum  6 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 10 10 --RUN --fsdp_config_activation_checkpointing false -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m   7b.yaml -g 8 --microbatch_size  64 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 10 10 --RUN -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g 8 --microbatch_size  40 --accum  3 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 10 10 --RUN -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m  30b.yaml -g 8 --microbatch_size   6 --accum 21 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 10 10 --RUN -t ${PRECISION}
    # seqlen 4096
    python submit_benchmarks.py --project $PROJECT -m 125m.yaml -g 8 --microbatch_size  16 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 12 12 --RUN -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m 350m.yaml -g 8 --microbatch_size  16 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 12 12 --RUN -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m 760m.yaml -g 8 --microbatch_size  12 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 12 12 --RUN -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m   1b.yaml -g 8 --microbatch_size   7 --accum  4 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 12 12 --RUN -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m   3b.yaml -g 8 --microbatch_size   5 --accum  6 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 12 12 --RUN --fsdp_config_activation_checkpointing false -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m   7b.yaml -g 8 --microbatch_size  16 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 12 12 --RUN -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g 8 --microbatch_size  10 --accum  3 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 12 12 --RUN -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m  30b.yaml -g 8 --microbatch_size   1 --accum 21 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 12 12 --RUN -t ${PRECISION}
    # seqlen 8192
    python submit_benchmarks.py --project $PROJECT -m 125m.yaml -g 8 --microbatch_size   8 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 13 13 --RUN -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m 350m.yaml -g 8 --microbatch_size   8 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 13 13 --RUN -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m 760m.yaml -g 8 --microbatch_size   6 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 13 13 --RUN -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m   1b.yaml -g 8 --microbatch_size   3 --accum  4 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 13 13 --RUN -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m   3b.yaml -g 8 --microbatch_size   3 --accum  6 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 13 13 --RUN --fsdp_config_activation_checkpointing false -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m   7b.yaml -g 8 --microbatch_size   8 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 13 13 --RUN -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g 8 --microbatch_size   5 --accum  3 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 13 13 --RUN -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m  30b.yaml -g 8 --microbatch_size   2 --accum 1 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 13 13 --RUN -t ${PRECISION}
    # seqlen 16384
    python submit_benchmarks.py --project $PROJECT -m 125m.yaml -g 8 --microbatch_size   4 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 14 14 --RUN -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m 350m.yaml -g 8 --microbatch_size   4 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 14 14 --RUN -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m 760m.yaml -g 8 --microbatch_size   3 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 14 14 --RUN -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m   1b.yaml -g 8 --microbatch_size   2 --accum  4 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 14 14 --RUN -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m   3b.yaml -g 8 --microbatch_size   1 --accum  6 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 14 14 --RUN --fsdp_config_activation_checkpointing false -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m   7b.yaml -g 8 --microbatch_size   4 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 14 14 --RUN -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g 8 --microbatch_size   3 --accum  3 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 14 14 --RUN -t ${PRECISION}
    # seqlen 32768
    python submit_benchmarks.py --project $PROJECT -m 125m.yaml -g 8 --microbatch_size   2 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 15 15 --RUN -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m 350m.yaml -g 8 --microbatch_size   2 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 15 15 --RUN -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m 760m.yaml -g 8 --microbatch_size   1 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 15 15 --RUN -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m   1b.yaml -g 8 --microbatch_size   1 --accum  4 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 15 15 --RUN -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m   3b.yaml -g 8 --microbatch_size   3 --accum  6 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 15 15 --RUN -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m   7b.yaml -g 8 --microbatch_size   2 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 15 15 --RUN -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g 8 --microbatch_size   1 --accum  3 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 15 15 --RUN -t ${PRECISION}
    # seqlen 65536
    python submit_benchmarks.py --project $PROJECT -m 125m.yaml -g 8 --microbatch_size   1 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 16 16 --RUN -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m 350m.yaml -g 8 --microbatch_size   1 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 16 16 --RUN -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m 760m.yaml -g 8 --microbatch_size   1 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 16 16 --RUN --fsdp_config_activation_checkpointing true -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m   1b.yaml -g 8 --microbatch_size   1 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 16 16 --RUN --fsdp_config_activation_checkpointing true -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m   3b.yaml -g 8 --microbatch_size   1 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 16 16 --RUN -t ${PRECISION}
    python submit_benchmarks.py --project $PROJECT -m   7b.yaml -g 8 --microbatch_size   1 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_H100 -s 16 16 --RUN -t ${PRECISION}
done


# A100 80GB

# seqlen 2048
python submit_benchmarks.py --project $PROJECT -m 125m.yaml -g 8 --microbatch_size  40 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 11 11 --RUN
python submit_benchmarks.py --project $PROJECT -m 350m.yaml -g 8 --microbatch_size  32 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 11 11 --RUN
python submit_benchmarks.py --project $PROJECT -m 760m.yaml -g 8 --microbatch_size  24 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 11 11 --RUN
python submit_benchmarks.py --project $PROJECT -m   1b.yaml -g 8 --microbatch_size  14 --accum  1 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 11 11 --RUN
python submit_benchmarks.py --project $PROJECT -m   3b.yaml -g 8 --microbatch_size  10 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 11 11 --RUN --fsdp_config_activation_checkpointing false
python submit_benchmarks.py --project $PROJECT -m   7b.yaml -g 8 --microbatch_size  6 --accum  1 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 11 11 --RUN --fsdp_config_activation_checkpointing false
python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g 8 --microbatch_size  2 --accum  1 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 11 11 --RUN --fsdp_config_activation_checkpointing false
python submit_benchmarks.py --project $PROJECT -m  30b.yaml -g 8 --microbatch_size   3 --accum 21 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 11 11 --RUN

python submit_benchmarks.py --project $PROJECT -m   7b.yaml -g 8 --microbatch_size  7 --accum  1 --image $IMAGE1 --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 11 11 --RUN --fsdp_config_activation_checkpointing false
python submit_benchmarks.py --project $PROJECT -m   7b.yaml -g 8 --microbatch_size  7 --accum  1 --image $IMAGE0 --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 11 11 --RUN --fsdp_config_activation_checkpointing false

python submit_benchmarks.py --project $PROJECT -m   7b.yaml -g 8 --microbatch_size  6 --accum  1 --image $IMAGE1 --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 11 11 --RUN --fsdp_config_activation_checkpointing false
python submit_benchmarks.py --project $PROJECT -m   7b.yaml -g 8 --microbatch_size  6 --accum  1 --image $IMAGE0 --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 11 11 --RUN --fsdp_config_activation_checkpointing false

# INCREASE GPU COUNT
for GPU_NUM in 16 32 64
do
    python submit_benchmarks.py --project $PROJECT -m 125m.yaml -g $GPU_NUM --microbatch_size 32 --accum  1 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 11 11 --RUN
    python submit_benchmarks.py --project $PROJECT -m 350m.yaml -g $GPU_NUM --microbatch_size 32 --accum  1 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 11 11 --RUN
    python submit_benchmarks.py --project $PROJECT -m 760m.yaml -g $GPU_NUM --microbatch_size 24 --accum  1 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 11 11 --RUN
    python submit_benchmarks.py --project $PROJECT -m   1b.yaml -g $GPU_NUM --microbatch_size 20 --accum  1 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 11 11 --RUN
    python submit_benchmarks.py --project $PROJECT -m   7b.yaml -g $GPU_NUM --microbatch_size 32 --accum  1 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 11 11 --RUN --fsdp_config_activation_checkpointing false
done

python submit_benchmarks.py --project $PROJECT -m   3b.yaml -g 16       --microbatch_size 10 --accum  3 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 11 11 --RUN --fsdp_config_activation_checkpointing false
python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g 16       --microbatch_size 2 --accum  1 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 11 11 --RUN --fsdp_config_activation_checkpointing false
python submit_benchmarks.py --project $PROJECT -m  30b.yaml -g 16       --microbatch_size 10 --accum  3 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 11 11 --RUN
python submit_benchmarks.py --project $PROJECT -m   3b.yaml -g    32    --microbatch_size 6 --accum  3 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 11 11 --RUN --fsdp_config_activation_checkpointing false
python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g    32    --microbatch_size 2 --accum  1 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 11 11 --RUN --fsdp_config_activation_checkpointing false
python submit_benchmarks.py --project $PROJECT -m   3b.yaml -g       64 --microbatch_size 6 --accum  3 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 11 11 --RUN --fsdp_config_activation_checkpointing false
python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g       64 --microbatch_size 2 --accum  1 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 11 11 --RUN --fsdp_config_activation_checkpointing false
python submit_benchmarks.py --project $PROJECT -m  30b.yaml -g    32    --microbatch_size 14 --accum  3 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 11 11 --RUN
python submit_benchmarks.py --project $PROJECT -m  70b.yaml -g    32    --microbatch_size  2 --accum 16 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 11 11 --RUN
python submit_benchmarks.py --project $PROJECT -m  30b.yaml -g       64 --microbatch_size 16 --accum  3 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 11 11 --RUN
python submit_benchmarks.py --project $PROJECT -m  70b.yaml -g       64 --microbatch_size  8 --accum  4 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 11 11 --RUN

# SCALE SEQUENCE LENGTH
# seqlen 512
python submit_benchmarks.py --project $PROJECT -m 125m.yaml -g 8 --microbatch_size 128 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --precision fp8 --gpu_type a100_80gb --cluster $CLUSTER_80GB -s  9  9 --RUN
python submit_benchmarks.py --project $PROJECT -m 350m.yaml -g 8 --microbatch_size 128 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s  9  9 --RUN
python submit_benchmarks.py --project $PROJECT -m 760m.yaml -g 8 --microbatch_size  96 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s  9  9 --RUN
python submit_benchmarks.py --project $PROJECT -m   1b.yaml -g 8 --microbatch_size  56 --accum  4 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s  9  9 --RUN
python submit_benchmarks.py --project $PROJECT -m   3b.yaml -g 8 --microbatch_size  40 --accum  6 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s  9  9 --RUN --fsdp_config_activation_checkpointing false
python submit_benchmarks.py --project $PROJECT -m   7b.yaml -g 8 --microbatch_size 64 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s  9  9 --RUN --fsdp_config_activation_checkpointing false
python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g 8 --microbatch_size  20 --accum  3 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s  9  9 --RUN --fsdp_config_activation_checkpointing false
python submit_benchmarks.py --project $PROJECT -m  30b.yaml -g 8 --microbatch_size  12 --accum 21 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s  9  9 --RUN
# seqlen 1024
python submit_benchmarks.py --project $PROJECT -m 125m.yaml -g 8 --microbatch_size  64 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 10 10 --RUN
python submit_benchmarks.py --project $PROJECT -m 350m.yaml -g 8 --microbatch_size  64 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 10 10 --RUN
python submit_benchmarks.py --project $PROJECT -m 760m.yaml -g 8 --microbatch_size  48 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 10 10 --RUN
python submit_benchmarks.py --project $PROJECT -m   1b.yaml -g 8 --microbatch_size  18 --accum  4 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 10 10 --RUN
python submit_benchmarks.py --project $PROJECT -m   3b.yaml -g 8 --microbatch_size  20 --accum  6 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 10 10 --RUN --fsdp_config_activation_checkpointing false
python submit_benchmarks.py --project $PROJECT -m   7b.yaml -g 8 --microbatch_size  64 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 10 10 --RUN
python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g 8 --microbatch_size  40 --accum  3 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 10 10 --RUN
python submit_benchmarks.py --project $PROJECT -m  30b.yaml -g 8 --microbatch_size   6 --accum 21 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 10 10 --RUN
# seqlen 4096
python submit_benchmarks.py --project $PROJECT -m 125m.yaml -g 8 --microbatch_size  16 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 12 12 --RUN
python submit_benchmarks.py --project $PROJECT -m 350m.yaml -g 8 --microbatch_size  16 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 12 12 --RUN
python submit_benchmarks.py --project $PROJECT -m 760m.yaml -g 8 --microbatch_size  12 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 12 12 --RUN
python submit_benchmarks.py --project $PROJECT -m   1b.yaml -g 8 --microbatch_size   7 --accum  4 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 12 12 --RUN
python submit_benchmarks.py --project $PROJECT -m   3b.yaml -g 8 --microbatch_size   5 --accum  6 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 12 12 --RUN --fsdp_config_activation_checkpointing false
python submit_benchmarks.py --project $PROJECT -m   7b.yaml -g 8 --microbatch_size  16 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 12 12 --RUN
python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g 8 --microbatch_size  10 --accum  3 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 12 12 --RUN
python submit_benchmarks.py --project $PROJECT -m  30b.yaml -g 8 --microbatch_size   1 --accum 21 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 12 12 --RUN
# seqlen 8192
python submit_benchmarks.py --project $PROJECT -m 125m.yaml -g 8 --microbatch_size   8 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 13 13 --RUN
python submit_benchmarks.py --project $PROJECT -m 350m.yaml -g 8 --microbatch_size   8 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 13 13 --RUN
python submit_benchmarks.py --project $PROJECT -m 760m.yaml -g 8 --microbatch_size   6 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 13 13 --RUN
python submit_benchmarks.py --project $PROJECT -m   1b.yaml -g 8 --microbatch_size   3 --accum  4 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 13 13 --RUN
python submit_benchmarks.py --project $PROJECT -m   3b.yaml -g 8 --microbatch_size   3 --accum  6 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 13 13 --RUN --fsdp_config_activation_checkpointing false
python submit_benchmarks.py --project $PROJECT -m   7b.yaml -g 8 --microbatch_size   8 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 13 13 --RUN
python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g 8 --microbatch_size   5 --accum  3 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 13 13 --RUN
python submit_benchmarks.py --project $PROJECT -m  30b.yaml -g 8 --microbatch_size   2 --accum 1 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 13 13 --RUN
# seqlen 16384
python submit_benchmarks.py --project $PROJECT -m 125m.yaml -g 8 --microbatch_size   4 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 14 14 --RUN
python submit_benchmarks.py --project $PROJECT -m 350m.yaml -g 8 --microbatch_size   4 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 14 14 --RUN
python submit_benchmarks.py --project $PROJECT -m 760m.yaml -g 8 --microbatch_size   3 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 14 14 --RUN
python submit_benchmarks.py --project $PROJECT -m   1b.yaml -g 8 --microbatch_size   2 --accum  4 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 14 14 --RUN
python submit_benchmarks.py --project $PROJECT -m   3b.yaml -g 8 --microbatch_size   1 --accum  6 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 14 14 --RUN --fsdp_config_activation_checkpointing false
python submit_benchmarks.py --project $PROJECT -m   7b.yaml -g 8 --microbatch_size   4 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 14 14 --RUN
python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g 8 --microbatch_size   3 --accum  3 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 14 14 --RUN
# seqlen 32768
python submit_benchmarks.py --project $PROJECT -m 125m.yaml -g 8 --microbatch_size   2 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 15 15 --RUN
python submit_benchmarks.py --project $PROJECT -m 350m.yaml -g 8 --microbatch_size   2 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 15 15 --RUN
python submit_benchmarks.py --project $PROJECT -m 760m.yaml -g 8 --microbatch_size   1 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 15 15 --RUN
python submit_benchmarks.py --project $PROJECT -m   1b.yaml -g 8 --microbatch_size   1 --accum  4 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 15 15 --RUN
python submit_benchmarks.py --project $PROJECT -m   3b.yaml -g 8 --microbatch_size   3 --accum  6 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 15 15 --RUN
python submit_benchmarks.py --project $PROJECT -m   7b.yaml -g 8 --microbatch_size   2 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 15 15 --RUN
python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g 8 --microbatch_size   1 --accum  3 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 15 15 --RUN
# seqlen 65536
python submit_benchmarks.py --project $PROJECT -m 125m.yaml -g 8 --microbatch_size   1 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 16 16 --RUN
python submit_benchmarks.py --project $PROJECT -m 350m.yaml -g 8 --microbatch_size   1 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 16 16 --RUN
python submit_benchmarks.py --project $PROJECT -m 760m.yaml -g 8 --microbatch_size   1 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 16 16 --RUN --fsdp_config_activation_checkpointing true
python submit_benchmarks.py --project $PROJECT -m   1b.yaml -g 8 --microbatch_size   1 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 16 16 --RUN --fsdp_config_activation_checkpointing true
python submit_benchmarks.py --project $PROJECT -m   3b.yaml -g 8 --microbatch_size   1 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 16 16 --RUN
python submit_benchmarks.py --project $PROJECT -m   7b.yaml -g 8 --microbatch_size   1 --accum  2 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_80gb --cluster $CLUSTER_80GB -s 16 16 --RUN
