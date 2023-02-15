#!/bin/bash

PROJECT=YOUR_PROJECT
GIT_COMMIT=YOUR_GIT_COMMIT
CLUSTER_40GB=YOUR_CLUSTER_40GB
CLUSTER_80GB=YOUR_CLUSTER_80GB
IMAGE="mosaicml/pytorch:1.13.1_cu117-python3.10-ubuntu20.04"


# 40GB
# python submit_benchmarks.py --project $PROJECT -s 11 11 -b 19 22 -m 125m.yaml -g 8 16 32 64 --image $IMAGE --git_commit $GIT_COMMIT --pad_vocab_multiple 16 --gpu_type a100_40gb --cluster $CLUSTER_40GB --RUN --microbatch_size 16
# python submit_benchmarks.py --project $PROJECT -s 11 11 -b 19 22 -m 350m.yaml -g 8 16 32 64 --image $IMAGE --git_commit $GIT_COMMIT --pad_vocab_multiple 16 --gpu_type a100_40gb --cluster $CLUSTER_40GB --RUN --microbatch_size 8
# python submit_benchmarks.py --project $PROJECT -s 11 11 -b 19 22 -m 760m.yaml -g 8 16 32 64 --image $IMAGE --git_commit $GIT_COMMIT --pad_vocab_multiple 16 --gpu_type a100_40gb --cluster $CLUSTER_40GB --RUN --microbatch_size 6
# python submit_benchmarks.py --project $PROJECT -s 11 11 -b 19 22 -m 1b.yaml -g 8 16 32 64 --image $IMAGE --git_commit $GIT_COMMIT --pad_vocab_multiple 16 --gpu_type a100_40gb --cluster $CLUSTER_40GB --RUN --microbatch_size 4
# python submit_benchmarks.py --project $PROJECT -s 11 11 -b 19 22 -m 3b.yaml -g 8 16 32 64 --image $IMAGE --git_commit $GIT_COMMIT --pad_vocab_multiple 16 --gpu_type a100_40gb --cluster $CLUSTER_40GB --RUN --microbatch_size 14
# python submit_benchmarks.py --project $PROJECT -s 11 11 -b 19 22 -m 7b.yaml -g 8 --image $IMAGE --git_commit $GIT_COMMIT --pad_vocab_multiple 16 --gpu_type a100_40gb --cluster $CLUSTER_40GB --RUN --microbatch_size 6
# python submit_benchmarks.py --project $PROJECT -s 11 11 -b 19 22 -m 7b.yaml -g 16 32 64 --image $IMAGE --git_commit $GIT_COMMIT --pad_vocab_multiple 16 --gpu_type a100_40gb --cluster $CLUSTER_40GB --RUN --microbatch_size 8


# 80GB
# python submit_benchmarks.py --project $PROJECT -s 11 11 -b 19 22 -m 125m.yaml -g 8 16 --image $IMAGE --git_commit $GIT_COMMIT --pad_vocab_multiple 16 --gpu_type a100_80gb --cluster $CLUSTER_80GB --RUN --microbatch_size 32
# python submit_benchmarks.py --project $PROJECT -s 11 11 -b 19 19 -m 350m.yaml -g 8 16 --image $IMAGE --git_commit $GIT_COMMIT --pad_vocab_multiple 16 --gpu_type a100_80gb --cluster $CLUSTER_80GB --RUN --microbatch_size 16
# python submit_benchmarks.py --project $PROJECT -s 11 11 -b 19 19 -m 760m.yaml -g 8 16 --image $IMAGE --git_commit $GIT_COMMIT --pad_vocab_multiple 16 --gpu_type a100_80gb --cluster $CLUSTER_80GB --RUN --microbatch_size 12
# python submit_benchmarks.py --project $PROJECT -s 11 11 -b 19 19 -m 1b.yaml -g 8 16 --image $IMAGE --git_commit $GIT_COMMIT --pad_vocab_multiple 16 --gpu_type a100_80gb --cluster $CLUSTER_80GB --RUN --microbatch_size 10
# python submit_benchmarks.py --project $PROJECT -s 11 11 -b 19 19 -m 3b.yaml -g 8 16 --image $IMAGE --git_commit $GIT_COMMIT --pad_vocab_multiple 16 --gpu_type a100_80gb --cluster $CLUSTER_80GB --RUN --microbatch_size 6 --fsdp_config_activation_checkpointing false
# python submit_benchmarks.py --project $PROJECT -s 11 11 -b 19 19 -m 7b.yaml -g 8 16 --image $IMAGE --git_commit $GIT_COMMIT --pad_vocab_multiple 16 --gpu_type a100_80gb --cluster $CLUSTER_80GB --RUN --microbatch_size 20
# python submit_benchmarks.py --project $PROJECT -s 11 11 -b 19 19 -m 13b.yaml -g 8 16 --image $IMAGE --git_commit $GIT_COMMIT --pad_vocab_multiple 16 --gpu_type a100_80gb --cluster $CLUSTER_80GB --RUN --microbatch_size 12
# python submit_benchmarks.py --project $PROJECT -s 11 11 -b 19 22 -m 13b.yaml -g 16 --image $IMAGE --git_commit $GIT_COMMIT --pad_vocab_multiple 16 --gpu_type a100_80gb --cluster $CLUSTER_80GB --RUN --microbatch_size 16
