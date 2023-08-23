#!/bin/bash

PROJECT="interntorch2"
GIT_COMMIT="v0.0.4"
IMAGE="mosaicml/pytorch:2.0.1_cu118-python3.10-ubuntu20.04"

# A100 40GB

# seqlen 2048
python submit_benchmarks.py --project $PROJECT -m 125m.yaml -g 8 --microbatch_size  24 --accum  3 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type a100_40gb --cluster $CLUSTER_40GB -s 11 11 --RUN --LOCAL