#!/bin/bash

PROJECT="opt30b"
GIT_COMMIT="v0.0.4"
IMAGE="mosaicml/pytorch:2.0.1_cu118-python3.10-ubuntu20.04"
# IMAGE="mosaicml/pytorch:2.1.0_cu121-nightly20230827-python3.10-ubuntu20.04"
CLUSTER_80GB=r14z2

python submit_benchmarks.py --project $PROJECT -m  30b.yaml -g 8 --microbatch_size   3 --accum 21 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_80GB -s 11 11 --RUN
python submit_benchmarks.py --project $PROJECT -m  30b.yaml -g 16       --microbatch_size 10 --accum  3 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_80GB -s 11 11 --RUN
python submit_benchmarks.py --project $PROJECT -m  30b.yaml -g    32    --microbatch_size 14 --accum  3 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_80GB -s 11 11 --RUN
python submit_benchmarks.py --project $PROJECT -m  30b.yaml -g 8 --microbatch_size   2 --accum 1 --image $IMAGE --git_commit $GIT_COMMIT --gpu_type h100_80gb --cluster $CLUSTER_80GB -s 13 13 --RUN