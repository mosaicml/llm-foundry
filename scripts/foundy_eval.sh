#!/bin/bash
#SBATCH --nodes=1 # number of nodes to use, 24 p4d(e) = 192 A100 GPUs
#SBATCH --ntasks=1
#SBATCH --job-name=eval-mpt # name of your job
#SBATCH --output=logs/%x_%j.out # logfile for stdout
#SBATCH --error=logs/%x_%j.err # logfile for stderr, remove it to merge both outputs
#SBATCH --ntasks-per-node 1 # Number of GPU per node
#SBATCH --gpus-per-node=8 # Number of GPU per node
#SBATCH --gpus-per-task=8 # Number of GPU per node
#SBATCH --gres=gpu:8 # number of GPU we reserve

export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
## for future multi-node runs
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`
echo go $COUNT_NODE
echo $HOSTNAMES

set -euxo pipefail

# use MTP 7B by default
MODEL=${1:-mpt-760m}
USER_NAME='amro'

export PYTHONPATH=$LLM_FOUNDRY_PATH
# default variables for Enroot
export LLM_FOUNDRY_PATH=/fsx/ubuntu/users/${USER_NAME}/llm-foundry
export APPS_PATH=/fsx/ubuntu/users/josh/josh-mpt-run-files
export ENROOT_IMAGE=$APPS_PATH/llm-foundry.sqsh
export IMAGE=$ENROOT_IMAGE
export FSX_PATH=/fsx/ubuntu
export FSX_MOUNT=$FSX_PATH/$USER_NAME:$FSX_PATH/$USER_NAME
export APPS_MOUNT=$APPS_PATH:$APPS_PATH

## EFA settings
export FI_EFA_USE_DEVICE_RDMA=1 # use for p4d
export FI_EFA_FORK_SAFE=1
# export NCCL_ALGO=Ring
export FI_LOG_LEVEL=1
export FI_PROVIDER=efa # change to eth if you want to use ENA for comparisons
export FI_EFA_ENABLE_SHM_TRANSFER=1
export FI_EFA_USE_HUGE_PAGE=0
# https://discuss.pytorch.org/t/nccl-network-is-unreachable-connection-refused-when-initializing-ddp/137352
# https://github.com/pytorch/pytorch/issues/68893
#export NCCL_SOCKET_IFNAME=ens
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO

# variables for Enroot
declare -a ARGS=(
    --container-image $IMAGE
    --container-mounts /fsx:/fsx,/fsx/ubuntu/users/${USER_NAME}/llm-foundry:/app #,/fsx/.aws:/root/.aws
)

NODES=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
NODES_ARRAY=($NODES)
HEAD_NODE=${NODES_ARRAY[0]}
MASTER_ADDR=$(hostname --ip-address)
MASTER_PORT=$RANDOM
NNODES=$SLURM_JOB_NUM_NODES
NPROC=$SLURM_GPUS_PER_NODE
WORLD_SIZE=$(( $NNODES * $NPROC ))
srun -l "${ARGS[@]}" python3 -c "import streaming; streaming.base.util.clean_stale_shared_memory(); import os; print('PYTHONPATH: ', os.environ['PYTHONPATH'])"

echo USER_NAME $USER_NAME
echo NODES $NODES
echo NODES_ARRAY $NODES_ARRAY
echo HEAD_NODE $HEAD_NODE
echo MASTER_ADDR $MASTER_ADDR
echo MASTER_PORT $MASTER_PORT
echo NNODES $NNODES
echo NPROC $NPROC
echo WORLD_SIZE $WORLD_SIZE
echo ARGS ${ARGS[@]}


MODEL_NAME_OR_PATH=s3://datology-research/matthew/checkpoints/scuderia-poc/mpt-760m/ep1-ba27351-rank0.pt
EVAL_YAML_RELATIVE_PATH=users/${USER_NAME}/llm-foundry/scripts/eval/yamls/custom/eval-mpt-760m.yaml


date
echo "Run eval"
NODE_RANK=0
NODE=${HEAD_NODE}
srun -u --nodelist=${NODE} --ntasks=1 -l "${ARGS[@]}" composer \
    --world_size ${WORLD_SIZE} \
    --nproc ${NPROC} \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    --verbose /fsx/ubuntu/users/${USER_NAME}/llm-foundry/scripts/eval/eval.py \
    $FSX_PATH/${EVAL_YAML_RELATIVE_PATH} \
    model_name_or_path=${MODEL_NAME_OR_PATH} \

wait
date
echo "I am finished"