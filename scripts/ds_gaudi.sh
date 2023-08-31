#!/bin/bash

MODEL_SIZE=$1
GTBS=$2
DTMS=$3
N_HEADS=$4
ATTN_IMPL=$5
NAME="${MODEL_SIZE}-${GTBS}-${DTMS}-${N_HEADS}-${ATTN_IMPL}"

DEEPSPEED_USE_HPU=1 composer -n 8 --world_size 8 train/train.py "train/yamls/pretrain/mpt-${MODEL_SIZE}.yaml" \
	run_name=$NAME \
	device=hpu \
	optimizer="{'name':'decoupled_lionw', 'lr':0.0001}" \
	data_local=/root/abhi/datasets/c4 \
	train_loader.dataset.split=train_small \
	train_loader.dataset.num_canonical_nodes=1 \
	train_loader.dataset.shuffle_block_size=100 \
	train_loader.dataset.predownload=100 \
	eval_loader.dataset.split=val_small \
	eval_loader.dataset.num_canonical_nodes=1 \
	eval_loader.dataset.predownload=100 \
	eval_interval=0 \
	deepspeed_config.bf16.enabled=true \
	deepspeed_config.zero_allow_untested_optimizer=true \
	deepspeed_config.zero_optimization.stage=3 \
	deepspeed_config.zero_optimization.overlap_comm=false \
	deepspeed_config.zero_optimization.contiguous_gradients=true \
	deepspeed_config.zero_optimization.reduce_scatter=false \
	fsdp_config=null \
	model.attn_config.attn_impl=$ATTN_IMPL \
	model.loss_fn=torch_crossentropy \
	model.init_device=hpu \
	model.n_heads=$N_HEADS \
	max_duration=100ba \
	callbacks.speed_monitor.window_size=5 \
	global_train_batch_size=$GTBS \
	device_train_microbatch_size=$DTMS \
	loggers.wandb="{}"

pkill -9 python3
