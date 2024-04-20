# MPT Training Benchmarks

Benchmark measurements for MPT models trained on [MosaicML platform](https://www.mosaicml.com/platform), including throughput, MFU, and HFU. Each model is based on optimized configurations of various sizes in the [yamls](../yamls) folder, ranging from a 125m to 70B parameter models.

To reproduce table results, first run:
```
./sweep.sh
```

Then, after the runs are completed:
```
python collect_results.py --save-path results
```
will use our Python API to collect and calculate the benchmark results, and then save as both a CSV file `results.csv`, and a markdown table `results.md`.


```
python submit_benchmarks.py --cluster [your_mosaicml_cluster] ARGS --RUN
```
can be used to sweep a larger set of configurations. For example usage of `submit_benchmarks.py` see `sweep.sh` which lists all benchmarks in the tables.

> **Note**
> The `collect_results.py` will by default find all runs with `tput` in the run name. To customize this project tag, use `--project` in both the submission and collection scripts.


## MFU and HFU

Model FLOPs Utilization (MFU) and Hardware FLOPS Utilization (HFU) are estimates, based on the measured throughput and the known FLOPs of the computation, of what percentage of the hardware's FLOPs are being used during training.

MFU calculates the utilization from the floating point operations required for a single forward/backwards pass of the model, and does not account for the additional compute required for other implementation details such as activation checkpointing. Thus, MFU is independent of implementation and hardware.

HFU attempts to capture the actual floating point operations incurred during the forward/backwards pass on the hardware. While it is a more accurate measurement of hardware utilization, it is less general and is difficult to compare across various hardware and implementation details.

For more information, see [Korthikanti et al, 2022](https://arxiv.org/abs/2205.05198). All FLOP calculations exclude the operations required for normalization, activation, and residuals.

### MFU

Per token, each parameter is used for a MAC (2 FLOPS) per network operation. Neural Network training has 3 network operations: forward pass, backward pass, and computation of parameter gradient.

The attention mechanism forward pass FLOPS are: `attn_flops_per_seq = n_layers * 2 * 2 * (d_model * (seq_len**2))`
```
flops_per_token = 2 * n_params
flops_per_seq = flops_per_token * seq_len
mfu* = 3 * flops_per_seq * seq_per_sec / (gpu_num * GPU_AVAILABLE_FLOPS)

attn_flops_per_seq = n_layers * 2 * 2 * (d_model * (seq_len**2))
mfu = (3 * flops_per_seq + 3 * attn_flops_per_seq) * seq_per_sec / (gpu_num * GPU_AVAILABLE_FLOPS)
```

### HFU

The HFU numbers shown below account for the fact that the networks use checkpointing and recomputes activations. This effectively requires an extra forward pass through the network.
```
hfu* = 4 * flops_per_seq * seq_per_sec / (gpu_num * GPU_AVAILABLE_FLOPS)
hfu = (4 * flops_per_seq + 4 * attn_flops_per_seq) * seq_per_sec / (gpu_num * GPU_AVAILABLE_FLOPS)
```

Note that these are approximations. Actual HFU would be higher since it includes the floating point operations for normalization, activation, and residual layers, as well as **all** recomputation. For example, our models use Flash Attention, which requires including an extra recompute factor for its recomputation in the forward pass. Therefore, the attention multiplier would be 5 instead of 4.

## Results

Below we include several configurations across different hardware platforms, sequence lengths and batch sizes. It is easy to benchmark configurations for your own use case. For example, using the Mosaic platform, to test MPT {13B, 30B} using fp16 with a batch size of 2M tokens and seq len {2k, 4k, 8k, 16k} run:
```
python submit_benchmarks.py -m 13b.yaml 30b.yaml -t fp16 -b 21 21 -s 11 14 --RUN
```
This will run 8 configs for 12 steps to get throughput numbers. `python collect_results.py` can then be used to parse all output training logs and create the tables below.

Our microbatching engine enables microbatch sizes that do not divide global batch size while being mathematically faithful to the global batch size. For example, a total batch size of 48, and a micro batch of 11, means we will accumulate gradients across microbatches of 11, 11, 11, 11, 4.

[comment]: # (TODO: Update tables with torch 2.0 after next Composer release)

## H100 80GB BF16 (Large Scale, >= 128 GPUs)
|  Model | SeqLen (T) | # GPUs | GPU | MFU | HFU | Model TFLOP | MicroBatchSize | GradAccum | GlobalBatchSize | Throughput (S/s) | Throughput (T/s) | Throughput (T/s/GPU) | GlobalBatchSize (T) | Precision | MP Mode | Sharding Strategy | Activation Checkpointing | Activation CPUOffload | NumParams |
|  --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  70b | 2048 | 512 | h100_80gb | 41.25 | 55.0 | 408 | 8 | 1 | 4096 | 251 | 515636 | 1007 | 8388608 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 64862437376 |
|  70b | 2048 | 256 | h100_80gb | 42.42 | 56.56 | 419 | 8 | 1 | 2048 | 129 | 265149 | 1035 | 4194304 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 64862437376 |
|  70b | 2048 | 128 | h100_80gb | 43.36 | 57.81 | 428 | 8 | 1 | 1024 | 66 | 135490 | 1058 | 2097152 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 64862437376 |
|  30b | 2048 | 512 | h100_80gb | 40.27 | 53.69 | 398 | 8 | 1 | 4096 | 528 | 1083366 | 2115 | 8388608 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 29975214080 |
|  30b | 2048 | 256 | h100_80gb | 40.89 | 54.52 | 404 | 8 | 1 | 2048 | 268 | 550022 | 2148 | 4194304 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 29975214080 |
|  30b | 2048 | 128 | h100_80gb | 41.85 | 55.8 | 414 | 8 | 1 | 1024 | 137 | 281491 | 2199 | 2097152 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 29975214080 |
|  13b | 2048 | 512 | h100_80gb | 41.12 | 54.83 | 406 | 16 | 1 | 8192 | 1238 | 2535811 | 4952 | 16777216 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 12853954560 |
|  13b | 2048 | 256 | h100_80gb | 41.42 | 55.23 | 409 | 16 | 1 | 4096 | 623 | 1277214 | 4989 | 8388608 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 12853954560 |
|  13b | 2048 | 128 | h100_80gb | 42.18 | 56.24 | 417 | 16 | 1 | 2048 | 317 | 650264 | 5080 | 4194304 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 12853954560 |
|  7b | 2048 | 512 | h100_80gb | 42.2 | 42.2 | 417 | 6 | 1 | 3072 | 2417 | 4951479 | 9670 | 6291456 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 6658859008 |
|  7b | 2048 | 256 | h100_80gb | 44.15 | 44.15 | 436 | 6 | 1 | 1536 | 1264 | 2590548 | 10119 | 3145728 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 6658859008 |
|  7b | 2048 | 128 | h100_80gb | 45.71 | 45.71 | 452 | 6 | 1 | 768 | 654 | 1340830 | 10475 | 1572864 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 6658859008 |
|  3b | 2048 | 512 | h100_80gb | 39.24 | 39.24 | 388 | 8 | 1 | 4096 | 5416 | 11092218 | 21664 | 8388608 | amp_bf16 | DEFAULT | SHARD_GRAD_OP | False | False | 2651837440 |
|  3b | 2048 | 256 | h100_80gb | 41.25 | 41.25 | 408 | 8 | 1 | 2048 | 2846 | 5829686 | 22772 | 4194304 | amp_bf16 | DEFAULT | SHARD_GRAD_OP | False | False | 2651837440 |
|  3b | 2048 | 128 | h100_80gb | 42.43 | 42.43 | 419 | 8 | 1 | 1024 | 1463 | 2998098 | 23422 | 2097152 | amp_bf16 | DEFAULT | SHARD_GRAD_OP | False | False | 2651837440 |
|  1b | 2048 | 512 | h100_80gb | 36.65 | 36.65 | 362 | 12 | 1 | 6144 | 9959 | 20396905 | 39837 | 12582912 | amp_bf16 | DEFAULT | SHARD_GRAD_OP | False | False | 1315950592 |
|  1b | 2048 | 256 | h100_80gb | 39.15 | 39.15 | 387 | 12 | 1 | 3072 | 5319 | 10894207 | 42555 | 6291456 | amp_bf16 | DEFAULT | SHARD_GRAD_OP | False | False | 1315950592 |
|  1b | 2048 | 128 | h100_80gb | 40.6 | 40.6 | 401 | 12 | 1 | 1536 | 2757 | 5647854 | 44123 | 3145728 | amp_bf16 | DEFAULT | SHARD_GRAD_OP | False | False | 1315950592 |


## H100 80GB BF16
|  Model | SeqLen (T) | # GPUs | GPU | MFU | HFU | Model TFLOP | MicroBatchSize | GradAccum | GlobalBatchSize | Throughput (S/s) | Throughput (T/s) | Throughput (T/s/GPU) | GlobalBatchSize (T) | Precision | MP Mode | Sharding Strategy | Activation Checkpointing | Activation CPUOffload | NumParams |
|  --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  70b | 2048 | 64 | h100_80gb | 42.57 | 56.76 | 421 | 8 | 4 | 2048 | 32 | 66523 | 1039 | 4194304 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 64862437376 |
|  70b | 2048 | 32 | h100_80gb | 36.15 | 48.2 | 357 | 2 | 16 | 1024 | 13 | 28242 | 882 | 2097152 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 64862437376 |
|  30b | 8192 | 8 | h100_80gb | 29.92 | 39.9 | 296 | 1 | 21 | 168 | 1 | 11072 | 1384 | 1376256 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 30019254272 |
|  30b | 4096 | 8 | h100_80gb | 35.86 | 47.81 | 354 | 1 | 21 | 168 | 3 | 14419 | 1802 | 688128 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 29989894144 |
|  30b | 2048 | 32 | h100_80gb | 43.92 | 58.57 | 434 | 14 | 3 | 1344 | 36 | 73860 | 2308 | 2752512 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 29975214080 |
|  30b | 2048 | 16 | h100_80gb | 43.07 | 57.42 | 426 | 10 | 3 | 480 | 17 | 36209 | 2263 | 983040 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 29975214080 |
|  30b | 2048 | 8 | h100_80gb | 38.11 | 50.82 | 377 | 3 | 21 | 504 | 7 | 16022 | 2002 | 1032192 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 29975214080 |
|  30b | 1024 | 8 | h100_80gb | 38.76 | 51.68 | 383 | 6 | 21 | 1008 | 16 | 16672 | 2084 | 1032192 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 29967874048 |
|  13b | 32768 | 8 | h100_80gb | 31.68 | 42.24 | 313 | 1 | 3 | 24 | 0 | 15812 | 1976 | 786432 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 13011240960 |
|  13b | 16384 | 8 | h100_80gb | 35.55 | 47.4 | 351 | 3 | 3 | 72 | 1 | 23881 | 2985 | 1179648 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 12927354880 |
|  13b | 4096 | 8 | h100_80gb | 41.6 | 55.47 | 411 | 10 | 3 | 240 | 9 | 37740 | 4717 | 983040 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 12864440320 |
|  13b | 2048 | 64 | h100_80gb | 39.86 | 39.86 | 394 | 2 | 1 | 128 | 150 | 307209 | 4800 | 262144 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 12853954560 |
|  13b | 2048 | 32 | h100_80gb | 39.95 | 39.95 | 395 | 2 | 1 | 64 | 75 | 153960 | 4811 | 131072 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 12853954560 |
|  13b | 2048 | 16 | h100_80gb | 39.58 | 39.58 | 391 | 2 | 1 | 32 | 37 | 76280 | 4767 | 65536 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 12853954560 |
|  13b | 2048 | 8 | h100_80gb | 39.79 | 39.79 | 393 | 2 | 1 | 16 | 18 | 38336 | 4792 | 32768 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 12853954560 |
|  13b | 1024 | 8 | h100_80gb | 44.27 | 59.03 | 438 | 40 | 3 | 960 | 42 | 44019 | 5502 | 983040 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 12848711680 |
|  7b | 65536 | 8 | h100_80gb | 28.59 | 38.13 | 282 | 1 | 2 | 16 | 0 | 15654 | 1956 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 6918905856 |
|  7b | 32768 | 8 | h100_80gb | 30.94 | 41.25 | 306 | 2 | 2 | 32 | 0 | 26550 | 3318 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 6784688128 |
|  7b | 8192 | 8 | h100_80gb | 37.14 | 49.52 | 367 | 8 | 2 | 128 | 6 | 55481 | 6935 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 6684024832 |
|  7b | 4096 | 8 | h100_80gb | 40.42 | 53.9 | 399 | 16 | 2 | 256 | 16 | 68893 | 8611 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 6667247616 |
|  7b | 2048 | 8 | h100_80gb | 46.44 | 46.44 | 459 | 6 | 1 | 48 | 41 | 85144 | 10643 | 98304 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 6658859008 |
|  7b | 1024 | 8 | h100_80gb | 42.83 | 57.11 | 423 | 64 | 2 | 1024 | 79 | 81628 | 10203 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 6654664704 |
|  3b | 65536 | 8 | h100_80gb | 26.81 | 35.74 | 265 | 1 | 2 | 16 | 0 | 26099 | 3262 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 2814366720 |
|  3b | 32768 | 8 | h100_80gb | 28.84 | 38.46 | 285 | 3 | 6 | 144 | 1 | 46984 | 5873 | 4718592 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 2730480640 |
|  3b | 16384 | 8 | h100_80gb | 36.34 | 36.34 | 359 | 1 | 6 | 48 | 5 | 89223 | 11152 | 786432 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 2688537600 |
|  3b | 8192 | 8 | h100_80gb | 40.31 | 40.31 | 398 | 3 | 6 | 144 | 16 | 132626 | 16578 | 1179648 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 2667566080 |
|  3b | 4096 | 8 | h100_80gb | 42.31 | 42.31 | 418 | 5 | 6 | 240 | 40 | 167712 | 20964 | 983040 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 2657080320 |
|  3b | 2048 | 64 | h100_80gb | 40.8 | 40.8 | 403 | 6 | 3 | 1152 | 703 | 1441663 | 22525 | 2359296 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 2651837440 |
|  3b | 2048 | 32 | h100_80gb | 41.7 | 41.7 | 412 | 6 | 3 | 576 | 359 | 736701 | 23021 | 1179648 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 2651837440 |
|  3b | 2048 | 16 | h100_80gb | 43.73 | 43.73 | 432 | 10 | 3 | 480 | 188 | 386285 | 24142 | 983040 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 2651837440 |
|  3b | 1024 | 8 | h100_80gb | 46.2 | 46.2 | 457 | 20 | 6 | 960 | 211 | 216369 | 27046 | 983040 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 2649216000 |
|  3b | 512 | 8 | h100_80gb | 46.32 | 46.32 | 458 | 40 | 6 | 1920 | 436 | 223721 | 27965 | 983040 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 2647905280 |
|  1b | 65536 | 8 | h100_80gb | 26.34 | 35.12 | 260 | 1 | 2 | 16 | 0 | 44050 | 5506 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 1445974016 |
|  1b | 32768 | 8 | h100_80gb | 33.54 | 33.54 | 331 | 1 | 4 | 32 | 2 | 96203 | 12025 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 1378865152 |
|  1b | 16384 | 8 | h100_80gb | 35.22 | 35.22 | 348 | 2 | 4 | 64 | 9 | 157194 | 19649 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 1345310720 |
|  1b | 8192 | 8 | h100_80gb | 37.73 | 37.73 | 373 | 3 | 4 | 96 | 28 | 233256 | 29157 | 786432 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 1328533504 |
|  1b | 4096 | 8 | h100_80gb | 40.26 | 40.26 | 398 | 7 | 4 | 224 | 75 | 308282 | 38535 | 917504 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 1320144896 |
|  1b | 2048 | 64 | h100_80gb | 40.85 | 40.85 | 404 | 20 | 1 | 1280 | 1387 | 2841754 | 44402 | 2621440 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 1315950592 |
|  1b | 2048 | 32 | h100_80gb | 41.52 | 41.52 | 410 | 20 | 1 | 640 | 705 | 1444183 | 45130 | 1310720 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 1315950592 |
|  1b | 2048 | 16 | h100_80gb | 42.36 | 42.36 | 419 | 20 | 1 | 320 | 359 | 736596 | 46037 | 655360 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 1315950592 |
|  1b | 2048 | 8 | h100_80gb | 41.82 | 41.82 | 413 | 14 | 1 | 112 | 177 | 363645 | 45455 | 229376 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 1315950592 |
|  1b | 1024 | 8 | h100_80gb | 41.95 | 41.95 | 415 | 18 | 4 | 576 | 382 | 391287 | 48910 | 589824 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 1313853440 |
|  1b | 512 | 8 | h100_80gb | 43.21 | 43.21 | 427 | 56 | 4 | 1792 | 816 | 418201 | 52275 | 917504 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 1312804864 |
|  760m | 32768 | 8 | h100_80gb | 31.84 | 31.84 | 315 | 1 | 2 | 16 | 3 | 130333 | 16291 | 524288 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 807656448 |
|  760m | 16384 | 8 | h100_80gb | 33.57 | 33.57 | 332 | 3 | 2 | 48 | 13 | 222521 | 27815 | 786432 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 782490624 |
|  760m | 8192 | 8 | h100_80gb | 34.84 | 34.84 | 344 | 6 | 2 | 96 | 40 | 334602 | 41825 | 786432 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 769907712 |
|  760m | 4096 | 8 | h100_80gb | 35.83 | 35.83 | 354 | 12 | 2 | 192 | 108 | 443674 | 55459 | 786432 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 763616256 |
|  760m | 2048 | 32 | h100_80gb | 37.57 | 37.57 | 371 | 24 | 1 | 768 | 1062 | 2175091 | 67971 | 1572864 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 760470528 |
|  760m | 2048 | 16 | h100_80gb | 37.89 | 37.89 | 374 | 24 | 1 | 384 | 535 | 1096819 | 68551 | 786432 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 760470528 |
|  760m | 2048 | 8 | h100_80gb | 34.9 | 34.9 | 345 | 24 | 2 | 384 | 246 | 505177 | 63147 | 786432 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 760470528 |
|  760m | 1024 | 8 | h100_80gb | 39.76 | 39.76 | 393 | 48 | 2 | 768 | 613 | 628648 | 78581 | 786432 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 758897664 |
|  760m | 512 | 8 | h100_80gb | 40.42 | 40.42 | 399 | 96 | 2 | 1536 | 1308 | 669998 | 83749 | 786432 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 758111232 |

## H100 80GB FP8
|  Model | SeqLen (T) | # GPUs | GPU | MFU | HFU | Model TFLOP | MicroBatchSize | GradAccum | GlobalBatchSize | Throughput (S/s) | Throughput (T/s) | Throughput (T/s/GPU) | GlobalBatchSize (T) | Precision | MP Mode | Sharding Strategy | Activation Checkpointing | Activation CPUOffload | NumParams |
|  --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  3b | 32768 | 8 | h100_80gb | 14.38 | 19.18 | 284 | 3 | 6 | 144 | 1 | 46853 | 5856 | 4718592 | amp_fp8 | DEFAULT | FULL_SHARD | True | False | 2730480640 |
|  3b | 8192 | 8 | h100_80gb | 23.28 | 23.28 | 460 | 3 | 6 | 144 | 18 | 153174 | 19146 | 1179648 | amp_fp8 | DEFAULT | FULL_SHARD | False | False | 2667566080 |
|  3b | 2048 | 8 | h100_80gb | 27.7 | 27.7 | 548 | 10 | 6 | 480 | 119 | 244692 | 30586 | 983040 | amp_fp8 | DEFAULT | FULL_SHARD | False | False | 2651837440 |
|  3b | 512 | 8 | h100_80gb | 30.25 | 30.25 | 598 | 40 | 6 | 1920 | 570 | 292217 | 36527 | 983040 | amp_fp8 | DEFAULT | FULL_SHARD | False | False | 2647905280 |
|  1b | 32768 | 8 | h100_80gb | 17.55 | 17.55 | 347 | 1 | 4 | 32 | 3 | 100643 | 12580 | 1048576 | amp_fp8 | DEFAULT | FULL_SHARD | False | False | 1378865152 |
|  1b | 8192 | 8 | h100_80gb | 20.71 | 20.71 | 409 | 2 | 4 | 64 | 31 | 256087 | 32010 | 524288 | amp_fp8 | DEFAULT | FULL_SHARD | False | False | 1328533504 |
|  1b | 512 | 8 | h100_80gb | 29.06 | 29.06 | 575 | 56 | 4 | 1792 | 1098 | 562523 | 70315 | 917504 | amp_fp8 | DEFAULT | FULL_SHARD | False | False | 1312804864 |

## A100 80GB with 1600 Gbps node-node interconnect (RoCE)

|  Model | SeqLen (T) | # GPUs | GPU | MFU | HFU | Model TFLOP | MicroBatchSize | GradAccum | GlobalBatchSize | Throughput (S/s) | Throughput (T/s) | Throughput (T/s/GPU) | GlobalBatchSize (T) | Precision | MP Mode | Sharding Strategy | Activation Checkpointing | Activation CPUOffload | NumParams |
|  --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  70b | 2048 | 64 | a100_80gb | 53.33 | 71.1 | 166 | 8 | 4 | 2048 | 12 | 26274 | 410 | 4194304 | bf16 | PURE | FULL_SHARD | True | False | 64862437376 |
|  70b | 2048 | 32 | a100_80gb | 48.56 | 64.75 | 151 | 2 | 16 | 1024 | 5 | 11962 | 373 | 2097152 | bf16 | PURE | FULL_SHARD | True | False | 64862437376 |
|  30b | 8192 | 8 | a100_80gb | 39.38 | 52.5 | 122 | 1 | 21 | 168 | 0 | 4594 | 574 | 1376256 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 30019254272 |
|  30b | 4096 | 8 | a100_80gb | 51.37 | 68.49 | 160 | 1 | 21 | 168 | 1 | 6513 | 814 | 688128 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 29989894144 |
|  30b | 2048 | 8 | a100_80gb | 55.3 | 73.74 | 172 | 3 | 21 | 504 | 3 | 7330 | 916 | 1032192 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 29975214080 |
|  30b | 1024 | 8 | a100_80gb | 55.82 | 74.43 | 174 | 6 | 21 | 1008 | 7 | 7571 | 946 | 1032192 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 29967874048 |
|  30b | 512 | 8 | a100_80gb | 56.4 | 75.2 | 175 | 12 | 21 | 2016 | 15 | 7739 | 967 | 1032192 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 29964204032 |
|  13b | 32768 | 8 | a100_80gb | 51.69 | 68.92 | 161 | 1 | 3 | 24 | 0 | 8134 | 1016 | 786432 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 13011240960 |
|  13b | 16384 | 8 | a100_80gb | 54.07 | 72.1 | 168 | 3 | 3 | 72 | 0 | 11454 | 1431 | 1179648 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 12927354880 |
|  13b | 8192 | 8 | a100_80gb | 56.07 | 74.76 | 174 | 5 | 3 | 120 | 1 | 14362 | 1795 | 983040 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 12885411840 |
|  13b | 4096 | 8 | a100_80gb | 57.62 | 76.82 | 179 | 10 | 3 | 240 | 4 | 16482 | 2060 | 983040 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 12864440320 |
|  13b | 2048 | 8 | a100_80gb | 59.57 | 59.57 | 185 | 2 | 3 | 48 | 8 | 18097 | 2262 | 98304 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 12853954560 |
|  13b | 1024 | 8 | a100_80gb | 59.48 | 79.3 | 185 | 40 | 3 | 960 | 18 | 18647 | 2330 | 983040 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 12848711680 |
|  7b | 65536 | 8 | a100_80gb | 46.97 | 62.63 | 146 | 1 | 2 | 16 | 0 | 8108 | 1013 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 6918905856 |
|  7b | 32768 | 8 | a100_80gb | 49.46 | 65.94 | 154 | 2 | 2 | 32 | 0 | 13382 | 1672 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 6784688128 |
|  7b | 16384 | 8 | a100_80gb | 51.96 | 69.28 | 162 | 4 | 2 | 64 | 1 | 19629 | 2453 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 6717579264 |
|  7b | 8192 | 8 | a100_80gb | 54.47 | 72.62 | 169 | 8 | 2 | 128 | 3 | 25655 | 3206 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 6684024832 |
|  7b | 4096 | 8 | a100_80gb | 54.84 | 73.12 | 171 | 16 | 2 | 256 | 7 | 29472 | 3684 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 6667247616 |
|  7b | 2048 | 8 | a100_80gb | 64.23 | 64.23 | 200 | 6 | 2 | 96 | 18 | 37130 | 4641 | 196608 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 6658859008 |
|  7b | 1024 | 8 | a100_80gb | 58.01 | 77.35 | 180 | 64 | 2 | 1024 | 34 | 34857 | 4357 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 6654664704 |
|  3b | 65536 | 8 | a100_80gb | 46.05 | 61.41 | 143 | 1 | 2 | 16 | 0 | 14137 | 1767 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 2814366720 |
|  3b | 32768 | 8 | a100_80gb | 47.18 | 62.91 | 147 | 3 | 6 | 144 | 0 | 24235 | 3029 | 4718592 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 2730480640 |
|  3b | 16384 | 8 | a100_80gb | 57.13 | 57.13 | 178 | 1 | 6 | 48 | 2 | 44233 | 5529 | 786432 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 2688537600 |
|  3b | 8192 | 8 | a100_80gb | 59.34 | 59.34 | 185 | 3 | 6 | 144 | 7 | 61567 | 7695 | 1179648 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 2667566080 |
|  3b | 4096 | 8 | a100_80gb | 60.53 | 60.53 | 188 | 5 | 6 | 240 | 18 | 75658 | 9457 | 983040 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 2657080320 |
|  3b | 2048 | 8 | a100_80gb | 62.11 | 62.11 | 193 | 10 | 2 | 160 | 42 | 86491 | 10811 | 327680 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 2651837440 |
|  3b | 1024 | 8 | a100_80gb | 62.73 | 62.73 | 195 | 20 | 6 | 960 | 90 | 92643 | 11580 | 983040 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 2649216000 |
|  3b | 512 | 8 | a100_80gb | 63.71 | 63.71 | 198 | 40 | 6 | 1920 | 189 | 97019 | 12127 | 983040 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 2647905280 |
|  1b | 65536 | 8 | a100_80gb | 46.18 | 61.57 | 144 | 1 | 2 | 16 | 0 | 24353 | 3044 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 1445974016 |
|  1b | 32768 | 8 | a100_80gb | 55.52 | 55.52 | 173 | 1 | 4 | 32 | 1 | 50207 | 6275 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 1378865152 |
|  1b | 16384 | 8 | a100_80gb | 56.6 | 56.6 | 176 | 2 | 4 | 64 | 4 | 79650 | 9956 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 1345310720 |
|  1b | 8192 | 8 | a100_80gb | 56.69 | 56.69 | 176 | 3 | 4 | 96 | 13 | 110516 | 13814 | 786432 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 1328533504 |
|  1b | 4096 | 8 | a100_80gb | 59.0 | 59.0 | 184 | 7 | 4 | 224 | 34 | 142457 | 17807 | 917504 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 1320144896 |
|  1b | 2048 | 8 | a100_80gb | 59.86 | 59.86 | 186 | 14 | 4 | 448 | 80 | 164109 | 20513 | 917504 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 1315950592 |
|  1b | 1024 | 8 | a100_80gb | 60.15 | 60.15 | 187 | 18 | 4 | 576 | 172 | 176898 | 22112 | 589824 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 1313853440 |
|  1b | 512 | 8 | a100_80gb | 60.68 | 60.68 | 189 | 56 | 4 | 1792 | 361 | 185186 | 23148 | 917504 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 1312804864 |
|  760m | 65536 | 8 | a100_80gb | 45.34 | 60.45 | 141 | 1 | 2 | 16 | 0 | 33150 | 4143 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 857988096 |
|  760m | 32768 | 8 | a100_80gb | 54.57 | 54.57 | 170 | 1 | 2 | 16 | 2 | 70417 | 8802 | 524288 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 807656448 |
|  760m | 16384 | 8 | a100_80gb | 54.64 | 54.64 | 170 | 3 | 2 | 48 | 6 | 114198 | 14274 | 786432 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 782490624 |
|  760m | 8192 | 8 | a100_80gb | 55.31 | 55.31 | 172 | 6 | 2 | 96 | 20 | 167471 | 20933 | 786432 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 769907712 |
|  760m | 4096 | 8 | a100_80gb | 56.05 | 56.05 | 174 | 12 | 2 | 192 | 53 | 218808 | 27351 | 786432 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 763616256 |
|  760m | 2048 | 8 | a100_80gb | 56.85 | 56.85 | 177 | 24 | 2 | 384 | 126 | 259472 | 32434 | 786432 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 760470528 |
|  760m | 1024 | 8 | a100_80gb | 47.76 | 47.76 | 149 | 48 | 2 | 768 | 232 | 238122 | 29765 | 786432 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 758897664 |
|  760m | 512 | 8 | a100_80gb | 45.07 | 45.07 | 140 | 96 | 2 | 1536 | 460 | 235571 | 29446 | 786432 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 758111232 |
|  350m | 65536 | 8 | a100_80gb | 52.7 | 52.7 | 164 | 1 | 2 | 16 | 0 | 60195 | 7524 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 420997120 |
|  350m | 32768 | 8 | a100_80gb | 52.46 | 52.46 | 163 | 2 | 2 | 32 | 3 | 109222 | 13652 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 387442688 |
|  350m | 16384 | 8 | a100_80gb | 53.28 | 53.28 | 166 | 4 | 2 | 64 | 11 | 188478 | 23559 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 370665472 |
|  350m | 8192 | 8 | a100_80gb | 53.8 | 53.8 | 167 | 8 | 2 | 128 | 35 | 292559 | 36569 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 362276864 |
|  350m | 4096 | 8 | a100_80gb | 53.31 | 53.31 | 166 | 16 | 2 | 256 | 96 | 396442 | 49555 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 358082560 |
|  350m | 2048 | 8 | a100_80gb | 51.62 | 51.62 | 161 | 32 | 2 | 512 | 229 | 470263 | 58782 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 355985408 |
|  350m | 1024 | 8 | a100_80gb | 50.51 | 50.51 | 157 | 64 | 2 | 1024 | 506 | 518504 | 64813 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 354936832 |
|  350m | 512 | 8 | a100_80gb | 50.61 | 50.61 | 157 | 128 | 2 | 2048 | 1083 | 554643 | 69330 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 354412544 |
|  125m | 65536 | 8 | a100_80gb | 54.13 | 54.13 | 168 | 1 | 2 | 16 | 2 | 162946 | 20368 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 174070272 |
|  125m | 32768 | 8 | a100_80gb | 52.71 | 52.71 | 164 | 2 | 2 | 32 | 8 | 291256 | 36407 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 148904448 |
|  125m | 16384 | 8 | a100_80gb | 50.61 | 50.61 | 157 | 4 | 2 | 64 | 29 | 480322 | 60040 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 136321536 |
|  125m | 8192 | 8 | a100_80gb | 48.85 | 48.85 | 152 | 8 | 2 | 128 | 88 | 723142 | 90392 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 130030080 |
|  125m | 4096 | 8 | a100_80gb | 46.08 | 46.08 | 143 | 16 | 2 | 256 | 231 | 947172 | 118396 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 126884352 |
|  125m | 2048 | 8 | a100_80gb | 44.79 | 44.79 | 139 | 40 | 2 | 640 | 557 | 1142641 | 142830 | 1310720 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 125311488 |
|  125m | 2048 | 8 | a100_80gb | 44.45 | 44.45 | 138 | 32 | 2 | 512 | 553 | 1133901 | 141737 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 125311488 |
|  125m | 1024 | 8 | a100_80gb | 43.15 | 43.15 | 134 | 64 | 2 | 1024 | 1222 | 1251751 | 156468 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 124525056 |
|  125m | 512 | 8 | a100_80gb | 42.56 | 42.56 | 132 | 128 | 2 | 2048 | 2588 | 1325455 | 165681 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 124131840 |

## A100 40GB with 1600 Gbps node-node interconnect (RoCE)

|  Model | SeqLen (T) | # GPUs | GPU | MFU | HFU | Model TFLOP| MicroBatchSize | GradAccum | GlobalBatchSize | Throughput (S/s) | Throughput (T/s) | Throughput (T/s/GPU) | GlobalBatchSize (T) | Precision | MP Mode | Sharding Strategy | Activation Checkpointing | Activation CPUOffload | NumParams |
|  --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  70b | 2048 | 128 | a100_40gb | 48.91 | 65.21 | 152 | 4 | 1 | 512 | 23 | 48194 | 376 | 1048576 | bf16 | PURE | FULL_SHARD | True | False | 64862437376 |
|  70b | 2048 | 64 | a100_40gb | 35.87 | 47.82 | 111 | 2 | 1 | 128 | 8 | 17672 | 276 | 262144 | bf16 | PURE | FULL_SHARD | True | False | 64862437376 |
|  30b | 2048 | 128 | a100_40gb | 52.25 | 69.66 | 163 | 6 | 1 | 768 | 54 | 110803 | 865 | 1572864 | bf16 | PURE | FULL_SHARD | True | False | 29975214080 |
|  30b | 2048 | 32 | a100_40gb | 51.74 | 68.98 | 161 | 4 | 1 | 128 | 13 | 27431 | 857 | 262144 | bf16 | PURE | FULL_SHARD | True | False | 29975214080 |
|  13b | 8192 | 8 | a100_40gb | 43.95 | 58.6 | 137 | 1 | 16 | 128 | 1 | 11258 | 1407 | 1048576 | bf16 | PURE | FULL_SHARD | True | False | 12885411840 |
|  13b | 4096 | 8 | a100_40gb | 44.85 | 59.8 | 139 | 2 | 16 | 256 | 3 | 12830 | 1603 | 1048576 | bf16 | PURE | FULL_SHARD | True | False | 12864440320 |
|  13b | 2048 | 128 | a100_40gb | 51.93 | 69.24 | 162 | 16 | 1 | 2048 | 123 | 252444 | 1972 | 4194304 | bf16 | PURE | FULL_SHARD | True | False | 12853954560 |
|  13b | 2048 | 64 | a100_40gb | 52.04 | 69.39 | 162 | 16 | 1 | 1024 | 61 | 126479 | 1976 | 2097152 | bf16 | PURE | FULL_SHARD | True | False | 12853954560 |
|  13b | 2048 | 32 | a100_40gb | 52.62 | 70.16 | 164 | 14 | 1 | 448 | 31 | 63946 | 1998 | 917504 | bf16 | PURE | FULL_SHARD | True | False | 12853954560 |
|  13b | 2048 | 16 | a100_40gb | 52.5 | 70.0 | 163 | 10 | 1 | 160 | 15 | 31900 | 1993 | 327680 | bf16 | PURE | FULL_SHARD | True | False | 12853954560 |
|  13b | 2048 | 8 | a100_40gb | 43.94 | 58.58 | 137 | 4 | 16 | 512 | 6 | 13347 | 1668 | 1048576 | bf16 | PURE | FULL_SHARD | True | False | 12853954560 |
|  13b | 1024 | 8 | a100_40gb | 44.07 | 58.76 | 137 | 8 | 16 | 1024 | 13 | 13817 | 1727 | 1048576 | bf16 | PURE | FULL_SHARD | True | False | 12848711680 |
|  13b | 512 | 8 | a100_40gb | 44.28 | 59.04 | 138 | 16 | 16 | 2048 | 27 | 14108 | 1763 | 1048576 | bf16 | PURE | FULL_SHARD | True | False | 12846090240 |
|  7b | 16384 | 8 | a100_40gb | 47.65 | 63.53 | 148 | 1 | 4 | 32 | 1 | 17998 | 2249 | 524288 | bf16 | PURE | FULL_SHARD | True | False | 6717579264 |
|  7b | 8192 | 8 | a100_40gb | 49.04 | 65.38 | 153 | 3 | 4 | 96 | 2 | 23098 | 2887 | 786432 | bf16 | PURE | FULL_SHARD | True | False | 6684024832 |
|  7b | 4096 | 8 | a100_40gb | 50.11 | 66.82 | 156 | 6 | 4 | 192 | 6 | 26930 | 3366 | 786432 | bf16 | PURE | FULL_SHARD | True | False | 6667247616 |
|  7b | 2048 | 128 | a100_40gb | 50.14 | 66.85 | 156 | 18 | 1 | 2304 | 226 | 463749 | 3623 | 4718592 | bf16 | PURE | FULL_SHARD | True | False | 6658859008 |
|  7b | 2048 | 64 | a100_40gb | 50.73 | 67.64 | 158 | 18 | 1 | 1152 | 114 | 234614 | 3665 | 2359296 | bf16 | PURE | FULL_SHARD | True | False | 6658859008 |
|  7b | 2048 | 32 | a100_40gb | 51.55 | 68.73 | 160 | 18 | 1 | 576 | 58 | 119202 | 3725 | 1179648 | bf16 | PURE | FULL_SHARD | True | False | 6658859008 |
|  7b | 2048 | 16 | a100_40gb | 50.44 | 67.26 | 157 | 16 | 1 | 256 | 28 | 58322 | 3645 | 524288 | bf16 | PURE | FULL_SHARD | True | False | 6658859008 |
|  7b | 2048 | 8 | a100_40gb | 50.92 | 67.89 | 158 | 12 | 4 | 384 | 14 | 29436 | 3679 | 786432 | bf16 | PURE | FULL_SHARD | True | False | 6658859008 |
|  7b | 1024 | 8 | a100_40gb | 51.31 | 68.42 | 160 | 24 | 4 | 768 | 30 | 30833 | 3854 | 786432 | bf16 | PURE | FULL_SHARD | True | False | 6654664704 |
|  7b | 512 | 8 | a100_40gb | 50.85 | 67.8 | 158 | 48 | 4 | 1536 | 60 | 31167 | 3895 | 786432 | bf16 | PURE | FULL_SHARD | True | False | 6652567552 |
|  3b | 32768 | 8 | a100_40gb | 46.03 | 61.37 | 143 | 1 | 4 | 32 | 0 | 23640 | 2955 | 1048576 | bf16 | PURE | FULL_SHARD | True | False | 2730480640 |
|  3b | 16384 | 8 | a100_40gb | 46.14 | 61.52 | 143 | 2 | 8 | 128 | 2 | 35726 | 4465 | 2097152 | bf16 | PURE | FULL_SHARD | True | False | 2688537600 |
|  3b | 8192 | 8 | a100_40gb | 55.13 | 55.13 | 172 | 1 | 8 | 64 | 6 | 57193 | 7149 | 524288 | bf16 | PURE | FULL_SHARD | False | False | 2667566080 |
|  3b | 4096 | 8 | a100_40gb | 56.18 | 56.18 | 175 | 2 | 8 | 128 | 17 | 70223 | 8777 | 524288 | bf16 | PURE | FULL_SHARD | False | False | 2657080320 |
|  3b | 2048 | 128 | a100_40gb | 54.8 | 54.8 | 170 | 6 | 1 | 768 | 596 | 1220885 | 9538 | 1572864 | bf16 | PURE | FULL_SHARD | False | False | 2651837440 |
|  3b | 2048 | 64 | a100_40gb | 55.94 | 55.94 | 174 | 6 | 1 | 384 | 304 | 623167 | 9736 | 786432 | bf16 | PURE | FULL_SHARD | False | False | 2651837440 |
|  3b | 2048 | 32 | a100_40gb | 56.96 | 56.96 | 177 | 6 | 1 | 192 | 154 | 317261 | 9914 | 393216 | bf16 | PURE | FULL_SHARD | False | False | 2651837440 |
|  3b | 2048 | 16 | a100_40gb | 56.02 | 56.02 | 174 | 5 | 1 | 80 | 76 | 156013 | 9750 | 163840 | bf16 | PURE | FULL_SHARD | False | False | 2651837440 |
|  3b | 2048 | 8 | a100_40gb | 57.82 | 57.82 | 180 | 5 | 8 | 320 | 39 | 80520 | 10065 | 655360 | bf16 | PURE | FULL_SHARD | False | False | 2651837440 |
|  3b | 1024 | 8 | a100_40gb | 58.14 | 58.14 | 181 | 10 | 8 | 640 | 83 | 85854 | 10731 | 655360 | bf16 | PURE | FULL_SHARD | False | False | 2649216000 |
|  3b | 512 | 8 | a100_40gb | 59.49 | 59.49 | 185 | 20 | 8 | 1280 | 176 | 90596 | 11324 | 655360 | bf16 | PURE | FULL_SHARD | False | False | 2647905280 |
|  1b | 32768 | 8 | a100_40gb | 45.07 | 60.1 | 140 | 1 | 4 | 32 | 1 | 40762 | 5095 | 1048576 | bf16 | PURE | FULL_SHARD | True | False | 1378865152 |
|  1b | 16384 | 8 | a100_40gb | 55.23 | 55.23 | 172 | 1 | 8 | 64 | 4 | 77723 | 9715 | 1048576 | bf16 | PURE | FULL_SHARD | False | False | 1345310720 |
|  1b | 8192 | 8 | a100_40gb | 55.29 | 55.29 | 172 | 2 | 8 | 128 | 13 | 107799 | 13474 | 1048576 | bf16 | PURE | FULL_SHARD | False | False | 1328533504 |
|  1b | 4096 | 8 | a100_40gb | 55.85 | 55.85 | 174 | 4 | 8 | 256 | 32 | 134851 | 16856 | 1048576 | bf16 | PURE | FULL_SHARD | False | False | 1320144896 |
|  1b | 2048 | 128 | a100_40gb | 54.41 | 54.41 | 169 | 10 | 1 | 1280 | 1165 | 2386897 | 18647 | 2621440 | bf16 | PURE | FULL_SHARD | False | False | 1315950592 |
|  1b | 2048 | 64 | a100_40gb | 55.44 | 55.44 | 172 | 10 | 1 | 640 | 593 | 1216104 | 19001 | 1310720 | bf16 | PURE | FULL_SHARD | False | False | 1315950592 |
|  1b | 2048 | 32 | a100_40gb | 45.39 | 45.39 | 141 | 10 | 1 | 320 | 243 | 497782 | 15555 | 655360 | bf16 | PURE | FULL_SHARD | False | False | 1315950592 |
|  1b | 2048 | 16 | a100_40gb | 55.69 | 55.69 | 173 | 8 | 1 | 128 | 149 | 305372 | 19085 | 262144 | bf16 | PURE | FULL_SHARD | False | False | 1315950592 |
|  1b | 2048 | 8 | a100_40gb | 56.23 | 56.23 | 175 | 8 | 8 | 512 | 75 | 154171 | 19271 | 1048576 | bf16 | PURE | FULL_SHARD | False | False | 1315950592 |
|  1b | 1024 | 8 | a100_40gb | 57.02 | 57.02 | 177 | 16 | 8 | 1024 | 163 | 167677 | 20959 | 1048576 | bf16 | PURE | FULL_SHARD | False | False | 1313853440 |
|  1b | 512 | 8 | a100_40gb | 57.1 | 57.1 | 178 | 32 | 8 | 2048 | 340 | 174256 | 21782 | 1048576 | bf16 | PURE | FULL_SHARD | False | False | 1312804864 |
|  760m | 32768 | 8 | a100_40gb | 44.53 | 59.37 | 138 | 1 | 4 | 32 | 1 | 57464 | 7183 | 1048576 | bf16 | PURE | FULL_SHARD | True | False | 807656448 |
|  760m | 16384 | 8 | a100_40gb | 53.26 | 53.26 | 166 | 1 | 4 | 32 | 6 | 111316 | 13914 | 524288 | bf16 | PURE | FULL_SHARD | False | False | 782490624 |
|  760m | 8192 | 8 | a100_40gb | 53.12 | 53.12 | 165 | 3 | 4 | 96 | 19 | 160853 | 20106 | 786432 | bf16 | PURE | FULL_SHARD | False | False | 769907712 |
|  760m | 4096 | 8 | a100_40gb | 53.0 | 53.0 | 165 | 6 | 4 | 192 | 50 | 206909 | 25863 | 786432 | bf16 | PURE | FULL_SHARD | False | False | 763616256 |
|  760m | 2048 | 128 | a100_40gb | 50.73 | 50.73 | 158 | 12 | 1 | 1536 | 1808 | 3704382 | 28940 | 3145728 | bf16 | PURE | FULL_SHARD | False | False | 760470528 |
|  760m | 2048 | 64 | a100_40gb | 51.44 | 51.44 | 160 | 12 | 1 | 768 | 917 | 1878030 | 29344 | 1572864 | bf16 | PURE | FULL_SHARD | False | False | 760470528 |
|  760m | 2048 | 32 | a100_40gb | 51.97 | 51.97 | 162 | 12 | 1 | 384 | 463 | 948745 | 29648 | 786432 | bf16 | PURE | FULL_SHARD | False | False | 760470528 |
|  760m | 2048 | 16 | a100_40gb | 51.9 | 51.9 | 161 | 12 | 1 | 192 | 231 | 473723 | 29607 | 393216 | bf16 | PURE | FULL_SHARD | False | False | 760470528 |
|  760m | 2048 | 8 | a100_40gb | 52.89 | 52.89 | 165 | 12 | 4 | 384 | 117 | 241389 | 30173 | 786432 | bf16 | PURE | FULL_SHARD | False | False | 760470528 |
|  760m | 1024 | 8 | a100_40gb | 53.63 | 53.63 | 167 | 24 | 4 | 768 | 261 | 267376 | 33422 | 786432 | bf16 | PURE | FULL_SHARD | False | False | 758897664 |
|  760m | 512 | 8 | a100_40gb | 53.47 | 53.47 | 166 | 48 | 4 | 1536 | 545 | 279504 | 34938 | 786432 | bf16 | PURE | FULL_SHARD | False | False | 758111232 |
|  350m | 32768 | 8 | a100_40gb | 51.55 | 51.55 | 160 | 1 | 4 | 32 | 3 | 107329 | 13416 | 1048576 | bf16 | PURE | FULL_SHARD | False | False | 387442688 |
|  350m | 16384 | 8 | a100_40gb | 51.78 | 51.78 | 161 | 2 | 4 | 64 | 11 | 183175 | 22896 | 1048576 | bf16 | PURE | FULL_SHARD | False | False | 370665472 |
|  350m | 8192 | 8 | a100_40gb | 51.39 | 51.39 | 160 | 4 | 4 | 128 | 34 | 279466 | 34933 | 1048576 | bf16 | PURE | FULL_SHARD | False | False | 362276864 |
|  350m | 4096 | 8 | a100_40gb | 50.38 | 50.38 | 157 | 8 | 4 | 256 | 91 | 374670 | 46833 | 1048576 | bf16 | PURE | FULL_SHARD | False | False | 358082560 |
|  350m | 2048 | 128 | a100_40gb | 45.61 | 45.61 | 142 | 18 | 1 | 2304 | 3245 | 6647647 | 51934 | 4718592 | bf16 | PURE | FULL_SHARD | False | False | 355985408 |
|  350m | 2048 | 64 | a100_40gb | 46.27 | 46.27 | 144 | 18 | 1 | 1152 | 1646 | 3372118 | 52689 | 2359296 | bf16 | PURE | FULL_SHARD | False | False | 355985408 |
|  350m | 2048 | 32 | a100_40gb | 47.26 | 47.26 | 147 | 18 | 1 | 576 | 840 | 1721978 | 53811 | 1179648 | bf16 | PURE | FULL_SHARD | False | False | 355985408 |
|  350m | 2048 | 16 | a100_40gb | 48.66 | 48.66 | 151 | 18 | 1 | 288 | 432 | 886622 | 55413 | 589824 | bf16 | PURE | FULL_SHARD | False | False | 355985408 |
|  350m | 2048 | 8 | a100_40gb | 49.17 | 49.17 | 153 | 16 | 4 | 512 | 218 | 447963 | 55995 | 1048576 | bf16 | PURE | FULL_SHARD | False | False | 355985408 |
|  350m | 1024 | 8 | a100_40gb | 48.73 | 48.73 | 152 | 32 | 4 | 1024 | 488 | 500184 | 62523 | 1048576 | bf16 | PURE | FULL_SHARD | False | False | 354936832 |
|  350m | 512 | 8 | a100_40gb | 48.39 | 48.39 | 150 | 64 | 4 | 2048 | 1035 | 530277 | 66284 | 1048576 | bf16 | PURE | FULL_SHARD | False | False | 354412544 |
|  125m | 32768 | 8 | a100_40gb | 47.27 | 47.27 | 147 | 1 | 4 | 32 | 7 | 261208 | 32651 | 1048576 | bf16 | PURE | FULL_SHARD | False | False | 148904448 |
|  125m | 16384 | 8 | a100_40gb | 46.77 | 46.77 | 145 | 2 | 3 | 48 | 27 | 443876 | 55484 | 786432 | bf16 | PURE | FULL_SHARD | False | False | 136321536 |
|  125m | 8192 | 8 | a100_40gb | 46.94 | 46.94 | 146 | 5 | 3 | 120 | 84 | 694868 | 86858 | 983040 | bf16 | PURE | FULL_SHARD | False | False | 130030080 |
|  125m | 4096 | 8 | a100_40gb | 44.82 | 44.82 | 139 | 13 | 3 | 312 | 224 | 921297 | 115162 | 1277952 | bf16 | PURE | FULL_SHARD | False | False | 126884352 |
|  125m | 2048 | 128 | a100_40gb | 38.86 | 38.86 | 121 | 26 | 1 | 3328 | 7746 | 15863837 | 123936 | 6815744 | bf16 | PURE | FULL_SHARD | False | False | 125311488 |
|  125m | 2048 | 64 | a100_40gb | 39.27 | 39.27 | 122 | 26 | 1 | 1664 | 3913 | 8015010 | 125234 | 3407872 | bf16 | PURE | FULL_SHARD | False | False | 125311488 |
|  125m | 2048 | 32 | a100_40gb | 39.86 | 39.86 | 124 | 26 | 1 | 832 | 1986 | 4067922 | 127122 | 1703936 | bf16 | PURE | FULL_SHARD | False | False | 125311488 |
|  125m | 2048 | 16 | a100_40gb | 40.93 | 40.93 | 127 | 26 | 1 | 416 | 1019 | 2088560 | 130535 | 851968 | bf16 | PURE | FULL_SHARD | False | False | 125311488 |
|  125m | 2048 | 8 | a100_40gb | 42.75 | 42.75 | 133 | 26 | 3 | 624 | 532 | 1090678 | 136334 | 1277952 | bf16 | PURE | FULL_SHARD | False | False | 125311488 |
|  125m | 1024 | 8 | a100_40gb | 40.89 | 40.89 | 127 | 52 | 3 | 1248 | 1158 | 1186314 | 148289 | 1277952 | bf16 | PURE | FULL_SHARD | False | False | 124525056 |
|  125m | 512 | 8 | a100_40gb | 40.26 | 40.26 | 125 | 104 | 3 | 2496 | 2448 | 1253886 | 156735 | 1277952 | bf16 | PURE | FULL_SHARD | False | False | 124131840 |
