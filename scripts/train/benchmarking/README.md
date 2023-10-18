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
> The `collect_results.py` will by default find all runs with `tput` in the run name. To customize this project tag, use `--project` in both the submissing and collection scripts.


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

Note that these are approximations. Actual HFU would be higher since it includes the floating point operations for normalization, activation, and residual lyaers, as well as **all** recomputation. For example, our models use Flash Attention, which requires including an extra recompute factor for its recomputation in the forward pass. Therefore, the attention multipler would be 5 instead of 4.

## Results

Below we include several configurations across different hardware platforms, sequence lengths and batch sizes. It is easy to benchmark configurations for your own use case. For example, using the Mosaic platform, to test MPT {13B, 30B} using fp16 with a batch size of 2M tokens and seq len {2k, 4k, 8k, 16k} run:
```
python submit_benchmarks.py -m 13b.yaml 30b.yaml -t fp16 -b 21 21 -s 11 14 --RUN
```
This will run 8 configs for 12 steps to get throughput numbers. `python collect_results.py` can then be used to parse all output training logs and create the tables below.

Our microbatching engine enables microbatch sizes that do not divde Global Batchsize while being mathematically faithful to the global batch size. For example, a total batch size of 48, and a micro batch of 11, means we will accumulate gradients across microbatches of 11, 11, 11, 11, 4.

[comment]: # (TODO: Update tables with torch 2.0 after next Composer release)

## H100 80GB BF16
|  Model | SeqLen (T) | # GPUs | GPU | MFU | HFU | Model TFLOP | MicroBatchSize | GradAccum | GlobalBatchSize | Throughput (S/s) | Throughput (T/s) | Throughput (T/s/GPU) | GlobalBatchSize (T) | Precision | MP Mode | Sharding Strategy | Activation Checkpointing | Activation CPUOffload | NumParams |
|  --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  70b | 2048 | 64 | h100_80gb | 42.57 | 56.76 |  4.212583 | 8 | 4 | 2048 | 32 | 66523 | 1039 | 4194304 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 64862437376 |
|  70b | 2048 | 32 | h100_80gb | 36.15 | 48.2 |  3.576911 | 2 | 16 | 1024 | 13 | 28242 | 882 | 2097152 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 64862437376 |
|  30b | 8192 | 8 | h100_80gb | 29.92 | 39.9 |  2.961077 | 1 | 21 | 168 | 1 | 11072 | 1384 | 1376256 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 30019254272 |
|  30b | 4096 | 8 | h100_80gb | 35.86 | 47.81 |  3.548107 | 1 | 21 | 168 | 3 | 14419 | 1802 | 688128 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 29989894144 |
|  30b | 2048 | 32 | h100_80gb | 43.92 | 58.57 |  4.346371 | 14 | 3 | 1344 | 36 | 73860 | 2308 | 2752512 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 29975214080 |
|  30b | 2048 | 16 | h100_80gb | 43.07 | 57.42 |  4.261622 | 10 | 3 | 480 | 17 | 36209 | 2263 | 983040 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 29975214080 |
|  30b | 2048 | 8 | h100_80gb | 38.11 | 50.82 |  3.771361 | 3 | 21 | 504 | 7 | 16022 | 2002 | 1032192 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 29975214080 |
|  30b | 1024 | 8 | h100_80gb | 38.76 | 51.68 |  3.835386 | 6 | 21 | 1008 | 16 | 16672 | 2084 | 1032192 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 29967874048 |
|  13b | 32768 | 8 | h100_80gb | 31.68 | 42.24 |  3.134795 | 1 | 3 | 24 | 0 | 15812 | 1976 | 786432 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 13011240960 |
|  13b | 16384 | 8 | h100_80gb | 35.55 | 47.4 |  3.517379 | 3 | 3 | 72 | 1 | 23881 | 2985 | 1179648 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 12927354880 |
|  13b | 4096 | 8 | h100_80gb | 41.6 | 55.47 |  4.116250 | 10 | 3 | 240 | 9 | 37740 | 4717 | 983040 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 12864440320 |
|  13b | 2048 | 64 | h100_80gb | 39.86 | 39.86 |  3.943653 | 2 | 1 | 128 | 150 | 307209 | 4800 | 262144 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 12853954560 |
|  13b | 2048 | 32 | h100_80gb | 39.95 | 39.95 |  3.952796 | 2 | 1 | 64 | 75 | 153960 | 4811 | 131072 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 12853954560 |
|  13b | 2048 | 16 | h100_80gb | 39.58 | 39.58 |  3.916861 | 2 | 1 | 32 | 37 | 76280 | 4767 | 65536 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 12853954560 |
|  13b | 2048 | 8 | h100_80gb | 39.79 | 39.79 |  3.937055 | 2 | 1 | 16 | 18 | 38336 | 4792 | 32768 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 12853954560 |
|  13b | 1024 | 8 | h100_80gb | 44.27 | 59.03 |  4.380472 | 40 | 3 | 960 | 42 | 44019 | 5502 | 983040 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 12848711680 |
|  7b | 65536 | 8 | h100_80gb | 28.59 | 38.13 |  2.829422 | 1 | 2 | 16 | 0 | 15654 | 1956 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 6918905856 |
|  7b | 32768 | 8 | h100_80gb | 30.94 | 41.25 |  3.061503 | 2 | 2 | 32 | 0 | 26550 | 3318 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 6784688128 |
|  7b | 8192 | 8 | h100_80gb | 37.14 | 49.52 |  3.674893 | 8 | 2 | 128 | 6 | 55481 | 6935 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 6684024832 |
|  7b | 4096 | 8 | h100_80gb | 40.42 | 53.9 |  3.999793 | 16 | 2 | 256 | 16 | 68893 | 8611 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 6667247616 |
|  7b | 2048 | 8 | h100_80gb | 46.44 | 46.44 |  4.595096 | 6 | 1 | 48 | 41 | 85144 | 10643 | 98304 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 6658859008 |
|  7b | 1024 | 8 | h100_80gb | 42.83 | 57.11 |  4.238424 | 64 | 2 | 1024 | 79 | 81628 | 10203 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 6654664704 |
|  3b | 65536 | 8 | h100_80gb | 26.81 | 35.74 |  2.652666 | 1 | 2 | 16 | 0 | 26099 | 3262 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 2814366720 |
|  3b | 32768 | 8 | h100_80gb | 28.84 | 38.46 |  2.854036 | 3 | 6 | 144 | 1 | 46984 | 5873 | 4718592 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 2730480640 |
|  3b | 16384 | 8 | h100_80gb | 36.34 | 36.34 |  3.595418 | 1 | 6 | 48 | 5 | 89223 | 11152 | 786432 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 2688537600 |
|  3b | 8192 | 8 | h100_80gb | 40.31 | 40.31 |  3.988482 | 3 | 6 | 144 | 16 | 132626 | 16578 | 1179648 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 2667566080 |
|  3b | 4096 | 8 | h100_80gb | 42.31 | 42.31 |  4.186327 | 5 | 6 | 240 | 40 | 167712 | 20964 | 983040 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 2657080320 |
|  3b | 2048 | 64 | h100_80gb | 40.8 | 40.8 |  4.037625 | 6 | 3 | 1152 | 703 | 1441663 | 22525 | 2359296 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 2651837440 |
|  3b | 2048 | 32 | h100_80gb | 41.7 | 41.7 |  4.126519 | 6 | 3 | 576 | 359 | 736701 | 23021 | 1179648 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 2651837440 |
|  3b | 2048 | 16 | h100_80gb | 43.73 | 43.73 |  4.327437 | 10 | 3 | 480 | 188 | 386285 | 24142 | 983040 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 2651837440 |
|  3b | 1024 | 8 | h100_80gb | 46.2 | 46.2 |  4.571338 | 20 | 6 | 960 | 211 | 216369 | 27046 | 983040 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 2649216000 |
|  3b | 512 | 8 | h100_80gb | 46.32 | 46.32 |  4.583697 | 40 | 6 | 1920 | 436 | 223721 | 27965 | 983040 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 2647905280 |
|  1b | 65536 | 8 | h100_80gb | 26.34 | 35.12 |  2.606140 | 1 | 2 | 16 | 0 | 44050 | 5506 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 1445974016 |
|  1b | 32768 | 8 | h100_80gb | 33.54 | 33.54 |  3.319089 | 1 | 4 | 32 | 2 | 96203 | 12025 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 1378865152 |
|  1b | 16384 | 8 | h100_80gb | 35.22 | 35.22 |  3.484904 | 2 | 4 | 64 | 9 | 157194 | 19649 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 1345310720 |
|  1b | 8192 | 8 | h100_80gb | 37.73 | 37.73 |  3.732998 | 3 | 4 | 96 | 28 | 233256 | 29157 | 786432 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 1328533504 |
|  1b | 4096 | 8 | h100_80gb | 40.26 | 40.26 |  3.983310 | 7 | 4 | 224 | 75 | 308282 | 38535 | 917504 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 1320144896 |
|  1b | 2048 | 64 | h100_80gb | 40.85 | 40.85 |  4.042246 | 20 | 1 | 1280 | 1387 | 2841754 | 44402 | 2621440 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 1315950592 |
|  1b | 2048 | 32 | h100_80gb | 41.52 | 41.52 |  4.108550 | 20 | 1 | 640 | 705 | 1444183 | 45130 | 1310720 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 1315950592 |
|  1b | 2048 | 16 | h100_80gb | 42.36 | 42.36 |  4.191080 | 20 | 1 | 320 | 359 | 736596 | 46037 | 655360 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 1315950592 |
|  1b | 2048 | 8 | h100_80gb | 41.82 | 41.82 |  4.138134 | 14 | 1 | 112 | 177 | 363645 | 45455 | 229376 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 1315950592 |
|  1b | 1024 | 8 | h100_80gb | 41.95 | 41.95 |  4.151125 | 18 | 4 | 576 | 382 | 391287 | 48910 | 589824 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 1313853440 |
|  1b | 512 | 8 | h100_80gb | 43.21 | 43.21 |  4.275497 | 56 | 4 | 1792 | 816 | 418201 | 52275 | 917504 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 1312804864 |
|  760m | 32768 | 8 | h100_80gb | 31.84 | 31.84 |  3.151039 | 1 | 2 | 16 | 3 | 130333 | 16291 | 524288 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 807656448 |
|  760m | 16384 | 8 | h100_80gb | 33.57 | 33.57 |  3.321888 | 3 | 2 | 48 | 13 | 222521 | 27815 | 786432 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 782490624 |
|  760m | 8192 | 8 | h100_80gb | 34.84 | 34.84 |  3.447797 | 6 | 2 | 96 | 40 | 334602 | 41825 | 786432 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 769907712 |
|  760m | 4096 | 8 | h100_80gb | 35.83 | 35.83 |  3.545866 | 12 | 2 | 192 | 108 | 443674 | 55459 | 786432 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 763616256 |
|  760m | 2048 | 32 | h100_80gb | 37.57 | 37.57 |  3.717227 | 24 | 1 | 768 | 1062 | 2175091 | 67971 | 1572864 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 760470528 |
|  760m | 2048 | 16 | h100_80gb | 37.89 | 37.89 |  3.748925 | 24 | 1 | 384 | 535 | 1096819 | 68551 | 786432 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 760470528 |
|  760m | 2048 | 8 | h100_80gb | 34.9 | 34.9 |  3.453389 | 24 | 2 | 384 | 246 | 505177 | 63147 | 786432 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 760470528 |
|  760m | 1024 | 8 | h100_80gb | 39.76 | 39.76 |  3.934058 | 48 | 2 | 768 | 613 | 628648 | 78581 | 786432 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 758897664 |
|  760m | 512 | 8 | h100_80gb | 40.42 | 40.42 |  3.999190 | 96 | 2 | 1536 | 1308 | 669998 | 83749 | 786432 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 758111232 |

## H100 80GB FP8
|  Model | SeqLen (T) | # GPUs | GPU | MFU | HFU | Model TFLOP | MicroBatchSize | GradAccum | GlobalBatchSize | Throughput (S/s) | Throughput (T/s) | Throughput (T/s/GPU) | GlobalBatchSize (T) | Precision | MP Mode | Sharding Strategy | Activation Checkpointing | Activation CPUOffload | NumParams |
|  --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  3b | 32768 | 8 | h100_80gb | 14.38 | 19.18 |  2.846055 | 3 | 6 | 144 | 1 | 46853 | 5856 | 4718592 | amp_fp8 | DEFAULT | FULL_SHARD | True | False | 2730480640 |
|  3b | 8192 | 8 | h100_80gb | 23.28 | 23.28 |  4.606436 | 3 | 6 | 144 | 18 | 153174 | 19146 | 1179648 | amp_fp8 | DEFAULT | FULL_SHARD | False | False | 2667566080 |
|  3b | 2048 | 8 | h100_80gb | 27.7 | 27.7 |  5.482421 | 10 | 6 | 480 | 119 | 244692 | 30586 | 983040 | amp_fp8 | DEFAULT | FULL_SHARD | False | False | 2651837440 |
|  3b | 512 | 8 | h100_80gb | 30.25 | 30.25 |  5.987086 | 40 | 6 | 1920 | 570 | 292217 | 36527 | 983040 | amp_fp8 | DEFAULT | FULL_SHARD | False | False | 2647905280 |
|  1b | 32768 | 8 | h100_80gb | 17.55 | 17.55 |  3.472285 | 1 | 4 | 32 | 3 | 100643 | 12580 | 1048576 | amp_fp8 | DEFAULT | FULL_SHARD | False | False | 1378865152 |
|  1b | 8192 | 8 | h100_80gb | 20.71 | 20.71 |  4.098371 | 2 | 4 | 64 | 31 | 256087 | 32010 | 524288 | amp_fp8 | DEFAULT | FULL_SHARD | False | False | 1328533504 |
|  1b | 512 | 8 | h100_80gb | 29.06 | 29.06 |  5.750972 | 56 | 4 | 1792 | 1098 | 562523 | 70315 | 917504 | amp_fp8 | DEFAULT | FULL_SHARD | False | False | 1312804864 |

## A100 80GB with 1600 Gbps node-node interconnect (RoCE)

|  Model | SeqLen (T) | # GPUs | GPU | MFU | HFU | Model TFLOP | MicroBatchSize | GradAccum | GlobalBatchSize | Throughput (S/s) | Throughput (T/s) | Throughput (T/s/GPU) | GlobalBatchSize (T) | Precision | MP Mode | Sharding Strategy | Activation Checkpointing | Activation CPUOffload | NumParams |
|  --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  70b | 2048 | 64 | a100_80gb | 53.33 | 71.1 | 1.663896 | 8 | 4 | 2048 | 12 | 26274 | 410 | 4194304 | bf16 | PURE | FULL_SHARD | True | False | 64862437376 |
|  70b | 2048 | 32 | a100_80gb | 48.56 | 64.75 | 1.515072 | 2 | 16 | 1024 | 5 | 11962 | 373 | 2097152 | bf16 | PURE | FULL_SHARD | True | False | 64862437376 |
|  30b | 8192 | 8 | a100_80gb | 39.38 | 52.5 |  1.228583 | 1 | 21 | 168 | 0 | 4594 | 574 | 1376256 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 30019254272 |
|  30b | 4096 | 8 | a100_80gb | 51.37 | 68.49 |  1.602714 | 1 | 21 | 168 | 1 | 6513 | 814 | 688128 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 29989894144 |
|  30b | 2048 | 8 | a100_80gb | 55.3 | 73.74 |  1.725420 | 3 | 21 | 504 | 3 | 7330 | 916 | 1032192 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 29975214080 |
|  30b | 1024 | 8 | a100_80gb | 55.82 | 74.43 |  1.741706 | 6 | 21 | 1008 | 7 | 7571 | 946 | 1032192 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 29967874048 |
|  30b | 512 | 8 | a100_80gb | 56.4 | 75.2 |  1.759659 | 12 | 21 | 2016 | 15 | 7739 | 967 | 1032192 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 29964204032 |
|  13b | 32768 | 8 | a100_80gb | 51.69 | 68.92 |  1.612741 | 1 | 3 | 24 | 0 | 8134 | 1016 | 786432 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 13011240960 |
|  13b | 16384 | 8 | a100_80gb | 54.07 | 72.1 |  1.687104 | 3 | 3 | 72 | 0 | 11454 | 1431 | 1179648 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 12927354880 |
|  13b | 8192 | 8 | a100_80gb | 56.07 | 74.76 |  1.749500 | 5 | 3 | 120 | 1 | 14362 | 1795 | 983040 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 12885411840 |
|  13b | 4096 | 8 | a100_80gb | 57.62 | 76.82 |  1.797680 | 10 | 3 | 240 | 4 | 16482 | 2060 | 983040 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 12864440320 |
|  13b | 2048 | 8 | a100_80gb | 59.57 | 59.57 |  1.858556 | 2 | 3 | 48 | 8 | 18097 | 2262 | 98304 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 12853954560 |
|  13b | 1024 | 8 | a100_80gb | 59.48 | 79.3 |  1.855662 | 40 | 3 | 960 | 18 | 18647 | 2330 | 983040 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 12848711680 |
|  7b | 65536 | 8 | a100_80gb | 46.97 | 62.63 |  1.465467 | 1 | 2 | 16 | 0 | 8108 | 1013 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 6918905856 |
|  7b | 32768 | 8 | a100_80gb | 49.46 | 65.94 |  1.543088 | 2 | 2 | 32 | 0 | 13382 | 1672 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 6784688128 |
|  7b | 16384 | 8 | a100_80gb | 51.96 | 69.28 |  1.621241 | 4 | 2 | 64 | 1 | 19629 | 2453 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 6717579264 |
|  7b | 8192 | 8 | a100_80gb | 54.47 | 72.62 |  1.699318 | 8 | 2 | 128 | 3 | 25655 | 3206 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 6684024832 |
|  7b | 4096 | 8 | a100_80gb | 54.84 | 73.12 |  1.711072 | 16 | 2 | 256 | 7 | 29472 | 3684 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 6667247616 |
|  7b | 2048 | 8 | a100_80gb | 64.23 | 64.23 |  2.003836 | 6 | 2 | 96 | 18 | 37130 | 4641 | 196608 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 6658859008 |
|  7b | 1024 | 8 | a100_80gb | 58.01 | 77.35 |  1.809897 | 64 | 2 | 1024 | 34 | 34857 | 4357 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 6654664704 |
|  3b | 65536 | 8 | a100_80gb | 46.05 | 61.41 |  1.436905 | 1 | 2 | 16 | 0 | 14137 | 1767 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 2814366720 |
|  3b | 32768 | 8 | a100_80gb | 47.18 | 62.91 |  1.472162 | 3 | 6 | 144 | 0 | 24235 | 3029 | 4718592 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 2730480640 |
|  3b | 16384 | 8 | a100_80gb | 57.13 | 57.13 |  1.782465 | 1 | 6 | 48 | 2 | 44233 | 5529 | 786432 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 2688537600 |
|  3b | 8192 | 8 | a100_80gb | 59.34 | 59.34 |  1.851516 | 3 | 6 | 144 | 7 | 61567 | 7695 | 1179648 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 2667566080 |
|  3b | 4096 | 8 | a100_80gb | 60.53 | 60.53 |  1.888534 | 5 | 6 | 240 | 18 | 75658 | 9457 | 983040 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 2657080320 |
|  3b | 2048 | 8 | a100_80gb | 62.11 | 62.11 |  1.937863 | 10 | 2 | 160 | 42 | 86491 | 10811 | 327680 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 2651837440 |
|  3b | 1024 | 8 | a100_80gb | 62.73 | 62.73 |  1.957319 | 20 | 6 | 960 | 90 | 92643 | 11580 | 983040 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 2649216000 |
|  3b | 512 | 8 | a100_80gb | 63.71 | 63.71 |  1.987776 | 40 | 6 | 1920 | 189 | 97019 | 12127 | 983040 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 2647905280 |
|  1b | 65536 | 8 | a100_80gb | 46.18 | 61.57 |  1.440850 | 1 | 2 | 16 | 0 | 24353 | 3044 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 1445974016 |
|  1b | 32768 | 8 | a100_80gb | 55.52 | 55.52 |  1.732203 | 1 | 4 | 32 | 1 | 50207 | 6275 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 1378865152 |
|  1b | 16384 | 8 | a100_80gb | 56.6 | 56.6 |  1.765798 | 2 | 4 | 64 | 4 | 79650 | 9956 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 1345310720 |
|  1b | 8192 | 8 | a100_80gb | 56.69 | 56.69 |  1.768689 | 3 | 4 | 96 | 13 | 110516 | 13814 | 786432 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 1328533504 |
|  1b | 4096 | 8 | a100_80gb | 59.0 | 59.0 |  1.840694 | 7 | 4 | 224 | 34 | 142457 | 17807 | 917504 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 1320144896 |
|  1b | 2048 | 8 | a100_80gb | 59.86 | 59.86 |  1.867501 | 14 | 4 | 448 | 80 | 164109 | 20513 | 917504 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 1315950592 |
|  1b | 1024 | 8 | a100_80gb | 60.15 | 60.15 |  1.876694 | 18 | 4 | 576 | 172 | 176898 | 22112 | 589824 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 1313853440 |
|  1b | 512 | 8 | a100_80gb | 60.68 | 60.68 |  1.893257 | 56 | 4 | 1792 | 361 | 185186 | 23148 | 917504 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 1312804864 |
|  760m | 65536 | 8 | a100_80gb | 45.34 | 60.45 |  1.414636 | 1 | 2 | 16 | 0 | 33150 | 4143 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 857988096 |
|  760m | 32768 | 8 | a100_80gb | 54.57 | 54.57 |  1.702462 | 1 | 2 | 16 | 2 | 70417 | 8802 | 524288 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 807656448 |
|  760m | 16384 | 8 | a100_80gb | 54.64 | 54.64 |  1.704797 | 3 | 2 | 48 | 6 | 114198 | 14274 | 786432 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 782490624 |
|  760m | 8192 | 8 | a100_80gb | 55.31 | 55.31 |  1.725649 | 6 | 2 | 96 | 20 | 167471 | 20933 | 786432 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 769907712 |
|  760m | 4096 | 8 | a100_80gb | 56.05 | 56.05 |  1.748726 | 12 | 2 | 192 | 53 | 218808 | 27351 | 786432 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 763616256 |
|  760m | 2048 | 8 | a100_80gb | 56.85 | 56.85 |  1.773749 | 24 | 2 | 384 | 126 | 259472 | 32434 | 786432 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 760470528 |
|  760m | 1024 | 8 | a100_80gb | 47.76 | 47.76 |  1.490159 | 48 | 2 | 768 | 232 | 238122 | 29765 | 786432 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 758897664 |
|  760m | 512 | 8 | a100_80gb | 45.07 | 45.07 |  1.406117 | 96 | 2 | 1536 | 460 | 235571 | 29446 | 786432 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 758111232 |
|  350m | 65536 | 8 | a100_80gb | 52.7 | 52.7 |  1.644340 | 1 | 2 | 16 | 0 | 60195 | 7524 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 420997120 |
|  350m | 32768 | 8 | a100_80gb | 52.46 | 52.46 |  1.636742 | 2 | 2 | 32 | 3 | 109222 | 13652 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 387442688 |
|  350m | 16384 | 8 | a100_80gb | 53.28 | 53.28 |  1.662338 | 4 | 2 | 64 | 11 | 188478 | 23559 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 370665472 |
|  350m | 8192 | 8 | a100_80gb | 53.8 | 53.8 |  1.678405 | 8 | 2 | 128 | 35 | 292559 | 36569 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 362276864 |
|  350m | 4096 | 8 | a100_80gb | 53.31 | 53.31 |  1.663304 | 16 | 2 | 256 | 96 | 396442 | 49555 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 358082560 |
|  350m | 2048 | 8 | a100_80gb | 51.62 | 51.62 |  1.610589 | 32 | 2 | 512 | 229 | 470263 | 58782 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 355985408 |
|  350m | 1024 | 8 | a100_80gb | 50.51 | 50.51 |  1.576003 | 64 | 2 | 1024 | 506 | 518504 | 64813 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 354936832 |
|  350m | 512 | 8 | a100_80gb | 50.61 | 50.61 |  1.578979 | 128 | 2 | 2048 | 1083 | 554643 | 69330 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 354412544 |
|  125m | 65536 | 8 | a100_80gb | 54.13 | 54.13 |  1.688979 | 1 | 2 | 16 | 2 | 162946 | 20368 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 174070272 |
|  125m | 32768 | 8 | a100_80gb | 52.71 | 52.71 |  1.644619 | 2 | 2 | 32 | 8 | 291256 | 36407 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 148904448 |
|  125m | 16384 | 8 | a100_80gb | 50.61 | 50.61 |  1.578980 | 4 | 2 | 64 | 29 | 480322 | 60040 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 136321536 |
|  125m | 8192 | 8 | a100_80gb | 48.85 | 48.85 |  1.524158 | 8 | 2 | 128 | 88 | 723142 | 90392 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 130030080 |
|  125m | 4096 | 8 | a100_80gb | 46.08 | 46.08 |  1.437679 | 16 | 2 | 256 | 231 | 947172 | 118396 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 126884352 |
|  125m | 2048 | 8 | a100_80gb | 44.79 | 44.79 |  1.397395 | 40 | 2 | 640 | 557 | 1142641 | 142830 | 1310720 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 125311488 |
|  125m | 2048 | 8 | a100_80gb | 44.45 | 44.45 |  1.386706 | 32 | 2 | 512 | 553 | 1133901 | 141737 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 125311488 |
|  125m | 1024 | 8 | a100_80gb | 43.15 | 43.15 |  1.346253 | 64 | 2 | 1024 | 1222 | 1251751 | 156468 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 124525056 |
|  125m | 512 | 8 | a100_80gb | 42.56 | 42.56 |  1.327798 | 128 | 2 | 2048 | 2588 | 1325455 | 165681 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 124131840 |

## A100 40GB with 1600 Gbps node-node interconnect (RoCE)

|  Model | SeqLen (T) | # GPUs | GPU | MFU | HFU | MicroBatchSize | GradAccum | GlobalBatchSize | Throughput (S/s) | Throughput (T/s) | Throughput (T/s/GPU) | GlobalBatchSize (T) | Precision | MP Mode | Sharding Strategy | Activation Checkpointing | Activation CPUOffload | NumParams |
|  --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  70b | 2048 | 128 | a100_40gb | 48.91 | 65.21 |  1.525992 | 4 | 1 | 512 | 23 | 48194 | 376 | 1048576 | bf16 | PURE | FULL_SHARD | True | False | 64862437376 |
|  70b | 2048 | 64 | a100_40gb | 35.87 | 47.82 |  1.119144 | 2 | 1 | 128 | 8 | 17672 | 276 | 262144 | bf16 | PURE | FULL_SHARD | True | False | 64862437376 |
|  30b | 2048 | 128 | a100_40gb | 52.25 | 69.66 |  1.630200 | 6 | 1 | 768 | 54 | 110803 | 865 | 1572864 | bf16 | PURE | FULL_SHARD | True | False | 29975214080 |
|  30b | 2048 | 32 | a100_40gb | 51.74 | 68.98 |  1.614288 | 4 | 1 | 128 | 13 | 27431 | 857 | 262144 | bf16 | PURE | FULL_SHARD | True | False | 29975214080 |
|  13b | 8192 | 8 | a100_40gb | 43.95 | 58.6 |  1.371240 | 1 | 16 | 128 | 1 | 11258 | 1407 | 1048576 | bf16 | PURE | FULL_SHARD | True | False | 12885411840 |
|  13b | 4096 | 8 | a100_40gb | 44.85 | 59.8 |  1.399320 | 2 | 16 | 256 | 3 | 12830 | 1603 | 1048576 | bf16 | PURE | FULL_SHARD | True | False | 12864440320 |
|  13b | 2048 | 128 | a100_40gb | 51.93 | 69.24 |  1.620216 | 16 | 1 | 2048 | 123 | 252444 | 1972 | 4194304 | bf16 | PURE | FULL_SHARD | True | False | 12853954560 |
|  13b | 2048 | 64 | a100_40gb | 52.04 | 69.39 |  1.623648 | 16 | 1 | 1024 | 61 | 126479 | 1976 | 2097152 | bf16 | PURE | FULL_SHARD | True | False | 12853954560 |
|  13b | 2048 | 32 | a100_40gb | 52.62 | 70.16 |  1.641744 | 14 | 1 | 448 | 31 | 63946 | 1998 | 917504 | bf16 | PURE | FULL_SHARD | True | False | 12853954560 |
|  13b | 2048 | 16 | a100_40gb | 52.5 | 70.0 |  1.638000 | 10 | 1 | 160 | 15 | 31900 | 1993 | 327680 | bf16 | PURE | FULL_SHARD | True | False | 12853954560 |
|  13b | 2048 | 8 | a100_40gb | 43.94 | 58.58 |  1.370928 | 4 | 16 | 512 | 6 | 13347 | 1668 | 1048576 | bf16 | PURE | FULL_SHARD | True | False | 12853954560 |
|  13b | 1024 | 8 | a100_40gb | 44.07 | 58.76 |  1.374984 | 8 | 16 | 1024 | 13 | 13817 | 1727 | 1048576 | bf16 | PURE | FULL_SHARD | True | False | 12848711680 |
|  13b | 512 | 8 | a100_40gb | 44.28 | 59.04 |  1.381536 | 16 | 16 | 2048 | 27 | 14108 | 1763 | 1048576 | bf16 | PURE | FULL_SHARD | True | False | 12846090240 |
|  7b | 16384 | 8 | a100_40gb | 47.65 | 63.53 |  1.486680 | 1 | 4 | 32 | 1 | 17998 | 2249 | 524288 | bf16 | PURE | FULL_SHARD | True | False | 6717579264 |
|  7b | 8192 | 8 | a100_40gb | 49.04 | 65.38 |  1.530048 | 3 | 4 | 96 | 2 | 23098 | 2887 | 786432 | bf16 | PURE | FULL_SHARD | True | False | 6684024832 |
|  7b | 4096 | 8 | a100_40gb | 50.11 | 66.82 |  1.563432 | 6 | 4 | 192 | 6 | 26930 | 3366 | 786432 | bf16 | PURE | FULL_SHARD | True | False | 6667247616 |
|  7b | 2048 | 128 | a100_40gb | 50.14 | 66.85 |  1.564368 | 18 | 1 | 2304 | 226 | 463749 | 3623 | 4718592 | bf16 | PURE | FULL_SHARD | True | False | 6658859008 |
|  7b | 2048 | 64 | a100_40gb | 50.73 | 67.64 |  1.582776 | 18 | 1 | 1152 | 114 | 234614 | 3665 | 2359296 | bf16 | PURE | FULL_SHARD | True | False | 6658859008 |
|  7b | 2048 | 32 | a100_40gb | 51.55 | 68.73 |  1.608360 | 18 | 1 | 576 | 58 | 119202 | 3725 | 1179648 | bf16 | PURE | FULL_SHARD | True | False | 6658859008 |
|  7b | 2048 | 16 | a100_40gb | 50.44 | 67.26 |  1.573728 | 16 | 1 | 256 | 28 | 58322 | 3645 | 524288 | bf16 | PURE | FULL_SHARD | True | False | 6658859008 |
|  7b | 2048 | 8 | a100_40gb | 50.92 | 67.89 |  1.588704 | 12 | 4 | 384 | 14 | 29436 | 3679 | 786432 | bf16 | PURE | FULL_SHARD | True | False | 6658859008 |
|  7b | 1024 | 8 | a100_40gb | 51.31 | 68.42 |  1.600872 | 24 | 4 | 768 | 30 | 30833 | 3854 | 786432 | bf16 | PURE | FULL_SHARD | True | False | 6654664704 |
|  7b | 512 | 8 | a100_40gb | 50.85 | 67.8 |  1.586520 | 48 | 4 | 1536 | 60 | 31167 | 3895 | 786432 | bf16 | PURE | FULL_SHARD | True | False | 6652567552 |
|  3b | 32768 | 8 | a100_40gb | 46.03 | 61.37 |  1.436136 | 1 | 4 | 32 | 0 | 23640 | 2955 | 1048576 | bf16 | PURE | FULL_SHARD | True | False | 2730480640 |
|  3b | 16384 | 8 | a100_40gb | 46.14 | 61.52 |  1.439568 | 2 | 8 | 128 | 2 | 35726 | 4465 | 2097152 | bf16 | PURE | FULL_SHARD | True | False | 2688537600 |
|  3b | 8192 | 8 | a100_40gb | 55.13 | 55.13 |  1.720056 | 1 | 8 | 64 | 6 | 57193 | 7149 | 524288 | bf16 | PURE | FULL_SHARD | False | False | 2667566080 |
|  3b | 4096 | 8 | a100_40gb | 56.18 | 56.18 |  1.752816 | 2 | 8 | 128 | 17 | 70223 | 8777 | 524288 | bf16 | PURE | FULL_SHARD | False | False | 2657080320 |
|  3b | 2048 | 128 | a100_40gb | 54.8 | 54.8 |  1.709760 | 6 | 1 | 768 | 596 | 1220885 | 9538 | 1572864 | bf16 | PURE | FULL_SHARD | False | False | 2651837440 |
|  3b | 2048 | 64 | a100_40gb | 55.94 | 55.94 |  1.745328 | 6 | 1 | 384 | 304 | 623167 | 9736 | 786432 | bf16 | PURE | FULL_SHARD | False | False | 2651837440 |
|  3b | 2048 | 32 | a100_40gb | 56.96 | 56.96 |  1.777152 | 6 | 1 | 192 | 154 | 317261 | 9914 | 393216 | bf16 | PURE | FULL_SHARD | False | False | 2651837440 |
|  3b | 2048 | 16 | a100_40gb | 56.02 | 56.02 |  1.747824 | 5 | 1 | 80 | 76 | 156013 | 9750 | 163840 | bf16 | PURE | FULL_SHARD | False | False | 2651837440 |
|  3b | 2048 | 8 | a100_40gb | 57.82 | 57.82 |  1.803984 | 5 | 8 | 320 | 39 | 80520 | 10065 | 655360 | bf16 | PURE | FULL_SHARD | False | False | 2651837440 |
|  3b | 1024 | 8 | a100_40gb | 58.14 | 58.14 |  1.813968 | 10 | 8 | 640 | 83 | 85854 | 10731 | 655360 | bf16 | PURE | FULL_SHARD | False | False | 2649216000 |
|  3b | 512 | 8 | a100_40gb | 59.49 | 59.49 |  1.856088 | 20 | 8 | 1280 | 176 | 90596 | 11324 | 655360 | bf16 | PURE | FULL_SHARD | False | False | 2647905280 |
|  1b | 32768 | 8 | a100_40gb | 45.07 | 60.1 |  1.406184 | 1 | 4 | 32 | 1 | 40762 | 5095 | 1048576 | bf16 | PURE | FULL_SHARD | True | False | 1378865152 |
|  1b | 16384 | 8 | a100_40gb | 55.23 | 55.23 |  1.723176 | 1 | 8 | 64 | 4 | 77723 | 9715 | 1048576 | bf16 | PURE | FULL_SHARD | False | False | 1345310720 |
|  1b | 8192 | 8 | a100_40gb | 55.29 | 55.29 |  1.725048 | 2 | 8 | 128 | 13 | 107799 | 13474 | 1048576 | bf16 | PURE | FULL_SHARD | False | False | 1328533504 |
|  1b | 4096 | 8 | a100_40gb | 55.85 | 55.85 |  1.742520 | 4 | 8 | 256 | 32 | 134851 | 16856 | 1048576 | bf16 | PURE | FULL_SHARD | False | False | 1320144896 |
|  1b | 2048 | 128 | a100_40gb | 54.41 | 54.41 |  1.697592 | 10 | 1 | 1280 | 1165 | 2386897 | 18647 | 2621440 | bf16 | PURE | FULL_SHARD | False | False | 1315950592 |
|  1b | 2048 | 64 | a100_40gb | 55.44 | 55.44 |  1.729728 | 10 | 1 | 640 | 593 | 1216104 | 19001 | 1310720 | bf16 | PURE | FULL_SHARD | False | False | 1315950592 |
|  1b | 2048 | 32 | a100_40gb | 45.39 | 45.39 |  1.416168 | 10 | 1 | 320 | 243 | 497782 | 15555 | 655360 | bf16 | PURE | FULL_SHARD | False | False | 1315950592 |
|  1b | 2048 | 16 | a100_40gb | 55.69 | 55.69 |  1.737528 | 8 | 1 | 128 | 149 | 305372 | 19085 | 262144 | bf16 | PURE | FULL_SHARD | False | False | 1315950592 |
|  1b | 2048 | 8 | a100_40gb | 56.23 | 56.23 |  1.754376 | 8 | 8 | 512 | 75 | 154171 | 19271 | 1048576 | bf16 | PURE | FULL_SHARD | False | False | 1315950592 |
|  1b | 1024 | 8 | a100_40gb | 57.02 | 57.02 |  1.779024 | 16 | 8 | 1024 | 163 | 167677 | 20959 | 1048576 | bf16 | PURE | FULL_SHARD | False | False | 1313853440 |
|  1b | 512 | 8 | a100_40gb | 57.1 | 57.1 |  1.781520 | 32 | 8 | 2048 | 340 | 174256 | 21782 | 1048576 | bf16 | PURE | FULL_SHARD | False | False | 1312804864 |
|  760m | 32768 | 8 | a100_40gb | 44.53 | 59.37 |  1.389336 | 1 | 4 | 32 | 1 | 57464 | 7183 | 1048576 | bf16 | PURE | FULL_SHARD | True | False | 807656448 |
|  760m | 16384 | 8 | a100_40gb | 53.26 | 53.26 |  1.661712 | 1 | 4 | 32 | 6 | 111316 | 13914 | 524288 | bf16 | PURE | FULL_SHARD | False | False | 782490624 |
|  760m | 8192 | 8 | a100_40gb | 53.12 | 53.12 |  1.657344 | 3 | 4 | 96 | 19 | 160853 | 20106 | 786432 | bf16 | PURE | FULL_SHARD | False | False | 769907712 |
|  760m | 4096 | 8 | a100_40gb | 53.0 | 53.0 |  1.653600 | 6 | 4 | 192 | 50 | 206909 | 25863 | 786432 | bf16 | PURE | FULL_SHARD | False | False | 763616256 |
|  760m | 2048 | 128 | a100_40gb | 50.73 | 50.73 |  1.582776 | 12 | 1 | 1536 | 1808 | 3704382 | 28940 | 3145728 | bf16 | PURE | FULL_SHARD | False | False | 760470528 |
|  760m | 2048 | 64 | a100_40gb | 51.44 | 51.44 |  1.604928 | 12 | 1 | 768 | 917 | 1878030 | 29344 | 1572864 | bf16 | PURE | FULL_SHARD | False | False | 760470528 |
|  760m | 2048 | 32 | a100_40gb | 51.97 | 51.97 |  1.621464 | 12 | 1 | 384 | 463 | 948745 | 29648 | 786432 | bf16 | PURE | FULL_SHARD | False | False | 760470528 |
|  760m | 2048 | 16 | a100_40gb | 51.9 | 51.9 |  1.619280 | 12 | 1 | 192 | 231 | 473723 | 29607 | 393216 | bf16 | PURE | FULL_SHARD | False | False | 760470528 |
|  760m | 2048 | 8 | a100_40gb | 52.89 | 52.89 |  1.650168 | 12 | 4 | 384 | 117 | 241389 | 30173 | 786432 | bf16 | PURE | FULL_SHARD | False | False | 760470528 |
|  760m | 1024 | 8 | a100_40gb | 53.63 | 53.63 |  1.673256 | 24 | 4 | 768 | 261 | 267376 | 33422 | 786432 | bf16 | PURE | FULL_SHARD | False | False | 758897664 |
|  760m | 512 | 8 | a100_40gb | 53.47 | 53.47 |  1.668264 | 48 | 4 | 1536 | 545 | 279504 | 34938 | 786432 | bf16 | PURE | FULL_SHARD | False | False | 758111232 |
|  350m | 32768 | 8 | a100_40gb | 51.55 | 51.55 |  1.608360 | 1 | 4 | 32 | 3 | 107329 | 13416 | 1048576 | bf16 | PURE | FULL_SHARD | False | False | 387442688 |
|  350m | 16384 | 8 | a100_40gb | 51.78 | 51.78 |  1.615536 | 2 | 4 | 64 | 11 | 183175 | 22896 | 1048576 | bf16 | PURE | FULL_SHARD | False | False | 370665472 |
|  350m | 8192 | 8 | a100_40gb | 51.39 | 51.39 |  1.603368 | 4 | 4 | 128 | 34 | 279466 | 34933 | 1048576 | bf16 | PURE | FULL_SHARD | False | False | 362276864 |
|  350m | 4096 | 8 | a100_40gb | 50.38 | 50.38 |  1.571856 | 8 | 4 | 256 | 91 | 374670 | 46833 | 1048576 | bf16 | PURE | FULL_SHARD | False | False | 358082560 |
|  350m | 2048 | 128 | a100_40gb | 45.61 | 45.61 |  1.423032 | 18 | 1 | 2304 | 3245 | 6647647 | 51934 | 4718592 | bf16 | PURE | FULL_SHARD | False | False | 355985408 |
|  350m | 2048 | 64 | a100_40gb | 46.27 | 46.27 |  1.443624 | 18 | 1 | 1152 | 1646 | 3372118 | 52689 | 2359296 | bf16 | PURE | FULL_SHARD | False | False | 355985408 |
|  350m | 2048 | 32 | a100_40gb | 47.26 | 47.26 |  1.474512 | 18 | 1 | 576 | 840 | 1721978 | 53811 | 1179648 | bf16 | PURE | FULL_SHARD | False | False | 355985408 |
|  350m | 2048 | 16 | a100_40gb | 48.66 | 48.66 |  1.518192 | 18 | 1 | 288 | 432 | 886622 | 55413 | 589824 | bf16 | PURE | FULL_SHARD | False | False | 355985408 |
|  350m | 2048 | 8 | a100_40gb | 49.17 | 49.17 |  1.534104 | 16 | 4 | 512 | 218 | 447963 | 55995 | 1048576 | bf16 | PURE | FULL_SHARD | False | False | 355985408 |
|  350m | 1024 | 8 | a100_40gb | 48.73 | 48.73 |  1.520376 | 32 | 4 | 1024 | 488 | 500184 | 62523 | 1048576 | bf16 | PURE | FULL_SHARD | False | False | 354936832 |
|  350m | 512 | 8 | a100_40gb | 48.39 | 48.39 |  1.509768 | 64 | 4 | 2048 | 1035 | 530277 | 66284 | 1048576 | bf16 | PURE | FULL_SHARD | False | False | 354412544 |
|  125m | 32768 | 8 | a100_40gb | 47.27 | 47.27 |  1.474824 | 1 | 4 | 32 | 7 | 261208 | 32651 | 1048576 | bf16 | PURE | FULL_SHARD | False | False | 148904448 |
|  125m | 16384 | 8 | a100_40gb | 46.77 | 46.77 |  1.459224 | 2 | 3 | 48 | 27 | 443876 | 55484 | 786432 | bf16 | PURE | FULL_SHARD | False | False | 136321536 |
|  125m | 8192 | 8 | a100_40gb | 46.94 | 46.94 |  1.464528 | 5 | 3 | 120 | 84 | 694868 | 86858 | 983040 | bf16 | PURE | FULL_SHARD | False | False | 130030080 |
|  125m | 4096 | 8 | a100_40gb | 44.82 | 44.82 |  1.398384 | 13 | 3 | 312 | 224 | 921297 | 115162 | 1277952 | bf16 | PURE | FULL_SHARD | False | False | 126884352 |
|  125m | 2048 | 128 | a100_40gb | 38.86 | 38.86 |  1.212432 | 26 | 1 | 3328 | 7746 | 15863837 | 123936 | 6815744 | bf16 | PURE | FULL_SHARD | False | False | 125311488 |
|  125m | 2048 | 64 | a100_40gb | 39.27 | 39.27 |  1.225224 | 26 | 1 | 1664 | 3913 | 8015010 | 125234 | 3407872 | bf16 | PURE | FULL_SHARD | False | False | 125311488 |
|  125m | 2048 | 32 | a100_40gb | 39.86 | 39.86 |  1.243632 | 26 | 1 | 832 | 1986 | 4067922 | 127122 | 1703936 | bf16 | PURE | FULL_SHARD | False | False | 125311488 |
|  125m | 2048 | 16 | a100_40gb | 40.93 | 40.93 |  1.277016 | 26 | 1 | 416 | 1019 | 2088560 | 130535 | 851968 | bf16 | PURE | FULL_SHARD | False | False | 125311488 |
|  125m | 2048 | 8 | a100_40gb | 42.75 | 42.75 |  1.333800 | 26 | 3 | 624 | 532 | 1090678 | 136334 | 1277952 | bf16 | PURE | FULL_SHARD | False | False | 125311488 |
|  125m | 1024 | 8 | a100_40gb | 40.89 | 40.89 |  1.275768 | 52 | 3 | 1248 | 1158 | 1186314 | 148289 | 1277952 | bf16 | PURE | FULL_SHARD | False | False | 124525056 |
|  125m | 512 | 8 | a100_40gb | 40.26 | 40.26 |  1.256112 | 104 | 3 | 2496 | 2448 | 1253886 | 156735 | 1277952 | bf16 | PURE | FULL_SHARD | False | False | 124131840 |
