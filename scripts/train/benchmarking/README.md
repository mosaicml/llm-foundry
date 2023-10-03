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

## H100 80GB
|  Model | SeqLen (T) | # GPUs | GPU | MFU | HFU | MicroBatchSize | GradAccum | GlobalBatchSize | Throughput (S/s) | Throughput (T/s) | Throughput (T/s/GPU) | GlobalBatchSize (T) | Precision | MP Mode | Sharding Strategy | Activation Checkpointing | Activation CPUOffload | NumParams |
|  --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  30b | 8192 | 8 | h100_80gb | 27.87 | 37.16 | 1 | 21 | 168 | 1 | 10311 | 1288 | 1376256 | amp_bf16 | PURE | FULL_SHARD | True | False | 30019254272 |
|  30b | 4096 | 8 | h100_80gb | 34.61 | 46.15 | 1 | 21 | 168 | 3 | 13917 | 1739 | 688128 | amp_bf16 | PURE | FULL_SHARD | True | False | 29989894144 |
|  30b | 2048 | 8 | h100_80gb | 37.54 | 50.05 | 3 | 21 | 504 | 7 | 15781 | 1972 | 1032192 | amp_bf16 | PURE | FULL_SHARD | True | False | 29975214080 |
|  30b | 1024 | 8 | h100_80gb | 38.21 | 50.94 | 6 | 21 | 1008 | 16 | 16433 | 2054 | 1032192 | amp_bf16 | PURE | FULL_SHARD | True | False | 29967874048 |
|  30b | 512 | 8 | h100_80gb | 38.64 | 51.52 | 12 | 21 | 2016 | 32 | 16816 | 2102 | 1032192 | amp_bf16 | PURE | FULL_SHARD | True | False | 29964204032 |
|  13b | 32768 | 8 | h100_80gb | 30.73 | 40.97 | 1 | 3 | 24 | 0 | 15338 | 1917 | 786432 | amp_bf16 | PURE | FULL_SHARD | True | False | 13011240960 |
|  13b | 8192 | 8 | h100_80gb | 37.15 | 49.53 | 5 | 3 | 120 | 3 | 30179 | 3772 | 983040 | amp_bf16 | PURE | FULL_SHARD | True | False | 12885411840 |
|  13b | 2048 | 8 | h100_80gb | 41.29 | 55.05 | 20 | 3 | 480 | 19 | 39779 | 4972 | 983040 | amp_bf16 | PURE | FULL_SHARD | True | False | 12853954560 |
|  13b | 512 | 8 | h100_80gb | 42.63 | 56.83 | 80 | 3 | 1920 | 84 | 43074 | 5384 | 983040 | amp_bf16 | PURE | FULL_SHARD | True | False | 12846090240 |
|  7b | 32768 | 8 | h100_80gb | 30.45 | 40.6 | 2 | 2 | 32 | 0 | 26127 | 3265 | 1048576 | amp_bf16 | PURE | FULL_SHARD | True | False | 6784688128 |
|  7b | 8192 | 8 | h100_80gb | 36.43 | 48.57 | 8 | 2 | 128 | 6 | 54419 | 6802 | 1048576 | amp_bf16 | PURE | FULL_SHARD | True | False | 6684024832 |
|  7b | 2048 | 8 | h100_80gb | 40.48 | 53.97 | 32 | 2 | 512 | 36 | 74217 | 9277 | 1048576 | amp_bf16 | PURE | FULL_SHARD | True | False | 6658859008 |
|  7b | 512 | 8 | h100_80gb | 42.02 | 56.02 | 128 | 2 | 2048 | 159 | 81676 | 10209 | 1048576 | amp_bf16 | PURE | FULL_SHARD | True | False | 6652567552 |
|  3b | 32768 | 8 | h100_80gb | 28.0 | 37.33 | 3 | 6 | 144 | 1 | 45607 | 5700 | 4718592 | amp_bf16 | PURE | FULL_SHARD | True | False | 2730480640 |
|  3b | 32768 | 8 | h100_80gb | 14.38 | 19.18 | 3 | 6 | 144 | 1 | 46853 | 5856 | 4718592 | amp_fp8 | DEFAULT | FULL_SHARD | True | False | 2730480640 |
|  3b | 8192 | 8 | h100_80gb | 40.35 | 40.35 | 3 | 6 | 144 | 16 | 132753 | 16594 | 1179648 | amp_bf16 | PURE | FULL_SHARD | False | False | 2667566080 |
|  3b | 8192 | 8 | h100_80gb | 23.28 | 23.28 | 3 | 6 | 144 | 18 | 153174 | 19146 | 1179648 | amp_fp8 | DEFAULT | FULL_SHARD | False | False | 2667566080 |
|  3b | 2048 | 8 | h100_80gb | 44.43 | 44.43 | 10 | 6 | 480 | 95 | 196229 | 24528 | 983040 | amp_bf16 | PURE | FULL_SHARD | False | False | 2651837440 |
|  3b | 2048 | 8 | h100_80gb | 27.7 | 27.7 | 10 | 6 | 480 | 119 | 244692 | 30586 | 983040 | amp_fp8 | DEFAULT | FULL_SHARD | False | False | 2651837440 |
|  3b | 512 | 8 | h100_80gb | 46.38 | 46.38 | 40 | 6 | 1920 | 437 | 223994 | 27999 | 983040 | amp_bf16 | PURE | FULL_SHARD | False | False | 2647905280 |
|  3b | 512 | 8 | h100_80gb | 30.25 | 30.25 | 40 | 6 | 1920 | 570 | 292217 | 36527 | 983040 | amp_fp8 | DEFAULT | FULL_SHARD | False | False | 2647905280 |
|  1b | 32768 | 8 | h100_80gb | 33.6 | 33.6 | 1 | 4 | 32 | 2 | 96354 | 12044 | 1048576 | amp_bf16 | PURE | FULL_SHARD | False | False | 1378865152 |
|  1b | 32768 | 8 | h100_80gb | 17.55 | 17.55 | 1 | 4 | 32 | 3 | 100643 | 12580 | 1048576 | amp_fp8 | DEFAULT | FULL_SHARD | False | False | 1378865152 |
|  1b | 8192 | 8 | h100_80gb | 36.74 | 36.74 | 2 | 4 | 64 | 27 | 227183 | 28397 | 524288 | amp_bf16 | PURE | FULL_SHARD | False | False | 1328533504 |
|  1b | 8192 | 8 | h100_80gb | 20.71 | 20.71 | 2 | 4 | 64 | 31 | 256087 | 32010 | 524288 | amp_fp8 | DEFAULT | FULL_SHARD | False | False | 1328533504 |
|  1b | 512 | 8 | h100_80gb | 29.06 | 29.06 | 56 | 4 | 1792 | 1098 | 562523 | 70315 | 917504 | amp_fp8 | DEFAULT | FULL_SHARD | False | False | 1312804864 |
|  350m | 32768 | 8 | h100_80gb | 28.98 | 28.98 | 1 | 4 | 32 | 5 | 191350 | 23918 | 1048576 | amp_bf16 | PURE | FULL_SHARD | False | False | 387442688 |
|  350m | 32768 | 8 | h100_80gb | 14.8 | 14.8 | 1 | 4 | 32 | 5 | 195516 | 24439 | 1048576 | amp_fp8 | DEFAULT | FULL_SHARD | False | False | 387442688 |
|  350m | 16384 | 8 | h100_80gb | 29.95 | 29.95 | 2 | 4 | 64 | 20 | 336016 | 42002 | 1048576 | amp_bf16 | PURE | FULL_SHARD | False | False | 370665472 |
|  350m | 16384 | 8 | h100_80gb | 15.31 | 15.31 | 2 | 4 | 64 | 20 | 343435 | 42929 | 1048576 | amp_fp8 | DEFAULT | FULL_SHARD | False | False | 370665472 |
|  350m | 512 | 8 | h100_80gb | 32.79 | 32.79 | 56 | 4 | 1792 | 2226 | 1139870 | 142483 | 917504 | amp_bf16 | PURE | FULL_SHARD | False | False | 354412544 |
|  350m | 512 | 8 | h100_80gb | 17.77 | 17.77 | 56 | 4 | 1792 | 2412 | 1235360 | 154420 | 917504 | amp_fp8 | DEFAULT | FULL_SHARD | False | False | 354412544 |

## A100 80GB with 1600 Gbps node-node interconnect (RoCE)

|  Model | SeqLen (T) | # GPUs | GPU | MFU | HFU | MicroBatchSize | GradAccum | GlobalBatchSize | Throughput (S/s) | Throughput (T/s) | Throughput (T/s/GPU) | GlobalBatchSize (T) | Precision | MP Mode | Sharding Strategy | Activation Checkpointing | Activation CPUOffload | NumParams |
|  --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  70b | 2048 | 64 | a100_80gb | 53.33 | 71.1 | 8 | 4 | 2048 | 12 | 26274 | 410 | 4194304 | bf16 | PURE | FULL_SHARD | True | False | 64862437376 |
|  70b | 2048 | 32 | a100_80gb | 48.56 | 64.75 | 2 | 16 | 1024 | 5 | 11962 | 373 | 2097152 | bf16 | PURE | FULL_SHARD | True | False | 64862437376 |
|  30b | 8192 | 8 | a100_80gb | 42.66 | 56.89 | 1 | 21 | 168 | 0 | 4977 | 622 | 1376256 | bf16 | PURE | FULL_SHARD | True | False | 30019254272 |
|  30b | 4096 | 8 | a100_80gb | 51.37 | 68.49 | 1 | 21 | 168 | 1 | 6513 | 814 | 688128 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 29989894144 |
|  30b | 2048 | 8 | a100_80gb | 55.3 | 73.74 | 3 | 21 | 504 | 3 | 7330 | 916 | 1032192 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 29975214080 |
|  30b | 1024 | 8 | a100_80gb | 55.82 | 74.43 | 6 | 21 | 1008 | 7 | 7571 | 946 | 1032192 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 29967874048 |
|  30b | 512 | 8 | a100_80gb | 56.4 | 75.2 | 12 | 21 | 2016 | 15 | 7739 | 967 | 1032192 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 29964204032 |
|  13b | 32768 | 8 | a100_80gb | 51.69 | 68.92 | 1 | 3 | 24 | 0 | 8134 | 1016 | 786432 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 13011240960 |
|  13b | 16384 | 8 | a100_80gb | 54.07 | 72.1 | 3 | 3 | 72 | 0 | 11454 | 1431 | 1179648 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 12927354880 |
|  13b | 8192 | 8 | a100_80gb | 56.07 | 74.76 | 5 | 3 | 120 | 1 | 14362 | 1795 | 983040 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 12885411840 |
|  13b | 4096 | 8 | a100_80gb | 57.62 | 76.82 | 10 | 3 | 240 | 4 | 16482 | 2060 | 983040 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 12864440320 |
|  13b | 2048 | 8 | a100_80gb | 59.57 | 59.57 | 2 | 3 | 48 | 8 | 18097 | 2262 | 98304 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 12853954560 |
|  13b | 1024 | 8 | a100_80gb | 59.48 | 79.3 | 40 | 3 | 960 | 18 | 18647 | 2330 | 983040 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 12848711680 |
|  7b | 65536 | 8 | a100_80gb | 46.97 | 62.63 | 1 | 2 | 16 | 0 | 8108 | 1013 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 6918905856 |
|  7b | 32768 | 8 | a100_80gb | 49.46 | 65.94 | 2 | 2 | 32 | 0 | 13382 | 1672 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 6784688128 |
|  7b | 16384 | 8 | a100_80gb | 51.96 | 69.28 | 4 | 2 | 64 | 1 | 19629 | 2453 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 6717579264 |
|  7b | 8192 | 8 | a100_80gb | 54.47 | 72.62 | 8 | 2 | 128 | 3 | 25655 | 3206 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 6684024832 |
|  7b | 4096 | 8 | a100_80gb | 54.84 | 73.12 | 16 | 2 | 256 | 7 | 29472 | 3684 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 6667247616 |
|  7b | 2048 | 8 | a100_80gb | 64.23 | 64.23 | 6 | 2 | 96 | 18 | 37130 | 4641 | 196608 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 6658859008 |
|  7b | 1024 | 8 | a100_80gb | 58.01 | 77.35 | 64 | 2 | 1024 | 34 | 34857 | 4357 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 6654664704 |
|  3b | 65536 | 8 | a100_80gb | 46.05 | 61.41 | 1 | 2 | 16 | 0 | 14137 | 1767 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 2814366720 |
|  3b | 32768 | 8 | a100_80gb | 47.18 | 62.91 | 3 | 6 | 144 | 0 | 24235 | 3029 | 4718592 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 2730480640 |
|  3b | 16384 | 8 | a100_80gb | 57.13 | 57.13 | 1 | 6 | 48 | 2 | 44233 | 5529 | 786432 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 2688537600 |
|  3b | 8192 | 8 | a100_80gb | 59.34 | 59.34 | 3 | 6 | 144 | 7 | 61567 | 7695 | 1179648 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 2667566080 |
|  3b | 4096 | 8 | a100_80gb | 60.53 | 60.53 | 5 | 6 | 240 | 18 | 75658 | 9457 | 983040 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 2657080320 |
|  3b | 2048 | 8 | a100_80gb | 62.11 | 62.11 | 10 | 2 | 160 | 42 | 86491 | 10811 | 327680 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 2651837440 |
|  3b | 1024 | 8 | a100_80gb | 62.73 | 62.73 | 20 | 6 | 960 | 90 | 92643 | 11580 | 983040 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 2649216000 |
|  3b | 512 | 8 | a100_80gb | 63.71 | 63.71 | 40 | 6 | 1920 | 189 | 97019 | 12127 | 983040 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 2647905280 |
|  1b | 65536 | 8 | a100_80gb | 46.18 | 61.57 | 1 | 2 | 16 | 0 | 24353 | 3044 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 1445974016 |
|  1b | 32768 | 8 | a100_80gb | 55.52 | 55.52 | 1 | 4 | 32 | 1 | 50207 | 6275 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 1378865152 |
|  1b | 16384 | 8 | a100_80gb | 56.6 | 56.6 | 2 | 4 | 64 | 4 | 79650 | 9956 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 1345310720 |
|  1b | 8192 | 8 | a100_80gb | 56.69 | 56.69 | 3 | 4 | 96 | 13 | 110516 | 13814 | 786432 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 1328533504 |
|  1b | 4096 | 8 | a100_80gb | 59.0 | 59.0 | 7 | 4 | 224 | 34 | 142457 | 17807 | 917504 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 1320144896 |
|  1b | 2048 | 8 | a100_80gb | 59.86 | 59.86 | 14 | 4 | 448 | 80 | 164109 | 20513 | 917504 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 1315950592 |
|  1b | 1024 | 8 | a100_80gb | 60.15 | 60.15 | 18 | 4 | 576 | 172 | 176898 | 22112 | 589824 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 1313853440 |
|  1b | 512 | 8 | a100_80gb | 60.68 | 60.68 | 56 | 4 | 1792 | 361 | 185186 | 23148 | 917504 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 1312804864 |
|  760m | 65536 | 8 | a100_80gb | 45.34 | 60.45 | 1 | 2 | 16 | 0 | 33150 | 4143 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 857988096 |
|  760m | 32768 | 8 | a100_80gb | 54.57 | 54.57 | 1 | 2 | 16 | 2 | 70417 | 8802 | 524288 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 807656448 |
|  760m | 16384 | 8 | a100_80gb | 54.64 | 54.64 | 3 | 2 | 48 | 6 | 114198 | 14274 | 786432 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 782490624 |
|  760m | 8192 | 8 | a100_80gb | 55.31 | 55.31 | 6 | 2 | 96 | 20 | 167471 | 20933 | 786432 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 769907712 |
|  760m | 4096 | 8 | a100_80gb | 56.05 | 56.05 | 12 | 2 | 192 | 53 | 218808 | 27351 | 786432 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 763616256 |
|  760m | 2048 | 8 | a100_80gb | 56.85 | 56.85 | 24 | 2 | 384 | 126 | 259472 | 32434 | 786432 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 760470528 |
|  760m | 1024 | 8 | a100_80gb | 47.76 | 47.76 | 48 | 2 | 768 | 232 | 238122 | 29765 | 786432 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 758897664 |
|  760m | 512 | 8 | a100_80gb | 45.07 | 45.07 | 96 | 2 | 1536 | 460 | 235571 | 29446 | 786432 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 758111232 |
|  350m | 65536 | 8 | a100_80gb | 52.7 | 52.7 | 1 | 2 | 16 | 0 | 60195 | 7524 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 420997120 |
|  350m | 32768 | 8 | a100_80gb | 52.46 | 52.46 | 2 | 2 | 32 | 3 | 109222 | 13652 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 387442688 |
|  350m | 16384 | 8 | a100_80gb | 53.28 | 53.28 | 4 | 2 | 64 | 11 | 188478 | 23559 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 370665472 |
|  350m | 8192 | 8 | a100_80gb | 53.8 | 53.8 | 8 | 2 | 128 | 35 | 292559 | 36569 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 362276864 |
|  350m | 4096 | 8 | a100_80gb | 53.31 | 53.31 | 16 | 2 | 256 | 96 | 396442 | 49555 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 358082560 |
|  350m | 2048 | 8 | a100_80gb | 51.62 | 51.62 | 32 | 2 | 512 | 229 | 470263 | 58782 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 355985408 |
|  350m | 1024 | 8 | a100_80gb | 50.51 | 50.51 | 64 | 2 | 1024 | 506 | 518504 | 64813 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 354936832 |
|  350m | 512 | 8 | a100_80gb | 50.61 | 50.61 | 128 | 2 | 2048 | 1083 | 554643 | 69330 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 354412544 |
|  125m | 65536 | 8 | a100_80gb | 54.13 | 54.13 | 1 | 2 | 16 | 2 | 162946 | 20368 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 174070272 |
|  125m | 32768 | 8 | a100_80gb | 52.71 | 52.71 | 2 | 2 | 32 | 8 | 291256 | 36407 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 148904448 |
|  125m | 16384 | 8 | a100_80gb | 50.61 | 50.61 | 4 | 2 | 64 | 29 | 480322 | 60040 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 136321536 |
|  125m | 8192 | 8 | a100_80gb | 48.85 | 48.85 | 8 | 2 | 128 | 88 | 723142 | 90392 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 130030080 |
|  125m | 4096 | 8 | a100_80gb | 46.08 | 46.08 | 16 | 2 | 256 | 231 | 947172 | 118396 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 126884352 |
|  125m | 2048 | 8 | a100_80gb | 44.45 | 44.45 | 32 | 2 | 512 | 553 | 1133901 | 141737 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 125311488 |
|  125m | 1024 | 8 | a100_80gb | 43.15 | 43.15 | 64 | 2 | 1024 | 1222 | 1251751 | 156468 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 124525056 |
|  125m | 512 | 8 | a100_80gb | 42.56 | 42.56 | 128 | 2 | 2048 | 2588 | 1325455 | 165681 | 1048576 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 124131840 |
|  125m | 2048 | 8 | a100_80gb | 44.79 | 44.79 | 40 | 2 | 640 | 557 | 1142641 | 142830 | 1310720 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 125311488 |

## A100 40GB with 1600 Gbps node-node interconnect (RoCE)

|  Model | SeqLen (T) | # GPUs | GPU | MFU | HFU | MicroBatchSize | GradAccum | GlobalBatchSize | Throughput (S/s) | Throughput (T/s) | Throughput (T/s/GPU) | GlobalBatchSize (T) | Precision | MP Mode | Sharding Strategy | Activation Checkpointing | Activation CPUOffload | NumParams |
|  --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  70b | 2048 | 128 | a100_40gb | 48.91 | 65.21 | 4 | 1 | 512 | 23 | 48194 | 376 | 1048576 | bf16 | PURE | FULL_SHARD | True | False | 64862437376 |
|  70b | 2048 | 64 | a100_40gb | 35.87 | 47.82 | 2 | 1 | 128 | 8 | 17672 | 276 | 262144 | bf16 | PURE | FULL_SHARD | True | False | 64862437376 |
|  30b | 2048 | 128 | a100_40gb | 52.25 | 69.66 | 6 | 1 | 768 | 54 | 110803 | 865 | 1572864 | bf16 | PURE | FULL_SHARD | True | False | 29975214080 |
|  30b | 2048 | 32 | a100_40gb | 51.74 | 68.98 | 4 | 1 | 128 | 13 | 27431 | 857 | 262144 | bf16 | PURE | FULL_SHARD | True | False | 29975214080 |
|  13b | 8192 | 8 | a100_40gb | 43.95 | 58.6 | 1 | 16 | 128 | 1 | 11258 | 1407 | 1048576 | bf16 | PURE | FULL_SHARD | True | False | 12885411840 |
|  13b | 4096 | 8 | a100_40gb | 44.85 | 59.8 | 2 | 16 | 256 | 3 | 12830 | 1603 | 1048576 | bf16 | PURE | FULL_SHARD | True | False | 12864440320 |
|  13b | 2048 | 128 | a100_40gb | 51.93 | 69.24 | 16 | 1 | 2048 | 123 | 252444 | 1972 | 4194304 | bf16 | PURE | FULL_SHARD | True | False | 12853954560 |
|  13b | 2048 | 64 | a100_40gb | 52.04 | 69.39 | 16 | 1 | 1024 | 61 | 126479 | 1976 | 2097152 | bf16 | PURE | FULL_SHARD | True | False | 12853954560 |
|  13b | 2048 | 32 | a100_40gb | 52.62 | 70.16 | 14 | 1 | 448 | 31 | 63946 | 1998 | 917504 | bf16 | PURE | FULL_SHARD | True | False | 12853954560 |
|  13b | 2048 | 16 | a100_40gb | 52.5 | 70.0 | 10 | 1 | 160 | 15 | 31900 | 1993 | 327680 | bf16 | PURE | FULL_SHARD | True | False | 12853954560 |
|  13b | 2048 | 8 | a100_40gb | 43.94 | 58.58 | 4 | 16 | 512 | 6 | 13347 | 1668 | 1048576 | bf16 | PURE | FULL_SHARD | True | False | 12853954560 |
|  13b | 1024 | 8 | a100_40gb | 44.07 | 58.76 | 8 | 16 | 1024 | 13 | 13817 | 1727 | 1048576 | bf16 | PURE | FULL_SHARD | True | False | 12848711680 |
|  13b | 512 | 8 | a100_40gb | 44.28 | 59.04 | 16 | 16 | 2048 | 27 | 14108 | 1763 | 1048576 | bf16 | PURE | FULL_SHARD | True | False | 12846090240 |
|  7b | 16384 | 8 | a100_40gb | 47.65 | 63.53 | 1 | 4 | 32 | 1 | 17998 | 2249 | 524288 | bf16 | PURE | FULL_SHARD | True | False | 6717579264 |
|  7b | 8192 | 8 | a100_40gb | 49.04 | 65.38 | 3 | 4 | 96 | 2 | 23098 | 2887 | 786432 | bf16 | PURE | FULL_SHARD | True | False | 6684024832 |
|  7b | 4096 | 8 | a100_40gb | 50.11 | 66.82 | 6 | 4 | 192 | 6 | 26930 | 3366 | 786432 | bf16 | PURE | FULL_SHARD | True | False | 6667247616 |
|  7b | 2048 | 128 | a100_40gb | 50.14 | 66.85 | 18 | 1 | 2304 | 226 | 463749 | 3623 | 4718592 | bf16 | PURE | FULL_SHARD | True | False | 6658859008 |
|  7b | 2048 | 64 | a100_40gb | 50.73 | 67.64 | 18 | 1 | 1152 | 114 | 234614 | 3665 | 2359296 | bf16 | PURE | FULL_SHARD | True | False | 6658859008 |
|  7b | 2048 | 32 | a100_40gb | 51.55 | 68.73 | 18 | 1 | 576 | 58 | 119202 | 3725 | 1179648 | bf16 | PURE | FULL_SHARD | True | False | 6658859008 |
|  7b | 2048 | 16 | a100_40gb | 50.44 | 67.26 | 16 | 1 | 256 | 28 | 58322 | 3645 | 524288 | bf16 | PURE | FULL_SHARD | True | False | 6658859008 |
|  7b | 2048 | 8 | a100_40gb | 50.92 | 67.89 | 12 | 4 | 384 | 14 | 29436 | 3679 | 786432 | bf16 | PURE | FULL_SHARD | True | False | 6658859008 |
|  7b | 1024 | 8 | a100_40gb | 51.31 | 68.42 | 24 | 4 | 768 | 30 | 30833 | 3854 | 786432 | bf16 | PURE | FULL_SHARD | True | False | 6654664704 |
|  7b | 512 | 8 | a100_40gb | 50.85 | 67.8 | 48 | 4 | 1536 | 60 | 31167 | 3895 | 786432 | bf16 | PURE | FULL_SHARD | True | False | 6652567552 |
|  3b | 32768 | 8 | a100_40gb | 46.03 | 61.37 | 1 | 4 | 32 | 0 | 23640 | 2955 | 1048576 | bf16 | PURE | FULL_SHARD | True | False | 2730480640 |
|  3b | 16384 | 8 | a100_40gb | 46.14 | 61.52 | 2 | 8 | 128 | 2 | 35726 | 4465 | 2097152 | bf16 | PURE | FULL_SHARD | True | False | 2688537600 |
|  3b | 8192 | 8 | a100_40gb | 55.13 | 55.13 | 1 | 8 | 64 | 6 | 57193 | 7149 | 524288 | bf16 | PURE | FULL_SHARD | False | False | 2667566080 |
|  3b | 4096 | 8 | a100_40gb | 56.18 | 56.18 | 2 | 8 | 128 | 17 | 70223 | 8777 | 524288 | bf16 | PURE | FULL_SHARD | False | False | 2657080320 |
|  3b | 2048 | 128 | a100_40gb | 54.8 | 54.8 | 6 | 1 | 768 | 596 | 1220885 | 9538 | 1572864 | bf16 | PURE | FULL_SHARD | False | False | 2651837440 |
|  3b | 2048 | 64 | a100_40gb | 55.94 | 55.94 | 6 | 1 | 384 | 304 | 623167 | 9736 | 786432 | bf16 | PURE | FULL_SHARD | False | False | 2651837440 |
|  3b | 2048 | 32 | a100_40gb | 56.96 | 56.96 | 6 | 1 | 192 | 154 | 317261 | 9914 | 393216 | bf16 | PURE | FULL_SHARD | False | False | 2651837440 |
|  3b | 2048 | 16 | a100_40gb | 56.02 | 56.02 | 5 | 1 | 80 | 76 | 156013 | 9750 | 163840 | bf16 | PURE | FULL_SHARD | False | False | 2651837440 |
|  3b | 2048 | 8 | a100_40gb | 57.82 | 57.82 | 5 | 8 | 320 | 39 | 80520 | 10065 | 655360 | bf16 | PURE | FULL_SHARD | False | False | 2651837440 |
|  3b | 1024 | 8 | a100_40gb | 58.14 | 58.14 | 10 | 8 | 640 | 83 | 85854 | 10731 | 655360 | bf16 | PURE | FULL_SHARD | False | False | 2649216000 |
|  3b | 512 | 8 | a100_40gb | 59.49 | 59.49 | 20 | 8 | 1280 | 176 | 90596 | 11324 | 655360 | bf16 | PURE | FULL_SHARD | False | False | 2647905280 |
|  1b | 32768 | 8 | a100_40gb | 45.07 | 60.1 | 1 | 4 | 32 | 1 | 40762 | 5095 | 1048576 | bf16 | PURE | FULL_SHARD | True | False | 1378865152 |
|  1b | 16384 | 8 | a100_40gb | 55.23 | 55.23 | 1 | 8 | 64 | 4 | 77723 | 9715 | 1048576 | bf16 | PURE | FULL_SHARD | False | False | 1345310720 |
|  1b | 8192 | 8 | a100_40gb | 55.29 | 55.29 | 2 | 8 | 128 | 13 | 107799 | 13474 | 1048576 | bf16 | PURE | FULL_SHARD | False | False | 1328533504 |
|  1b | 4096 | 8 | a100_40gb | 55.85 | 55.85 | 4 | 8 | 256 | 32 | 134851 | 16856 | 1048576 | bf16 | PURE | FULL_SHARD | False | False | 1320144896 |
|  1b | 2048 | 128 | a100_40gb | 54.41 | 54.41 | 10 | 1 | 1280 | 1165 | 2386897 | 18647 | 2621440 | bf16 | PURE | FULL_SHARD | False | False | 1315950592 |
|  1b | 2048 | 64 | a100_40gb | 55.44 | 55.44 | 10 | 1 | 640 | 593 | 1216104 | 19001 | 1310720 | bf16 | PURE | FULL_SHARD | False | False | 1315950592 |
|  1b | 2048 | 32 | a100_40gb | 45.39 | 45.39 | 10 | 1 | 320 | 243 | 497782 | 15555 | 655360 | bf16 | PURE | FULL_SHARD | False | False | 1315950592 |
|  1b | 2048 | 16 | a100_40gb | 55.69 | 55.69 | 8 | 1 | 128 | 149 | 305372 | 19085 | 262144 | bf16 | PURE | FULL_SHARD | False | False | 1315950592 |
|  1b | 2048 | 8 | a100_40gb | 56.23 | 56.23 | 8 | 8 | 512 | 75 | 154171 | 19271 | 1048576 | bf16 | PURE | FULL_SHARD | False | False | 1315950592 |
|  1b | 1024 | 8 | a100_40gb | 57.02 | 57.02 | 16 | 8 | 1024 | 163 | 167677 | 20959 | 1048576 | bf16 | PURE | FULL_SHARD | False | False | 1313853440 |
|  1b | 512 | 8 | a100_40gb | 57.1 | 57.1 | 32 | 8 | 2048 | 340 | 174256 | 21782 | 1048576 | bf16 | PURE | FULL_SHARD | False | False | 1312804864 |
|  760m | 32768 | 8 | a100_40gb | 44.53 | 59.37 | 1 | 4 | 32 | 1 | 57464 | 7183 | 1048576 | bf16 | PURE | FULL_SHARD | True | False | 807656448 |
|  760m | 16384 | 8 | a100_40gb | 53.26 | 53.26 | 1 | 4 | 32 | 6 | 111316 | 13914 | 524288 | bf16 | PURE | FULL_SHARD | False | False | 782490624 |
|  760m | 8192 | 8 | a100_40gb | 53.12 | 53.12 | 3 | 4 | 96 | 19 | 160853 | 20106 | 786432 | bf16 | PURE | FULL_SHARD | False | False | 769907712 |
|  760m | 4096 | 8 | a100_40gb | 53.0 | 53.0 | 6 | 4 | 192 | 50 | 206909 | 25863 | 786432 | bf16 | PURE | FULL_SHARD | False | False | 763616256 |
|  760m | 2048 | 128 | a100_40gb | 50.73 | 50.73 | 12 | 1 | 1536 | 1808 | 3704382 | 28940 | 3145728 | bf16 | PURE | FULL_SHARD | False | False | 760470528 |
|  760m | 2048 | 64 | a100_40gb | 51.44 | 51.44 | 12 | 1 | 768 | 917 | 1878030 | 29344 | 1572864 | bf16 | PURE | FULL_SHARD | False | False | 760470528 |
|  760m | 2048 | 32 | a100_40gb | 51.97 | 51.97 | 12 | 1 | 384 | 463 | 948745 | 29648 | 786432 | bf16 | PURE | FULL_SHARD | False | False | 760470528 |
|  760m | 2048 | 16 | a100_40gb | 51.9 | 51.9 | 12 | 1 | 192 | 231 | 473723 | 29607 | 393216 | bf16 | PURE | FULL_SHARD | False | False | 760470528 |
|  760m | 2048 | 8 | a100_40gb | 52.89 | 52.89 | 12 | 4 | 384 | 117 | 241389 | 30173 | 786432 | bf16 | PURE | FULL_SHARD | False | False | 760470528 |
|  760m | 1024 | 8 | a100_40gb | 53.63 | 53.63 | 24 | 4 | 768 | 261 | 267376 | 33422 | 786432 | bf16 | PURE | FULL_SHARD | False | False | 758897664 |
|  760m | 512 | 8 | a100_40gb | 53.47 | 53.47 | 48 | 4 | 1536 | 545 | 279504 | 34938 | 786432 | bf16 | PURE | FULL_SHARD | False | False | 758111232 |
|  350m | 32768 | 8 | a100_40gb | 51.55 | 51.55 | 1 | 4 | 32 | 3 | 107329 | 13416 | 1048576 | bf16 | PURE | FULL_SHARD | False | False | 387442688 |
|  350m | 16384 | 8 | a100_40gb | 51.78 | 51.78 | 2 | 4 | 64 | 11 | 183175 | 22896 | 1048576 | bf16 | PURE | FULL_SHARD | False | False | 370665472 |
|  350m | 8192 | 8 | a100_40gb | 51.39 | 51.39 | 4 | 4 | 128 | 34 | 279466 | 34933 | 1048576 | bf16 | PURE | FULL_SHARD | False | False | 362276864 |
|  350m | 4096 | 8 | a100_40gb | 50.38 | 50.38 | 8 | 4 | 256 | 91 | 374670 | 46833 | 1048576 | bf16 | PURE | FULL_SHARD | False | False | 358082560 |
|  350m | 2048 | 128 | a100_40gb | 45.61 | 45.61 | 18 | 1 | 2304 | 3245 | 6647647 | 51934 | 4718592 | bf16 | PURE | FULL_SHARD | False | False | 355985408 |
|  350m | 2048 | 64 | a100_40gb | 46.27 | 46.27 | 18 | 1 | 1152 | 1646 | 3372118 | 52689 | 2359296 | bf16 | PURE | FULL_SHARD | False | False | 355985408 |
|  350m | 2048 | 32 | a100_40gb | 47.26 | 47.26 | 18 | 1 | 576 | 840 | 1721978 | 53811 | 1179648 | bf16 | PURE | FULL_SHARD | False | False | 355985408 |
|  350m | 2048 | 16 | a100_40gb | 48.66 | 48.66 | 18 | 1 | 288 | 432 | 886622 | 55413 | 589824 | bf16 | PURE | FULL_SHARD | False | False | 355985408 |
|  350m | 2048 | 8 | a100_40gb | 49.17 | 49.17 | 16 | 4 | 512 | 218 | 447963 | 55995 | 1048576 | bf16 | PURE | FULL_SHARD | False | False | 355985408 |
|  350m | 1024 | 8 | a100_40gb | 48.73 | 48.73 | 32 | 4 | 1024 | 488 | 500184 | 62523 | 1048576 | bf16 | PURE | FULL_SHARD | False | False | 354936832 |
|  350m | 512 | 8 | a100_40gb | 48.39 | 48.39 | 64 | 4 | 2048 | 1035 | 530277 | 66284 | 1048576 | bf16 | PURE | FULL_SHARD | False | False | 354412544 |
|  125m | 32768 | 8 | a100_40gb | 47.27 | 47.27 | 1 | 4 | 32 | 7 | 261208 | 32651 | 1048576 | bf16 | PURE | FULL_SHARD | False | False | 148904448 |
|  125m | 16384 | 8 | a100_40gb | 46.77 | 46.77 | 2 | 3 | 48 | 27 | 443876 | 55484 | 786432 | bf16 | PURE | FULL_SHARD | False | False | 136321536 |
|  125m | 8192 | 8 | a100_40gb | 46.94 | 46.94 | 5 | 3 | 120 | 84 | 694868 | 86858 | 983040 | bf16 | PURE | FULL_SHARD | False | False | 130030080 |
|  125m | 4096 | 8 | a100_40gb | 44.82 | 44.82 | 13 | 3 | 312 | 224 | 921297 | 115162 | 1277952 | bf16 | PURE | FULL_SHARD | False | False | 126884352 |
|  125m | 2048 | 128 | a100_40gb | 38.86 | 38.86 | 26 | 1 | 3328 | 7746 | 15863837 | 123936 | 6815744 | bf16 | PURE | FULL_SHARD | False | False | 125311488 |
|  125m | 2048 | 64 | a100_40gb | 39.27 | 39.27 | 26 | 1 | 1664 | 3913 | 8015010 | 125234 | 3407872 | bf16 | PURE | FULL_SHARD | False | False | 125311488 |
|  125m | 2048 | 32 | a100_40gb | 39.86 | 39.86 | 26 | 1 | 832 | 1986 | 4067922 | 127122 | 1703936 | bf16 | PURE | FULL_SHARD | False | False | 125311488 |
|  125m | 2048 | 16 | a100_40gb | 40.93 | 40.93 | 26 | 1 | 416 | 1019 | 2088560 | 130535 | 851968 | bf16 | PURE | FULL_SHARD | False | False | 125311488 |
|  125m | 2048 | 8 | a100_40gb | 42.75 | 42.75 | 26 | 3 | 624 | 532 | 1090678 | 136334 | 1277952 | bf16 | PURE | FULL_SHARD | False | False | 125311488 |
|  125m | 1024 | 8 | a100_40gb | 40.89 | 40.89 | 52 | 3 | 1248 | 1158 | 1186314 | 148289 | 1277952 | bf16 | PURE | FULL_SHARD | False | False | 124525056 |
|  125m | 512 | 8 | a100_40gb | 40.26 | 40.26 | 104 | 3 | 2496 | 2448 | 1253886 | 156735 | 1277952 | bf16 | PURE | FULL_SHARD | False | False | 124131840 |