# MosaicGPT Training Benchmarks

Benchmark measurements for MosaicGPT models trained on [MosaicML platform](https://www.mosaicml.com/platform), including throughput, MFU, and HFU. Each model is based on optimized configurations of various sizes in the [yamls](../yamls) folder, ranging from a 125m to 70B parameter models.

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

Model FLOPs Utilization (MFU) and Hardware FLOPS Utilization (HFU) are estimates, based on the throughput and the known FLOPs of the computation, of what percentage of the hardware's FLOPs are being used during training.

MFU calculates the utilizaiton from the floating point operations required for a single forward/backwards pass of the model, and do not account for the additional compute required for other implementation details such as activation checkpointing. Thus, MFU is independant of implementation and hardware.

HFU attempt to capture the actual floating point operations incurred during the forward/backwards pass on the hardware, and is a more accurate measurement of hardware utilization, but less general and difficult to compare across various hardware and implementation details.

For more information, see [Korthikanti et al, 2022](https://arxiv.org/abs/2205.05198). All FLOP calculations exclude the operations required for normalization, activation, and residuals.

### MFU

Per token, each parameter is used for a MAC (2 FLOPS) per network operation. Neural Network training has 3 network operations: forward pass, backward pass, and computation of parameter gradient.

The attention mechanism forward pass FLOPS are: `attn_flops_per_seq = n_layers * 2 * 2 * (d_model * (seq_len**2))`
```
flops_per_token = 2 * n_params
flops_per_seq = flops_per_token * seq_len
mfu* = 3 * flops_per_seq * throughput / (gpu_num * GPU_AVAILABLE_FLOPS)

attn_flops_per_seq = n_layers * 2 * 2 * (d_model * (seq_len**2))
mfu = (3 * flops_per_seq + 3 * attn_flops_per_seq) * throughput / (gpu_num * GPU_AVAILABLE_FLOPS)
```

### HFU

The HFU numbers shown below account for the fact that the networks use checkpointing and recomputes activations. This effectively requires an extra forward pass through the network.
```
hfu* = 4 * flops_per_seq * throughput / (gpu_num * GPU_AVAILABLE_FLOPS)
hfu = (4 * flops_per_seq + 4 * attn_flops_per_seq) * throughput / (gpu_num * GPU_AVAILABLE_FLOPS)
```

Note that these are approximations. Actual HFU would be higher since it includes the floating point operations for normalization, activation, and residual lyaers, as well as **all** recomputation. For example, our models use Flash Attention, which requires including an extra recompute factor for its recomputation in the forward pass. Therefore, the attention multipler would be 5 instead of 4.

## Results

Below we include several configurations across different hardware platforms, sequence lengths and batch sizes. It is easy to benchmark configurations for your own use case. For example, using Mosaic Cloud, to test MosaicGPT {13B, 30B} using fp16 with a batch size of 2M tokens and seq len {2k, 4k, 8k, 16k} run:
```
python submit_benchmarks.py -m 13b.yaml 30b.yaml -t fp16 -b 21 21 -s 11 14 --RUN
```
This will run 8 configs for 12 steps to get throughput numbers. `python collect_results.py` can then be used to parse all output training logs and create the tables below.

Our microbatching engine enables microbatch sizes that do not divde Global Batchsize while being mathematically faithful to the global batch size. For example, a total batch size of 48, and a micro batch of 11, means we will accumulate gradients across microbatches of 11, 11, 11, 11, 4.

#### TODO: Update tables with torch 2.0 after next Composer release

## A100 80GB

|  Model | SeqLen (T) | # GPUs | GPU | MFU | HFU | MicroBatchSize | GradAccum | GlobalBatchSize | Throughput (S/s) | Throughput (T/s) | Throughput (T/s/GPU) | GlobalBatchSize (T) | Precision | MP Mode | Sharding Strategy | Activation Checkpointing | Activation CPUOffload | NumParams |
|  --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  70b | 2048 | 64 | a100_80gb | 53.33 | 71.1 | 8 | 4 | 2048 | 12 | 26274 | 410 | 4194304 | bf16 | PURE | FULL_SHARD | True | False | 64862437376 |
|  70b | 2048 | 32 | a100_80gb | 48.56 | 64.75 | 2 | 16 | 1024 | 5 | 11962 | 373 | 2097152 | bf16 | PURE | FULL_SHARD | True | False | 64862437376 |
|  30b | 8192 | 8 | a100_80gb | 42.66 | 56.89 | 1 | 21 | 168 | 0 | 4977 | 622 | 1376256 | bf16 | PURE | FULL_SHARD | True | False | 30019254272 |
|  30b | 4096 | 8 | a100_80gb | 49.12 | 65.49 | 1 | 21 | 168 | 1 | 6227 | 778 | 688128 | bf16 | PURE | FULL_SHARD | True | False | 29989894144 |
|  30b | 2048 | 64 | a100_80gb | 52.93 | 70.57 | 16 | 3 | 3072 | 27 | 56126 | 876 | 6291456 | bf16 | PURE | FULL_SHARD | True | False | 29975214080 |
|  30b | 2048 | 32 | a100_80gb | 53.48 | 71.3 | 14 | 3 | 1344 | 13 | 28353 | 886 | 2752512 | bf16 | PURE | FULL_SHARD | True | False | 29975214080 |
|  30b | 2048 | 16 | a100_80gb | 53.4 | 71.2 | 10 | 3 | 480 | 6 | 14157 | 884 | 983040 | bf16 | PURE | FULL_SHARD | True | False | 29975214080 |
|  30b | 2048 | 8 | a100_80gb | 47.57 | 63.43 | 3 | 21 | 504 | 3 | 6305 | 788 | 1032192 | bf16 | PURE | FULL_SHARD | True | False | 29975214080 |
|  30b | 1024 | 8 | a100_80gb | 51.69 | 68.92 | 6 | 21 | 1008 | 6 | 7010 | 876 | 1032192 | bf16 | PURE | FULL_SHARD | True | False | 29967874048 |
|  30b | 512 | 8 | a100_80gb | 49.23 | 65.63 | 12 | 21 | 2016 | 13 | 6754 | 844 | 1032192 | bf16 | PURE | FULL_SHARD | True | False | 29964204032 |
|  13b | 32768 | 8 | a100_80gb | 49.53 | 66.04 | 1 | 3 | 24 | 0 | 7795 | 974 | 786432 | bf16 | PURE | FULL_SHARD | True | False | 13011240960 |
|  13b | 16384 | 8 | a100_80gb | 51.71 | 68.94 | 3 | 3 | 72 | 0 | 10953 | 1369 | 1179648 | bf16 | PURE | FULL_SHARD | True | False | 12927354880 |
|  13b | 8192 | 8 | a100_80gb | 52.83 | 70.44 | 5 | 3 | 120 | 1 | 13531 | 1691 | 983040 | bf16 | PURE | FULL_SHARD | True | False | 12885411840 |
|  13b | 4096 | 8 | a100_80gb | 53.62 | 71.5 | 10 | 3 | 240 | 3 | 15339 | 1917 | 983040 | bf16 | PURE | FULL_SHARD | True | False | 12864440320 |
|  13b | 2048 | 64 | a100_80gb | 52.51 | 70.01 | 32 | 1 | 2048 | 62 | 127624 | 1994 | 4194304 | bf16 | PURE | FULL_SHARD | True | False | 12853954560 |
|  13b | 2048 | 32 | a100_80gb | 52.86 | 70.48 | 32 | 1 | 1024 | 31 | 64241 | 2007 | 2097152 | bf16 | PURE | FULL_SHARD | True | False | 12853954560 |
|  13b | 2048 | 16 | a100_80gb | 53.14 | 70.86 | 24 | 1 | 384 | 15 | 32291 | 2018 | 786432 | bf16 | PURE | FULL_SHARD | True | False | 12853954560 |
|  13b | 2048 | 8 | a100_80gb | 54.38 | 72.51 | 20 | 3 | 480 | 8 | 16522 | 2065 | 983040 | bf16 | PURE | FULL_SHARD | True | False | 12853954560 |
|  13b | 1024 | 8 | a100_80gb | 55.23 | 73.63 | 40 | 3 | 960 | 16 | 17315 | 2164 | 983040 | bf16 | PURE | FULL_SHARD | True | False | 12848711680 |
|  13b | 512 | 8 | a100_80gb | 54.99 | 73.32 | 80 | 3 | 1920 | 34 | 17521 | 2190 | 983040 | bf16 | PURE | FULL_SHARD | True | False | 12846090240 |
|  7b | 65536 | 8 | a100_80gb | 42.61 | 56.82 | 1 | 2 | 16 | 0 | 7355 | 919 | 1048576 | bf16 | PURE | FULL_SHARD | True | False | 6918905856 |
|  7b | 32768 | 8 | a100_80gb | 48.18 | 64.24 | 2 | 2 | 32 | 0 | 13035 | 1629 | 1048576 | bf16 | PURE | FULL_SHARD | True | False | 6784688128 |
|  7b | 16384 | 8 | a100_80gb | 49.5 | 66.0 | 4 | 2 | 64 | 1 | 18698 | 2337 | 1048576 | bf16 | PURE | FULL_SHARD | True | False | 6717579264 |
|  7b | 8192 | 8 | a100_80gb | 50.71 | 67.62 | 8 | 2 | 128 | 2 | 23887 | 2985 | 1048576 | bf16 | PURE | FULL_SHARD | True | False | 6684024832 |
|  7b | 4096 | 8 | a100_80gb | 52.05 | 69.4 | 16 | 2 | 256 | 6 | 27973 | 3496 | 1048576 | bf16 | PURE | FULL_SHARD | True | False | 6667247616 |
|  7b | 2048 | 64 | a100_80gb | 50.8 | 67.73 | 32 | 1 | 2048 | 114 | 234932 | 3670 | 4194304 | bf16 | PURE | FULL_SHARD | True | False | 6658859008 |
|  7b | 2048 | 32 | a100_80gb | 51.16 | 68.22 | 32 | 1 | 1024 | 57 | 118310 | 3697 | 2097152 | bf16 | PURE | FULL_SHARD | True | False | 6658859008 |
|  7b | 2048 | 16 | a100_80gb | 51.59 | 68.79 | 32 | 1 | 512 | 29 | 59653 | 3728 | 1048576 | bf16 | PURE | FULL_SHARD | True | False | 6658859008 |
|  7b | 2048 | 8 | a100_80gb | 52.92 | 70.56 | 32 | 2 | 512 | 14 | 30596 | 3824 | 1048576 | bf16 | PURE | FULL_SHARD | True | False | 6658859008 |
|  7b | 1024 | 8 | a100_80gb | 53.66 | 71.55 | 64 | 2 | 1024 | 31 | 32243 | 4030 | 1048576 | bf16 | PURE | FULL_SHARD | True | False | 6654664704 |
|  7b | 512 | 8 | a100_80gb | 53.5 | 71.34 | 128 | 2 | 2048 | 64 | 32794 | 4099 | 1048576 | bf16 | PURE | FULL_SHARD | True | False | 6652567552 |
|  3b | 65536 | 8 | a100_80gb | 46.17 | 61.57 | 1 | 2 | 16 | 0 | 14174 | 1771 | 1048576 | bf16 | PURE | FULL_SHARD | True | False | 2814366720 |
|  3b | 32768 | 8 | a100_80gb | 46.73 | 62.31 | 3 | 6 | 144 | 0 | 24003 | 3000 | 4718592 | bf16 | PURE | FULL_SHARD | True | False | 2730480640 |
|  3b | 16384 | 8 | a100_80gb | 57.29 | 57.29 | 1 | 6 | 48 | 2 | 44356 | 5544 | 786432 | bf16 | PURE | FULL_SHARD | False | False | 2688537600 |
|  3b | 8192 | 8 | a100_80gb | 58.68 | 58.68 | 3 | 6 | 144 | 7 | 60883 | 7610 | 1179648 | bf16 | PURE | FULL_SHARD | False | False | 2667566080 |
|  3b | 4096 | 8 | a100_80gb | 59.51 | 59.51 | 5 | 6 | 240 | 18 | 74388 | 9298 | 983040 | bf16 | PURE | FULL_SHARD | False | False | 2657080320 |
|  3b | 2048 | 64 | a100_80gb | 58.36 | 58.36 | 12 | 3 | 2304 | 317 | 650175 | 10158 | 4718592 | bf16 | PURE | FULL_SHARD | False | False | 2651837440 |
|  3b | 2048 | 32 | a100_80gb | 59.22 | 59.22 | 12 | 3 | 1152 | 161 | 329856 | 10308 | 2359296 | bf16 | PURE | FULL_SHARD | False | False | 2651837440 |
|  3b | 2048 | 16 | a100_80gb | 59.08 | 59.08 | 10 | 3 | 480 | 80 | 164543 | 10283 | 983040 | bf16 | PURE | FULL_SHARD | False | False | 2651837440 |
|  3b | 2048 | 8 | a100_80gb | 59.77 | 59.77 | 10 | 6 | 480 | 40 | 83230 | 10403 | 983040 | bf16 | PURE | FULL_SHARD | False | False | 2651837440 |
|  3b | 1024 | 8 | a100_80gb | 61.56 | 61.56 | 20 | 6 | 960 | 88 | 90906 | 11363 | 983040 | bf16 | PURE | FULL_SHARD | False | False | 2649216000 |
|  3b | 512 | 8 | a100_80gb | 62.09 | 62.09 | 40 | 6 | 1920 | 184 | 94553 | 11819 | 983040 | bf16 | PURE | FULL_SHARD | False | False | 2647905280 |
|  1b | 65536 | 8 | a100_80gb | 45.29 | 60.39 | 1 | 2 | 16 | 0 | 23885 | 2985 | 1048576 | bf16 | PURE | FULL_SHARD | True | False | 1445974016 |
|  1b | 32768 | 8 | a100_80gb | 56.02 | 56.02 | 1 | 4 | 32 | 1 | 50657 | 6332 | 1048576 | bf16 | PURE | FULL_SHARD | False | False | 1378865152 |
|  1b | 16384 | 8 | a100_80gb | 55.84 | 55.84 | 2 | 4 | 64 | 4 | 78591 | 9823 | 1048576 | bf16 | PURE | FULL_SHARD | False | False | 1345310720 |
|  1b | 8192 | 8 | a100_80gb | 56.38 | 56.38 | 3 | 4 | 96 | 13 | 109915 | 13739 | 786432 | bf16 | PURE | FULL_SHARD | False | False | 1328533504 |
|  1b | 4096 | 8 | a100_80gb | 58.3 | 58.3 | 7 | 4 | 224 | 34 | 140767 | 17595 | 917504 | bf16 | PURE | FULL_SHARD | False | False | 1320144896 |
|  1b | 2048 | 64 | a100_80gb | 56.67 | 56.67 | 20 | 1 | 1280 | 606 | 1243103 | 19423 | 2621440 | bf16 | PURE | FULL_SHARD | False | False | 1315950592 |
|  1b | 2048 | 32 | a100_80gb | 56.74 | 56.74 | 20 | 1 | 640 | 303 | 622285 | 19446 | 1310720 | bf16 | PURE | FULL_SHARD | False | False | 1315950592 |
|  1b | 2048 | 16 | a100_80gb | 57.47 | 57.47 | 20 | 1 | 320 | 153 | 315117 | 19694 | 655360 | bf16 | PURE | FULL_SHARD | False | False | 1315950592 |
|  1b | 2048 | 8 | a100_80gb | 59.16 | 59.16 | 14 | 4 | 448 | 79 | 162214 | 20276 | 917504 | bf16 | PURE | FULL_SHARD | False | False | 1315950592 |
|  1b | 1024 | 8 | a100_80gb | 58.98 | 58.98 | 18 | 4 | 576 | 169 | 173458 | 21682 | 589824 | bf16 | PURE | FULL_SHARD | False | False | 1313853440 |
|  1b | 512 | 8 | a100_80gb | 60.38 | 60.38 | 56 | 4 | 1792 | 359 | 184268 | 23033 | 917504 | bf16 | PURE | FULL_SHARD | False | False | 1312804864 |
|  760m | 65536 | 8 | a100_80gb | 45.48 | 60.64 | 1 | 2 | 16 | 0 | 33252 | 4156 | 1048576 | bf16 | PURE | FULL_SHARD | True | False | 857988096 |
|  760m | 32768 | 8 | a100_80gb | 54.48 | 54.48 | 1 | 2 | 16 | 2 | 70305 | 8788 | 524288 | bf16 | PURE | FULL_SHARD | False | False | 807656448 |
|  760m | 16384 | 8 | a100_80gb | 55.21 | 55.21 | 3 | 2 | 48 | 7 | 115383 | 14422 | 786432 | bf16 | PURE | FULL_SHARD | False | False | 782490624 |
|  760m | 8192 | 8 | a100_80gb | 55.13 | 55.13 | 6 | 2 | 96 | 20 | 166928 | 20866 | 786432 | bf16 | PURE | FULL_SHARD | False | False | 769907712 |
|  760m | 4096 | 8 | a100_80gb | 55.2 | 55.2 | 12 | 2 | 192 | 52 | 215501 | 26937 | 786432 | bf16 | PURE | FULL_SHARD | False | False | 763616256 |
|  760m | 2048 | 64 | a100_80gb | 51.82 | 51.82 | 24 | 1 | 1536 | 923 | 1892166 | 29565 | 3145728 | bf16 | PURE | FULL_SHARD | False | False | 760470528 |
|  760m | 2048 | 32 | a100_80gb | 53.27 | 53.27 | 24 | 1 | 768 | 474 | 972497 | 30390 | 1572864 | bf16 | PURE | FULL_SHARD | False | False | 760470528 |
|  760m | 2048 | 16 | a100_80gb | 53.56 | 53.56 | 24 | 1 | 384 | 238 | 488871 | 30554 | 786432 | bf16 | PURE | FULL_SHARD | False | False | 760470528 |
|  760m | 2048 | 8 | a100_80gb | 55.67 | 55.67 | 24 | 2 | 384 | 124 | 254104 | 31763 | 786432 | bf16 | PURE | FULL_SHARD | False | False | 760470528 |
|  760m | 1024 | 8 | a100_80gb | 55.98 | 55.98 | 48 | 2 | 768 | 272 | 279108 | 34888 | 786432 | bf16 | PURE | FULL_SHARD | False | False | 758897664 |
|  760m | 512 | 8 | a100_80gb | 56.2 | 56.2 | 96 | 2 | 1536 | 573 | 293755 | 36719 | 786432 | bf16 | PURE | FULL_SHARD | False | False | 758111232 |
|  350m | 65536 | 8 | a100_80gb | 52.39 | 52.39 | 1 | 2 | 16 | 0 | 59835 | 7479 | 1048576 | bf16 | PURE | FULL_SHARD | False | False | 420997120 |
|  350m | 32768 | 8 | a100_80gb | 47.45 | 47.45 | 2 | 2 | 32 | 3 | 98793 | 12349 | 1048576 | bf16 | PURE | FULL_SHARD | False | False | 387442688 |
|  350m | 16384 | 8 | a100_80gb | 53.01 | 53.01 | 4 | 2 | 64 | 11 | 187535 | 23441 | 1048576 | bf16 | PURE | FULL_SHARD | False | False | 370665472 |
|  350m | 8192 | 8 | a100_80gb | 53.21 | 53.21 | 8 | 2 | 128 | 35 | 289398 | 36174 | 1048576 | bf16 | PURE | FULL_SHARD | False | False | 362276864 |
|  350m | 4096 | 8 | a100_80gb | 52.46 | 52.46 | 16 | 2 | 256 | 95 | 390131 | 48766 | 1048576 | bf16 | PURE | FULL_SHARD | False | False | 358082560 |
|  350m | 2048 | 64 | a100_80gb | 47.76 | 47.76 | 32 | 1 | 2048 | 1699 | 3480601 | 54384 | 4194304 | bf16 | PURE | FULL_SHARD | False | False | 355985408 |
|  350m | 2048 | 32 | a100_80gb | 48.58 | 48.58 | 32 | 1 | 1024 | 864 | 1770287 | 55321 | 2097152 | bf16 | PURE | FULL_SHARD | False | False | 355985408 |
|  350m | 2048 | 16 | a100_80gb | 50.53 | 50.53 | 32 | 1 | 512 | 449 | 920605 | 57537 | 1048576 | bf16 | PURE | FULL_SHARD | False | False | 355985408 |
|  350m | 2048 | 8 | a100_80gb | 51.73 | 51.73 | 32 | 2 | 512 | 230 | 471290 | 58911 | 1048576 | bf16 | PURE | FULL_SHARD | False | False | 355985408 |
|  350m | 1024 | 8 | a100_80gb | 51.28 | 51.28 | 64 | 2 | 1024 | 514 | 526393 | 65799 | 1048576 | bf16 | PURE | FULL_SHARD | False | False | 354936832 |
|  350m | 512 | 8 | a100_80gb | 51.18 | 51.18 | 128 | 2 | 2048 | 1095 | 560858 | 70107 | 1048576 | bf16 | PURE | FULL_SHARD | False | False | 354412544 |
|  125m | 65536 | 8 | a100_80gb | 54.31 | 54.31 | 1 | 2 | 16 | 2 | 163472 | 20434 | 1048576 | bf16 | PURE | FULL_SHARD | False | False | 174070272 |
|  125m | 32768 | 8 | a100_80gb | 53.15 | 53.15 | 2 | 2 | 32 | 8 | 293685 | 36710 | 1048576 | bf16 | PURE | FULL_SHARD | False | False | 148904448 |
|  125m | 16384 | 8 | a100_80gb | 51.58 | 51.58 | 4 | 2 | 64 | 29 | 489578 | 61197 | 1048576 | bf16 | PURE | FULL_SHARD | False | False | 136321536 |
|  125m | 8192 | 8 | a100_80gb | 49.18 | 49.18 | 8 | 2 | 128 | 88 | 727986 | 90998 | 1048576 | bf16 | PURE | FULL_SHARD | False | False | 130030080 |
|  125m | 4096 | 8 | a100_80gb | 46.62 | 46.62 | 16 | 2 | 256 | 233 | 958343 | 119792 | 1048576 | bf16 | PURE | FULL_SHARD | False | False | 126884352 |
|  125m | 2048 | 64 | a100_80gb | 40.77 | 40.77 | 32 | 1 | 2048 | 4063 | 8321727 | 130026 | 4194304 | bf16 | PURE | FULL_SHARD | False | False | 125311488 |
|  125m | 2048 | 32 | a100_80gb | 41.22 | 41.22 | 32 | 1 | 1024 | 2053 | 4206041 | 131438 | 2097152 | bf16 | PURE | FULL_SHARD | False | False | 125311488 |
|  125m | 2048 | 16 | a100_80gb | 41.92 | 41.92 | 32 | 1 | 512 | 1044 | 2139036 | 133689 | 1048576 | bf16 | PURE | FULL_SHARD | False | False | 125311488 |
|  125m | 2048 | 8 | a100_80gb | 44.04 | 44.04 | 32 | 2 | 512 | 548 | 1123506 | 140438 | 1048576 | bf16 | PURE | FULL_SHARD | False | False | 125311488 |
|  125m | 1024 | 8 | a100_80gb | 43.25 | 43.25 | 64 | 2 | 1024 | 1225 | 1254561 | 156820 | 1048576 | bf16 | PURE | FULL_SHARD | False | False | 124525056 |
|  125m | 512 | 8 | a100_80gb | 42.54 | 42.54 | 128 | 2 | 2048 | 2587 | 1325030 | 165628 | 1048576 | bf16 | PURE | FULL_SHARD | False | False | 124131840 |

## A100 40GB

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
