## Warning: The numbers in the tables below are based on torch 1.12 and are out of date. The current setup achieves higher throughput / MFU. This warning will be removed once the tables are updated.

# MosaicGPT Training Benchmarks

Benchmark measurements for MosaicGPT models trained on [MosaicML Cloud](https://www.mosaicml.com/cloud), including throughput, MFU, and HFU. Each model is based on optimized configurations of various sizes in the [yamls](../yamls) folder, ranging from a 125m to 70B parameter models.

To reproduce these results, first:
```
python submit_benchmarks.py -m 125m.yaml --cluster [your_mosaicml_cluster] --RUN
```

will use our Python API to submit a sweep of configurations for the 125M parameter model. To run all the configurations, omit the `-m` flag.

Then, after the runs are completed:
```
python collect_results.py --save-path results
```
will use our Python API to collect and calculate the benchmark results, and then save as both a CSV file `results.csv`, and a markdown table `results.md`.

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

Note that these are approximations. Actual HFU would be higher since it includes the floating poit operations for normalization, activation, and residual lyaers, as well as **all** recomputation. For example, our models use Flash Attention, which requires including an extra recompute factor for its recomputation in the forward pass. Therefore, the attention multipler would be 5 instead of 4.

## Results

Below we include several configurations across different hardware platforms, sequence lengths and batch sizes. It is easy to benchmark configurations for your own use case. For example, using Mosaic Cloud, to test MosaicGPT {13B, 30B} using fp16 with a batch size of 2M tokens and seq len {2k, 4k, 8k, 16k} run:
```
python submit_benchmarks.py -m 13b.yaml 30b.yaml -t fp16 -b 21 21 -s 11 14 --RUN
```
This will run 8 configs for 12 steps to get throughput numbers. `python collect_results.py` can then be used to parse all output training logs and create the tables below.

## A100 80GB

| Model   | SeqLen (T) | GPUType       | MFU*   | MFU    | HFU*   | HFU    | Throughput (T/s) | GPUThroughput (T/s/GPU) | Throughput (S/s) | ParamCount  | GlobalBatchSize (T) | GlobalBatchSize (S) | MicroBatchSize (S) | GradAccum | ShardStrategy | ActCkpt | ActCPUoffload | Precision | MP Mode | NumGPUs | GPUType   |
| ------- | ---------- | ------------- | ------ | ------ | ------ | ------ | ---------------- | ----------------------- | ---------------- | ----------- | ------------------- | ------------------- | ------------------ | --------- | ------------- | ------- | ------------- | --------- | ------- | ------- | --------- |
|  gpt70b |       2048 | 128xA100_80GB | 0.4096 | 0.4265 | 0.5461 | 0.5687 |         42028.81 |                328.3501 |          20.5219 | 64861528064 |             2097152 |                1024 |                  8 |         1 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |     128 | a100_80gb |
|  gpt30b |       8192 |  64xA100_80GB | 0.3790 | 0.4502 | 0.5054 | 0.6003 |         42023.16 |                656.6118 |           5.1298 | 30018458624 |             1048576 |                 128 |                  2 |         1 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |      64 | a100_80gb |
|  gpt30b |       8192 |  32xA100_80GB | 0.3734 | 0.4435 | 0.4979 | 0.5914 |         20698.23 |                646.8198 |           2.5266 | 30018458624 |              524288 |                  64 |                  2 |         1 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |      32 | a100_80gb |
|  gpt30b |       4096 | 128xA100_80GB | 0.4249 | 0.4649 | 0.5666 | 0.6198 |         94312.24 |                736.8144 |          23.0254 | 29989098496 |             2097152 |                 512 |                  4 |         1 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |     128 | a100_80gb |
|  gpt30b |       4096 |  32xA100_80GB | 0.4245 | 0.4644 | 0.5660 | 0.6192 |         23553.76 |                736.0550 |           5.7504 | 29989098496 |              524288 |                 128 |                  4 |         1 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |      32 | a100_80gb |
|  gpt30b |       2048 | 256xA100_80GB | 0.4464 | 0.4674 | 0.5952 | 0.6232 |        198267.62 |                774.4829 |          96.8104 | 29974418432 |             4194304 |                2048 |                  8 |         1 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |     256 | a100_80gb |
|  gpt30b |       2048 | 128xA100_80GB | 0.4548 | 0.4762 | 0.6064 | 0.6349 |        100995.65 |                789.0285 |          49.3143 | 29974418432 |             2097152 |                1024 |                  8 |         1 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |     128 | a100_80gb |
|  gpt30b |       2048 |  64xA100_80GB | 0.4562 | 0.4777 | 0.6083 | 0.6369 |         50651.59 |                791.4310 |          24.7322 | 29974418432 |             1048576 |                 512 |                  8 |         1 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |      64 | a100_80gb |
|  gpt30b |       2048 |  32xA100_80GB | 0.4580 | 0.4796 | 0.6107 | 0.6394 |         25427.85 |                794.6202 |          12.4159 | 29974418432 |              524288 |                 256 |                  8 |         1 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |      32 | a100_80gb |
|  gpt30b |       2048 |  16xA100_80GB | 0.4562 | 0.4777 | 0.6083 | 0.6369 |         12663.75 |                791.4842 |           6.1835 | 29974418432 |              262144 |                 128 |                  8 |         1 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |      16 | a100_80gb |
|  gpt13b |       8192 | 128xA100_80GB | 0.3316 | 0.4180 | 0.4421 | 0.5573 |        171301.11 |               1338.2899 |          20.9108 | 12884843520 |             2097152 |                 256 |                  2 |         1 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |     128 | a100_80gb |
|  gpt13b |       8192 |  64xA100_80GB | 0.3351 | 0.4224 | 0.4468 | 0.5631 |         86550.86 |               1352.3571 |          10.5653 | 12884843520 |             1048576 |                 128 |                  2 |         1 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |      64 | a100_80gb |
|  gpt13b |       8192 |  32xA100_80GB | 0.3382 | 0.4263 | 0.4510 | 0.5684 |         43682.94 |               1365.0918 |           5.3324 | 12884843520 |              524288 |                  64 |                  2 |         1 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |      32 | a100_80gb |
|  gpt13b |       4096 |  64xA100_80GB | 0.3906 | 0.4416 | 0.5208 | 0.5888 |        101058.89 |               1579.0451 |          24.6726 | 12863872000 |             1048576 |                 256 |                  4 |         1 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |      64 | a100_80gb |
|  gpt13b |       4096 |  32xA100_80GB | 0.3878 | 0.4384 | 0.5171 | 0.5846 |         50169.12 |               1567.7850 |          12.2483 | 12863872000 |              524288 |                 128 |                  4 |         1 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |      32 | a100_80gb |
|  gpt13b |       2048 | 128xA100_80GB | 0.4226 | 0.4502 | 0.5635 | 0.6003 |        218852.00 |               1709.7813 |         106.8613 | 12853386240 |             2097152 |                1024 |                  8 |         1 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |     128 | a100_80gb |
|  gpt13b |       2048 |  64xA100_80GB | 0.4251 | 0.4529 | 0.5668 | 0.6038 |        110076.33 |               1719.9427 |          53.7482 | 12853386240 |             1048576 |                 512 |                  8 |         1 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |      64 | a100_80gb |
|  gpt13b |       2048 |  32xA100_80GB | 0.4197 | 0.4471 | 0.5596 | 0.5961 |         54335.55 |               1697.9859 |          26.5310 | 12853386240 |              524288 |                 256 |                  8 |         1 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |      32 | a100_80gb |
|  gpt13b |       2048 |  16xA100_80GB | 0.4274 | 0.4553 | 0.5699 | 0.6071 |         27666.72 |               1729.1699 |          13.5091 | 12853386240 |              262144 |                 128 |                  8 |         1 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |      16 | a100_80gb |
|  gpt13b |      16384 |   8xA100_80GB | 0.2691 | 0.4088 | 0.3588 | 0.5451 |          8659.76 |               1082.4704 |           0.5285 | 12926786560 |             4194304 |                 256 |                  1 |        32 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |       8 | a100_80gb |
|  gpt13b |       8192 |   8xA100_80GB | 0.3424 | 0.4316 | 0.4566 | 0.5755 |         11055.68 |               1381.9597 |           1.3496 | 12884843520 |             4194304 |                 512 |                  2 |        32 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |       8 | a100_80gb |
|  gpt13b |       4096 |   8xA100_80GB | 0.3974 | 0.4492 | 0.5298 | 0.5989 |         12850.50 |               1606.3130 |           3.1373 | 12863872000 |             4194304 |                1024 |                  4 |        32 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |       8 | a100_80gb |
|  gpt13b |       2048 |   8xA100_80GB | 0.4304 | 0.4585 | 0.5738 | 0.6113 |         13929.35 |               1741.1686 |           6.8014 | 12853386240 |             4194304 |                2048 |                  8 |        32 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |       8 | a100_80gb |
|  gpt13b |        512 |   8xA100_80GB | 0.4632 | 0.4708 | 0.6176 | 0.6277 |         15001.49 |               1875.1866 |          29.2998 | 12845521920 |              524288 |                1024 |                 32 |         4 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |       8 | a100_80gb |
|   gpt7b |      16384 |   8xA100_80GB | 0.2406 | 0.3945 | 0.3208 | 0.5260 |         14901.90 |               1862.7379 |           0.9095 |  6717124608 |             4194304 |                 256 |                  2 |        16 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |       8 | a100_80gb |
|   gpt7b |       8192 |   8xA100_80GB | 0.3120 | 0.4123 | 0.4161 | 0.5497 |         19422.58 |               2427.8221 |           2.3709 |  6683570176 |             4194304 |                 512 |                  4 |        16 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |       8 | a100_80gb |
|   gpt7b |       4096 |   8xA100_80GB | 0.3741 | 0.4344 | 0.4988 | 0.5791 |         23343.51 |               2917.9392 |           5.6991 |  6666792960 |             4194304 |                1024 |                  8 |        16 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |       8 | a100_80gb |
|   gpt7b |       2048 |   8xA100_80GB | 0.4106 | 0.4437 | 0.5474 | 0.5916 |         25651.12 |               3206.3898 |          12.5250 |  6658404352 |             4194304 |                2048 |                 16 |        16 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |       8 | a100_80gb |
|   gpt3b |      16384 |   8xA100_80GB | 0.1829 | 0.3654 | 0.2438 | 0.4872 |         28295.66 |               3536.9574 |           1.7270 |  2688253440 |             4194304 |                 256 |                  4 |         8 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |       8 | a100_80gb |
|   gpt3b |       8192 |   8xA100_80GB | 0.2522 | 0.3792 | 0.3363 | 0.5056 |         39341.83 |               4917.7293 |           4.8025 |  2667281920 |             4194304 |                 512 |                  8 |         8 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |       8 | a100_80gb |
|   gpt3b |       4096 |   8xA100_80GB | 0.3140 | 0.3933 | 0.4187 | 0.5245 |         49169.98 |               6146.2477 |          12.0044 |  2656796160 |             4194304 |                1024 |                 16 |         8 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |       8 | a100_80gb |
|   gpt3b |       2048 |   8xA100_80GB | 0.3551 | 0.4000 | 0.4735 | 0.5334 |         55710.27 |               6963.7837 |          27.2023 |  2651553280 |             4194304 |                2048 |                 32 |         8 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |       8 | a100_80gb |
|   gpt1b |      16384 |   8xA100_80GB | 0.1579 | 0.3471 | 0.2106 | 0.4627 |         48846.93 |               6105.8662 |           2.9814 |  1345083392 |             4194304 |                 256 |                  4 |         8 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |       8 | a100_80gb |
|   gpt1b |       8192 |   8xA100_80GB | 0.2206 | 0.3544 | 0.2942 | 0.4725 |         69094.52 |               8636.8154 |           8.4344 |  1328306176 |             4194304 |                 512 |                  8 |         8 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |       8 | a100_80gb |
|   gpt1b |       4096 |   8xA100_80GB | 0.2806 | 0.3662 | 0.3741 | 0.4882 |         88432.31 |              11054.0390 |          21.5899 |  1319917568 |             4194304 |                1024 |                 16 |         8 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |       8 | a100_80gb |
|   gpt1b |       2048 |   8xA100_80GB | 0.3187 | 0.3674 | 0.4249 | 0.4899 |        100751.50 |              12593.9379 |          49.1951 |  1315723264 |             4194304 |                2048 |                 32 |         8 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |       8 | a100_80gb |
| gpt760m |      16384 |   8xA100_80GB | 0.1741 | 0.4429 | 0.1741 | 0.4429 |         92578.78 |              11572.3469 |           5.6506 |   782320128 |             4194304 |                 256 |                  2 |        16 |    FULL_SHARD |   False |         False |  amp_bf16 | DEFAULT |       8 | a100_80gb |
| gpt350m |      16384 |   8xA100_80GB | 0.1465 | 0.4648 | 0.1465 | 0.4648 |        164446.54 |              20555.8170 |          10.0370 |   370551808 |             4194304 |                 256 |                  2 |        16 |    FULL_SHARD |   False |         False |  amp_bf16 | DEFAULT |       8 | a100_80gb |
| gpt125m |      16384 |   8xA100_80GB | 0.1266 | 0.4072 | 0.1266 | 0.4072 |        386517.07 |              48314.6342 |          23.5911 |   136236288 |             4194304 |                 256 |                  4 |         8 |    FULL_SHARD |   False |         False |  amp_bf16 | DEFAULT |       8 | a100_80gb |

## A100 40GB

| Model   | SeqLen (T) | GPUType       | MFU*   | MFU    | HFU*   | HFU    | Throughput (T/s) | GPUThroughput (T/s/GPU) | Throughput (S/s) | ParamCount  | GlobalBatchSize (T) | GlobalBatchSize (S) | MicroBatchSize (S) | GradAccum | ShardStrategy | ActCkpt | ActCPUoffload | Precision | MP Mode | NumGPUs | GPUType   |
| ------- | ---------- | ------------- | ------ | ------ | ------ | ------ | ---------------- | ----------------------- | ---------------- | ----------- | ------------------- | ------------------- | ------------------ | --------- | ------------- | ------- | ------------- | --------- | ------- | ------- | --------- |
|   gpt7b |      16384 |  16xA100_40GB | 0.2288 | 0.3750 | 0.3050 | 0.5001 |         28335.96 |               1770.9978 |           1.7295 |  6717124608 |             8388608 |                 512 |                  1 |        32 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |      16 | a100_40gb |
|   gpt7b |      16384 |   8xA100_40GB | 0.2133 | 0.3497 | 0.2844 | 0.4662 |         13209.76 |               1651.2205 |           0.8063 |  6717124608 |             8388608 |                 512 |                  1 |        64 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |       8 | a100_40gb |
|   gpt7b |       8192 |  16xA100_40GB | 0.2798 | 0.3697 | 0.3730 | 0.4929 |         34828.29 |               2176.7680 |           4.2515 |  6683570176 |             8388608 |                1024 |                  1 |        64 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |      16 | a100_40gb |
|   gpt7b |       8192 |   8xA100_40GB | 0.2836 | 0.3748 | 0.3782 | 0.4997 |         17654.50 |               2206.8122 |           2.1551 |  6683570176 |             8388608 |                1024 |                  1 |       128 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |       8 | a100_40gb |
|   gpt7b |       4096 |  16xA100_40GB | 0.3503 | 0.4068 | 0.4671 | 0.5423 |         43721.20 |               2732.5747 |          10.6741 |  6666792960 |             8388608 |                2048 |                  4 |        32 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |      16 | a100_40gb |
|   gpt7b |       4096 |   8xA100_40GB | 0.3321 | 0.3856 | 0.4428 | 0.5141 |         20723.96 |               2590.4947 |           5.0596 |  6666792960 |             8388608 |                2048 |                  2 |       128 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |       8 | a100_40gb |
|   gpt7b |       2048 |  16xA100_40GB | 0.3839 | 0.4149 | 0.5119 | 0.5531 |         47970.51 |               2998.1568 |          23.4231 |  6658404352 |             8388608 |                4096 |                  8 |        32 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |      16 | a100_40gb |
|   gpt7b |       2048 |   8xA100_40GB | 0.3628 | 0.3921 | 0.4838 | 0.5228 |         22668.72 |               2833.5898 |          11.0687 |  6658404352 |             8388608 |                4096 |                  4 |       128 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |       8 | a100_40gb |
|   gpt7b |       1024 |  16xA100_40GB | 0.4055 | 0.4219 | 0.5407 | 0.5625 |         50702.28 |               3168.8928 |          49.5140 |  6654210048 |             8388608 |                8192 |                 16 |        32 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |      16 | a100_40gb |
|   gpt7b |       1024 |   8xA100_40GB | 0.3760 | 0.3912 | 0.5014 | 0.5216 |         23507.13 |               2938.3910 |          22.9562 |  6654210048 |             8388608 |                8192 |                  8 |       128 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |       8 | a100_40gb |
|   gpt7b |       1024 |   8xA100_40GB | 0.3782 | 0.3935 | 0.5043 | 0.5246 |         23644.47 |               2955.5584 |          23.0903 |  6654210048 |              524288 |                 512 |                  8 |         8 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |       8 | a100_40gb |
|   gpt3b |      16384 |  16xA100_40GB | 0.1749 | 0.3496 | 0.2333 | 0.4662 |         54143.88 |               3383.9923 |           3.3047 |  2688253440 |             8388608 |                 512 |                  1 |        32 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |      16 | a100_40gb |
|   gpt3b |      16384 |   8xA100_40GB | 0.1754 | 0.3505 | 0.2338 | 0.4673 |         27136.33 |               3392.0410 |           1.6563 |  2688253440 |             8388608 |                 512 |                  1 |        64 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |       8 | a100_40gb |
|   gpt3b |       8192 |  16xA100_40GB | 0.2353 | 0.3537 | 0.3137 | 0.4716 |         73389.42 |               4586.8390 |           8.9587 |  2667281920 |             8388608 |                1024 |                  2 |        32 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |      16 | a100_40gb |
|   gpt3b |       8192 |   8xA100_40GB | 0.2379 | 0.3576 | 0.3172 | 0.4768 |         37105.58 |               4638.1978 |           4.5295 |  2667281920 |             8388608 |                1024 |                  2 |        64 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |       8 | a100_40gb |
|   gpt3b |       4096 |  16xA100_40GB | 0.2872 | 0.3598 | 0.3830 | 0.4797 |         89949.31 |               5621.8317 |          21.9603 |  2656796160 |             8388608 |                2048 |                  4 |        32 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |      16 | a100_40gb |
|   gpt3b |       4096 |   8xA100_40GB | 0.2925 | 0.3664 | 0.3901 | 0.4886 |         45806.55 |               5725.8189 |          11.1832 |  2656796160 |             8388608 |                2048 |                  4 |        64 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |       8 | a100_40gb |
|   gpt3b |       2048 |  16xA100_40GB | 0.3241 | 0.3651 | 0.4321 | 0.4867 |        101680.35 |               6355.0221 |          49.6486 |  2651553280 |             8388608 |                4096 |                  8 |        32 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |      16 | a100_40gb |
|   gpt3b |       2048 |   8xA100_40GB | 0.3312 | 0.3731 | 0.4416 | 0.4975 |         51966.01 |               6495.7517 |          25.3740 |  2651553280 |             8388608 |                4096 |                  8 |        64 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |       8 | a100_40gb |
|   gpt3b |       1024 |  16xA100_40GB | 0.3496 | 0.3717 | 0.4661 | 0.4956 |        109800.44 |               6862.5274 |         107.2270 |  2648931840 |             8388608 |                8192 |                 16 |        32 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |      16 | a100_40gb |
|   gpt3b |       1024 |   8xA100_40GB | 0.3595 | 0.3822 | 0.4793 | 0.5096 |         56451.90 |               7056.4877 |          55.1288 |  2648931840 |             8388608 |                8192 |                 16 |        64 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |       8 | a100_40gb |
|   gpt3b |        512 |  16xA100_40GB | 0.3625 | 0.3740 | 0.4833 | 0.4986 |        113903.78 |               7118.9862 |         222.4683 |  2647621120 |             8388608 |               16384 |                 32 |        32 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |      16 | a100_40gb |
|   gpt3b |        512 |   8xA100_40GB | 0.3699 | 0.3817 | 0.4933 | 0.5089 |         58127.21 |               7265.9008 |         113.5297 |  2647621120 |             8388608 |               16384 |                 32 |        64 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |       8 | a100_40gb |
|   gpt1b |      16384 |  16xA100_40GB | 0.1538 | 0.3379 | 0.2051 | 0.4506 |         95127.80 |               5945.4874 |           5.8061 |  1345083392 |             8388608 |                 512 |                  2 |        16 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |      16 | a100_40gb |
|   gpt1b |      16384 |   8xA100_40GB | 0.1551 | 0.3409 | 0.2069 | 0.4545 |         47981.69 |               5997.7114 |           2.9286 |  1345083392 |             8388608 |                 512 |                  2 |        32 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |       8 | a100_40gb |
|   gpt1b |       8192 |  16xA100_40GB | 0.2117 | 0.3400 | 0.2822 | 0.4534 |        132589.32 |               8286.8326 |          16.1852 |  1328306176 |             8388608 |                1024 |                  4 |        16 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |      16 | a100_40gb |
|   gpt1b |       8192 |   8xA100_40GB | 0.2133 | 0.3426 | 0.2844 | 0.4568 |         66805.43 |               8350.6790 |           8.1550 |  1328306176 |             8388608 |                1024 |                  4 |        32 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |       8 | a100_40gb |
|   gpt1b |       4096 |  16xA100_40GB | 0.2648 | 0.3456 | 0.3531 | 0.4608 |        166925.89 |              10432.8678 |          40.7534 |  1319917568 |             8388608 |                2048 |                  8 |        16 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |      16 | a100_40gb |
|   gpt1b |       4096 |   8xA100_40GB | 0.2684 | 0.3503 | 0.3579 | 0.4671 |         84606.61 |              10575.8259 |          20.6559 |  1319917568 |             8388608 |                2048 |                  8 |        32 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |       8 | a100_40gb |
|   gpt1b |       2048 |  16xA100_40GB | 0.3047 | 0.3513 | 0.4063 | 0.4684 |        192671.44 |              12041.9648 |          94.0778 |  1315723264 |             8388608 |                4096 |                 16 |        16 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |      16 | a100_40gb |
|   gpt1b |       2048 |   8xA100_40GB | 0.3100 | 0.3574 | 0.4133 | 0.4766 |         98012.45 |              12251.5558 |          47.8576 |  1315723264 |             8388608 |                4096 |                 16 |        32 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |       8 | a100_40gb |
|   gpt1b |       1024 |  16xA100_40GB | 0.3324 | 0.3578 | 0.4431 | 0.4771 |        210501.37 |              13156.3354 |         205.5677 |  1313626112 |             8388608 |                8192 |                 32 |        16 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |      16 | a100_40gb |
|   gpt1b |       1024 |   8xA100_40GB | 0.3380 | 0.3639 | 0.4506 | 0.4851 |        107026.65 |              13378.3309 |         104.5182 |  1313626112 |             8388608 |                8192 |                 32 |        32 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |       8 | a100_40gb |
|   gpt1b |        512 |  16xA100_40GB | 0.3464 | 0.3597 | 0.4618 | 0.4796 |        219562.21 |              13722.6381 |         428.8324 |  1312577536 |             8388608 |               16384 |                 64 |        16 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |      16 | a100_40gb |
|   gpt1b |        512 |   8xA100_40GB | 0.3538 | 0.3674 | 0.4718 | 0.4899 |        112139.27 |              14017.4086 |         219.0220 |  1312577536 |             8388608 |               16384 |                 64 |        32 |    FULL_SHARD |    True |         False |  amp_bf16 | DEFAULT |       8 | a100_40gb |
| gpt760m |      16384 |   8xA100_40GB | 0.1032 | 0.2627 | 0.1032 | 0.2627 |         54899.02 |               6862.3770 |           3.3508 |   782320128 |             8388608 |                 512 |                  1 |        64 |    FULL_SHARD |   False |         False |  amp_bf16 | DEFAULT |       8 | a100_40gb |
| gpt350m |      16384 |   8xA100_40GB | 0.1416 | 0.4493 | 0.1416 | 0.4493 |        158971.66 |              19871.4573 |           9.7029 |   370551808 |             8388608 |                 512 |                  1 |        64 |    FULL_SHARD |   False |         False |  amp_bf16 | DEFAULT |       8 | a100_40gb |
| gpt125m |      16384 |   8xA100_40GB | 0.1251 | 0.4023 | 0.1251 | 0.4023 |        381870.41 |              47733.8010 |          23.3075 |   136236288 |             8388608 |                 512 |                  2 |        32 |    FULL_SHARD |   False |         False |  amp_bf16 | DEFAULT |       8 | a100_40gb |
