|  Model | SeqLen (T) | # GPUs | GPU | MFU | HFU | MicroBatchSize | GradAccum | GlobalBatchSize | Throughput (S/s) | Throughput (T/s) | Throughput (T/s/GPU) | GlobalBatchSize (T) | Precision | MP Mode | Sharding Strategy | Activation Checkpointing | Activation CPUOffload | NumParams |
|  --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  13b | 2048 | 8 | a100_80gb | 58.67 | 58.67 | 2 | 2 | 32 | 8 | 17824 | 2228 | 65536 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 12853954560 |
|  13b | 2048 | 8 | a100_80gb | 49.18 | 49.18 | 1 | 2 | 16 | 7 | 14942 | 1867 | 32768 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 12853954560 |
|  7b | 2048 | 8 | a100_80gb | 62.46 | 62.46 | 4 | 2 | 64 | 17 | 36110 | 4513 | 131072 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 6658859008 |
|  7b | 2048 | 8 | a100_80gb | 55.97 | 55.97 | 2 | 2 | 32 | 15 | 32355 | 4044 | 65536 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 6658859008 |
|  7b | 2048 | 8 | a100_80gb | 46.59 | 46.59 | 1 | 2 | 16 | 13 | 26934 | 3366 | 32768 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 6658859008 |
|  3b | 2048 | 8 | a100_40gb | 40.65 | 40.65 | 1 | 2 | 16 | 27 | 56609 | 7076 | 32768 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 2651837440 |
|  3b | 2048 | 8 | a100_80gb | 34.93 | 46.57 | 1 | 2 | 16 | 23 | 48635 | 6079 | 32768 | amp_bf16 | DEFAULT | FULL_SHARD | True | True | 2651837440 |
|  3b | 2048 | 8 | a100_80gb | 35.02 | 46.69 | 1 | 2 | 16 | 23 | 48759 | 6094 | 32768 | amp_bf16 | DEFAULT | FULL_SHARD | True | True | 2651837440 |
