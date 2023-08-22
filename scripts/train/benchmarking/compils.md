|  Model | SeqLen (T) | # GPUs | GPU | MFU | HFU | MicroBatchSize | GradAccum | GlobalBatchSize | Throughput (S/s) | Throughput (T/s) | Throughput (T/s/GPU) | GlobalBatchSize (T) | Precision | MP Mode | Sharding Strategy | Activation Checkpointing | Activation CPUOffload | NumParams | Compile_Config |
|  --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  13b | 2048 | 8 | a100_80gb | 51.73 | 68.98 | 20 | 3 | 480 | 7 | 15716 | 1964 | 983040 | amp_bf16 | DEFAULT | FULL_SHARD | True | False | 12853954560 | None |
|  760m | 2048 | 8 | a100_80gb | 51.26 | 51.26 | 24 | 2 | 384 | 114 | 233935 | 29241 | 786432 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 760470528 | None |
|  760m | 2048 | 8 | a100_80gb | 43.15 | 43.15 | 12 | 2 | 192 | 96 | 196920 | 24615 | 393216 | amp_bf16 | DEFAULT | FULL_SHARD | False | False | 760470528 | {} |
