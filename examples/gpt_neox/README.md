# GPT-NeoX on the Mosaic Platform

[The Mosaic platform](https://www.mosaicml.com/blog/mosaicml-cloud-demo) enables easy training of distributed machine learning (ML) jobs. In this folder, we provide an example of how to run [GPT-NeoX](https://github.com/EleutherAI/gpt-neox), EleutherAI's library for training large language models, on the Mosaic platform.

You’ll find in this folder:

-   `multi_node.yaml` - a yaml to run a multi-node GPT-NeoX training job on the Mosaic platform.

## Prerequisites

Here’s what you’ll need to get started with running GPT-NeoX on the Mosaic platform

-   A docker image with the correctly installed GPT-NeoX dependencies (we tested with `shivanshupurohit/gpt-neox:112`).
-   A dataset prepared in the [expected format](https://github.com/EleutherAI/gpt-neox/blob/72c80715c366cc4ad623050d6bcb984fe6638814/README.md?plain=1#L122).

## Starting Training
We include the `.yaml` file required to run multi-node GPT-NeoX on the Mosaic platform. You just need to fill in the `cluster` field in the `.yaml` files, and change the `data-path` if using your own data. If you are using Weights & Biases, fill in `wandb_project` and `wandb_team`, otherwise remove the Weights & Biases related arguments (`use_wandb`, `wandb_project`, `wandb_team`, and `wandb_group`). The other GPT-NeoX configs can be modified as usual. The provided yaml file uses 16 GPUs, but all you have to do to use more is change the `gpu_num` field. You will likely want to adjust the parallelism configuration for your exact setup. See the [GPT-NeoX README](https://github.com/EleutherAI/gpt-neox/blob/main/README.md) for more information.

************Multi-Node Jobs************

Running a multi-node job is as simple as running `mcli run -f multi_node.yaml`.

There are a lot of logs emitted, but early on you should see something like

```
[2023-02-27 23:33:57,571] [INFO] [launch.py:82:main] WORLD INFO DICT: {'node-0': [0, 1, 2, 3, 4, 5, 6, 7], 'node-1': [0, 1, 2, 3, 4, 5, 6, 7]}
[2023-02-27 23:33:57,571] [INFO] [launch.py:88:main] nnodes=2, num_local_procs=8, node_rank=0
[2023-02-27 23:33:57,571] [INFO] [launch.py:103:main] global_rank_mapping=defaultdict(<class 'list'>, {'node-0': [0, 1, 2, 3, 4, 5, 6, 7], 'node-1': [8, 9, 10, 11, 12, 13, 14, 15]})
[2023-02-27 23:33:57,571] [INFO] [launch.py:104:main] dist_world_size=16
[2023-02-27 23:33:57,571] [INFO] [launch.py:112:main] Setting CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

and then once training has started

```
%comms: 4.818873821517855
 %optimizer_step 0.7121011008320476
 %forward: 19.83046088011732
 %backward: 69.33287269883512
[2023-02-27 23:34:53,987] [INFO] [logging.py:60:log_dist] [Rank 0] rank=0 time (ms) | train_batch: 0.00 | batch_input: 7.57 | forward: 519.08 | backward_microstep: 1815.10 | backward: 1814.87 | backward_inner_microstep: 1814.17 | backward_inner: 1813.91 | backward_allreduce_microstep: 0.35 | backward_allreduce: 0.12 | reduce_tied_grads: 0.33 | comms: 126.14 | reduce_grads: 125.80 | step: 18.64 | _step_clipping: 0.10 | _step_step: 17.40 | _step_zero_grad: 0.36 | _step_check_overflow: 0.21
[2023-02-27 23:34:56,819] [INFO] [logging.py:60:log_dist] [Rank 0] step=30, skipped=20, lr=[1.8749999999999998e-06, 1.8749999999999998e-06], mom=[[0.9, 0.95], [0.9, 0.95]]
steps: 30 loss: 10.7133 iter time (s): 0.283 samples/sec: 226.457
```
