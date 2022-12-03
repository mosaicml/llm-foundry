<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./assets/loss-curve-dark.png">
    <img alt="Compute-optimal training curves for LLMs of various sizes (125M -> 3B)." src="./assets/loss-curve-light.png" width="75%">
  </picture>
</p>

# Mosaic Large Language Models

This folder contains starter code for training LLMs with Composer + FSDP (in beta, use `composer>=0.11.0`).

Our goal was to build the simplest, most flexible, and still performant stack for training LLMs ([see our blog post](https://www.mosaicml.com/blog/gpt-3-quality-for-500k)).
To emphasize that flexibility, we designed this folder as a simple but feature-complete example of GPT pre-training
that you should feel free to download, fork, and customize for your application.
We even packed in a few tricks (e.g. [FlashAttention](https://github.com/HazyResearch/flash-attention)) to make training efficient, and there will be more to come!

You'll find in this folder:
* `src/mosaic_gpt.py` - a simple PyTorch GPT model, wrapped in `ComposerModel`, that can scale up to 70B parameters
* `src/data_c4.py` - a [MosaicML streaming dataset](https://streaming.docs.mosaicml.com/en/latest/) that can be used with a vanilla PyTorch dataloader.
* `main.py` - a script that builds a [Composer](https://github.com/mosaicml/composer) Trainer and calls `trainer.fit()`.
* `yamls/` - pre-baked configs for training compute-optimal LLMs from 125M up to 70B parameters.

At all model scales, we are training the exact same [vanilla PyTorch GPT model](./src/mosaic_gpt.py#L106), with no special parallelism strategies.
Composer + FSDP does all the heavy lifting to make sure we can scale up without running out of memory and while maintaining high performance.

Feel free to edit any or all of these files, and get a feel for using the LLM stack!
In `src/mosaic_gpt.py` you can see how easy it is to modify the architecture and replace a layer like `torch.nn.MultiheadAttention` with
a new one like `FlashMHA`. If you want to try and change the FSDP wrapping strategy (e.g. wrap all `GPTMLP` layers in addition to `GPTBlock`),
go ahead and [edit it here](./src/mosaic_gpt.py#L182)! You'll find a full guide on how to build custom models for Composer + FSDP under [src/README.md](./src/README.md).

Now that you've had a chance to explore the code, let's jump into actually running a training job:

# Prerequisites
Here's what you need to get started with our LLM stack:
* Use a Docker image with PyTorch 1.12+, e.g. [MosaicML's PyTorch base image](https://hub.docker.com/r/mosaicml/pytorch/tags)
   * Recommended tag: `mosaicml/pytorch:1.12.1_cu116-python3.9-ubuntu20.04`
   * This image comes pre-configured with the following dependencies:
      * PyTorch Version: 1.12.1
      * CUDA Version: 11.6
      * Python Version: 3.9
      * Ubuntu Version: 20.04
      * FlashAttention kernels from [HazyResearch](https://github.com/HazyResearch/flash-attention)
* Use a system with NVIDIA GPUs

* Install requirements via: `pip install -r requirements.txt`
  * `composer` with FSDP support (`composer>=0.11.0`)
  * `flash_attn`
  * `transformers`
  * `datasets`
  * `omegaconf`
  * `wandb`

* Prepare a local copy of the C4 dataset via instructions below.

# Dataset preparation
To run training, you'll need to make yourself a local copy of the pre-training dataset.
If you only want to profile these LLMs, we recommend that you **only download and prepare the `val` split**,
and use it for both train and eval in your script. Just change `split: train` to `split: val` in your run YAML, [e.g. here](./yamls/mosaic_gpt/125m.yaml#L32).
Alternatively, feel free to substitute our dataloader with one of your own in the entrypoint [main.py](./main.py#L93)!

In this benchmark, we train LLMs on the [C4: Colossal, Cleaned, Common Crawl dataset](https://huggingface.co/datasets/c4).
We first convert the dataset from its native format (a collection of zipped JSONs)
to MosaicML's streaming dataset format (a collection of binary `.mds` files).
Once in `.mds` format, we can store the dataset in a central location (filesystem, S3, GCS, etc.)
and stream the data to any compute cluster, with any number of devices, and any number of CPU workers, and it all ~ just works ~ .
You can read more about [the benefits of using mosaicml-streaming here](https://streaming.docs.mosaicml.com/en/latest/):

### Converting C4 to streaming dataset `.mds` format
To make yourself a copy of C4, use `convert_c4.py` like so:
```bash
# Download the 'val' split and convert to StreamingDataset format
# This will take 10 sec to 1 min depending on your Internet bandwidth
# You should see a dataset folder `./my-copy-c4/val` that is ~0.5GB
python convert_c4.py --out_root ./my-copy-c4 --splits val

# Download the 'train' split if you really want to train the model (not just profile)
# This will take 1-to-many hours depending on bandwidth, # CPUs, etc.
# The final folder `./my-copy-c4/train` will be ~800GB so make sure you have space!
python convert_c4.py --out_root ./my-copy-c4 --splits train
```

### Test the Dataloader

To verify that the dataloader works, run a quick test on your `val` split like so:

```bash
# This will construct a `StreamingC4` dataset from your `val` split,
# pass it into a PyTorch Dataloader, and iterate over it and print samples.
# Since remote and local are set to the same path, no streaming/copying takes place.
python src/data_c4.py ./my-copy-c4 ./my-copy-c4

# This will do the same thing, but stream data from {remote} -> {local}.
# The remote path can be a filesystem or object store URI.
python src/data_c4.py ./my-copy-c4 /tmp/cache-c4
python src/data_c4.py s3://my-bucket/my-copy-c4 /tmp/cache-c4
```

# How to start training

Now that you've installed dependencies and built a local copy of the C4 dataset, let's start training!

**Please remember** to edit the `data_remote` and `data_local` paths in your YAML to point to your local C4 copy.
Our streaming dataloader always streams from `data_remote` -> `data_local`, and if both paths are the same,
then no extra copying is done.

**Also remember** that if you only downloaded the `val` split, you need to make sure your train_dataloader is pointed that split.
Just change `split: train` to `split: val` in your YAML, [e.g. here](./yamls/mosaic_gpt/125m.yaml#L32).


### Single-Node training
We run the `main.py` script using our `composer` launcher, which generates N processes (1 per device).


If training on a single node, the `composer` launcher will autodetect the number of devices, so all you need to do is:

```bash
composer main.py yamls/mosaic_gpt/125m.yaml
```

To train with high performance on multi-node clusters, the easiest way is with MosaicML Cloud ;)

But if you really must try this manually on your own cluster, then just provide a few variables to `composer`
either directly via CLI, or via environment variables that can be read. Then launch the appropriate command on each node:

### Multi-Node via CLI args
```bash
# Using 2 nodes with 8 devices each
# Total world size is 16
# IP Address for Node 0 = [0.0.0.0]

# Node 0
composer --world_size 16 --node_rank 0 --master_addr 0.0.0.0 --master_port 7501 main.py yamls/mosaic_gpt/125m.yaml

# Node 1
composer --world_size 16 --node_rank 1 --master_addr 0.0.0.0 --master_port 7501 main.py yamls/mosaic_gpt/125m.yaml

```

### Multi-Node via environment variables

```bash
# Using 2 nodes with 8 devices each
# Total world size is 16
# IP Address for Node 0 = [0.0.0.0]

# Node 0
# export WORLD_SIZE=16
# export NODE_RANK=0
# export MASTER_ADDR=0.0.0.0
# export MASTER_PORT=7501
composer main.py yamls/mosaic_gpt/125m.yaml

# Node 1
# export WORLD_SIZE=16
# export NODE_RANK=1
# export MASTER_ADDR=0.0.0.0
# export MASTER_PORT=7501
composer main.py yamls/mosaic_gpt/125m.yaml
```

You should see logs being printed to your terminal like so.
You can also easily enable other experiment trackers like Weights and Biases or CometML,
by using [Composer's logging integrations](https://docs.mosaicml.com/en/v0.10.0/trainer/logging.html).

```bash
[batch=7/5000]: trainer/grad_accum: 4
[batch=7/5000]: loss/train/total: 9.8454
[batch=7/5000]: metrics/train/LanguageCrossEntropy: 9.8454
[batch=7/5000]: metrics/train/Perplexity: 18871.5430
[batch=8/5000]: wall_clock/train: 16.8334
[batch=8/5000]: wall_clock/val: 0.0000
[batch=8/5000]: wall_clock/total: 16.8334
[batch=8/5000]: lr-DecoupledAdamW/group0: 0.0000
[batch=8/5000]: trainer/global_step: 8
[batch=8/5000]: trainer/batch_idx: 8
[trace]: algorithm_traces/GradientClipping/Event.AFTER_TRAIN_BATCH:1

[batch=8/5000]: trainer/grad_accum: 4
[batch=8/5000]: loss/train/total: 9.7484
[batch=8/5000]: metrics/train/LanguageCrossEntropy: 9.7484
[batch=8/5000]: metrics/train/Perplexity: 17127.0938
[batch=9/5000]: wall_clock/train: 18.3862
[batch=9/5000]: wall_clock/val: 0.0000
[batch=9/5000]: wall_clock/total: 18.3862
[batch=9/5000]: lr-DecoupledAdamW/group0: 0.0000
[batch=9/5000]: trainer/global_step: 9
[batch=9/5000]: trainer/batch_idx: 9

train                           0%|                         | 9/5000 [00:18<2:15:12,  1.63s/ba, loss/train/total=9.7484]
```

# How many GPUs do I need to train a LLM?
This is a complicated question in general, but if we assume that you are using FSDP with `FULL_SHARD`,
activation checkpointing, but NOT `cpu_offload` (coming soon!), then a good rule of thumb is:

> Your total cluster memory in GB should be larger than  16 * N (# billions of params).

E.g. To train a GPT-13B model which has ~13 billion params,
have at least 16 * 13 = 208 GB of total memory across your GPUs.
You can accomplish this with 8xA100-40GB, or 4xA100-80GB, etc.

If you run into OOM errors when using small device counts, reduce `device_train_microbatch_size` until it succeeds.

Keep in mind: even though training will work in these minimalist settings, you will get much better throughput_per_device
if you use a larger cluster or devices with higher memory capacity,
because more memory will enable you to use larger microbatch sizes.

# Optimizing Performance
The YAMLs in this repo are relatively well tuned for medium-to-large NVIDIA A100-40GB clusters.
On different devices with more / less GPU memory,
you may wish to edit the `device_train_microbatch_size` or `fsdp_config` values.
In general, larger microbatch sizes and disabling `activation_checkpointing` lead to higher throughput.

Note that each YAML specifies a `global_train_batch_size`, which is an optimization choice, i.e. the **math** being performed,
and a `device_train_microbatch_size`, which is a system choice, i.e. how we **execute** that math.

Given these two values, our code automatically adjusts the # of gradient accumulation steps baed on the # of devices,
so you should be able to run the exact same YAML on 8 or 16 or 256 GPUs and get the same training results (within numerics).
This is nice because it means you can write device-count-agnostic training configs,
and not worry about OOM-ing or accidentally changing the optimization math.

In previous blogs ([1](https://www.mosaicml.com/blog/farewell-oom), [2](https://www.mosaicml.com/blog/billion-parameter-gpt-training-made-easy))
we also demonstrated Auto Grad Accum, which takes things a step further by letting Composer determine the `device_train_microbatch_size` on its own.
This makes our configs not only device-count-agnostic, but hardware-agnostic too!
You can try out this feature by setting `device_train_microbatch_size: auto`, but bear in mind that FSDP support is still in alpha mode
and may not always work with Auto Grad Accum (but we are working on it!).

# Contact Us
If you run into any problems with the code, please file Github issues directly to this repo.

you want train LLMs on MosaicML Cloud, reach out to us at [llm-early-access@mosaicml.com](mailto:llm-early-access@mosaicml.com)!
