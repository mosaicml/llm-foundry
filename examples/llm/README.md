<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./assets/loss-curve-dark.png">
    <img alt="Compute-optimal training curves for LLMs of various sizes (125M -> 3B)." src="./assets/loss-curve-light.png" width="75%">
  </picture>
</p>

# Mosaic Large Language Models

This folder contains starter code for training LLMs with Composer + FSDP.

Our goal was to build the simplest, most flexible, and still performant stack for training LLMs ([see our blog post](https://www.mosaicml.com/blog/gpt-3-quality-for-500k)).
To emphasize that flexibility, we designed this folder as a simple but feature-complete example of GPT pre-training
that you should feel free to download, fork, and customize for your application.
We even packed in a few tricks (e.g. [FlashAttention](https://github.com/HazyResearch/flash-attention)) to make training efficient, and there will be more to come!

You'll find in this folder:
* `src/models/mosaic_gpt/` - a simple PyTorch GPT model, wrapped in `ComposerModel`, that can scale up to 70B+ parameters
* `main.py` - a training script that builds a [Composer](https://github.com/mosaicml/composer) `Trainer` and calls `trainer.fit()`.
* `yamls/` - configs for training compute-optimal LLMs from 125M up to 70B parameters.
* `throughput/` - data on the training throughput of MosaicGPT on different cluster configurations.
* `inference/` - scripts to convert models to HuggingFace or ONNX format, and view generations.
* `mcloud/` - examples of how to use [MosaicML platform](https://www.mosaicml.com/platform) to seamlessly launch training, eval, and inference jobs :)


In the [common](../common) folder, you will also find:
* `common/builders.py`- A collection of convenient string-to-object mappings used to create objects that get passed to `Trainer`.
* `common/text_data.py`- a [MosaicML streaming dataset](https://streaming.docs.mosaicml.com/en/stable/) that can be used with a vanilla PyTorch dataloader.
* `common/convert_dataset.py`- an example of converting generic text data into `StreamingDataset` `.mds` shard files.

At all model scales, we are training the exact same [vanilla PyTorch GPT model](./src/mosaic_gpt.py#L106), with no special parallelism strategies.
Composer + FSDP does all the heavy lifting to make sure we can scale up without running out of memory and while maintaining high performance.

Feel free to edit any or all of these files, and get a feel for using the LLM stack!
In `src/mosaic_gpt.py` you can see how easy it is to modify the architecture and replace a layer like `torch.nn.MultiheadAttention` with
a new one like `FlashMHA`. If you want to try and change the FSDP wrapping strategy (e.g. wrap all `GPTMLP` layers in addition to `GPTBlock`),
go ahead and [edit it here](./src/mosaic_gpt.py#L182)! You'll find a full guide on how to build custom models for Composer + FSDP under [src/README.md](./src/README.md).

Now that you've had a chance to explore the code, let's jump into actually running a training job:

# Prerequisites
Here's what you need to get started with our LLM stack:
* Use a Docker image with PyTorch 1.13+, e.g. [MosaicML's PyTorch base image](https://hub.docker.com/r/mosaicml/pytorch/tags)
   * Recommended tag: `mosaicml/pytorch:1.13.1_cu117-python3.10-ubuntu20.04`
   * This image comes pre-configured with the following dependencies:
      * PyTorch Version: 1.13.1
      * CUDA Version: 11.7
      * Python Version: 3.10
      * Ubuntu Version: 20.04
      * FlashAttention kernels from [HazyResearch](https://github.com/HazyResearch/flash-attention)
* Use a system with NVIDIA GPUs

# Installation

To get started, clone this repo and install the requirements:

```bash
git clone https://github.com/mosaicml/examples.git
cd examples
pip install -e ".[llm]"  # or pip install -e ".[llm-cpu]" if no NVIDIA GPU
cd examples/llm
```

# Dataset preparation
To run training, you'll need to make yourself a copy of the pre-training dataset.
If you only want to profile these LLMs, we recommend that you **download and prepare the `train_small` and `val` splits**,
and skip the full `train` split. You'll just need to replace `split: train` with `split: train_small` in your run YAML, [e.g. here](./yamls/mosaic_gpt/125m.yaml#L40).
You can also accomplish this in your CLI command like so: `composer main.py ... train_loader.dataset.split=train_small`
Alternatively, feel free to substitute our dataloader with one of your own in [main.py](./main.py#L96)!

As an example, we train LLMs on the [C4: Colossal, Cleaned, Common Crawl dataset](https://huggingface.co/datasets/c4).
We first convert the dataset from its native format (a collection of zipped JSONs)
to MosaicML's streaming dataset format (a collection of binary `.mds` files).
Once in `.mds` format, we can store the dataset in a central location (filesystem, S3, GCS, etc.)
and stream the data to any compute cluster, with any number of devices, and any number of CPU workers, and it all ~ just works ~ .
You can read more about [the benefits of using mosaicml-streaming here](https://streaming.docs.mosaicml.com/en/stable/):

### Converting C4 to streaming dataset `.mds` format
To make yourself a copy of C4, use `convert_dataset.py` like so:
```bash
# Download the 'train_small' and 'val' splits and convert to StreamingDataset format
# This will take 20-60 seconds depending on your Internet bandwidth
# You should see two folders: `./my-copy-c4/train_small` and `./my-copy-c4/val` that are each ~0.5GB
# Note: We are using the `--concat_tokens` option to pre tokenize our samples to be of the max sequence length without padding
python ../common/convert_dataset.py --dataset c4 --data_subset en --out_root ./my-copy-c4 --splits train_small val --concat_tokens 2048 --tokenizer gpt2 --eos_text '<|endoftext|>'

# Download the 'train' split if you really want to train the model (not just profile)
# This will take 1-to-many hours depending on bandwidth, # CPUs, etc.
# The final folder `./my-copy-c4/train` will be ~800GB so make sure you have space!
# python ../common/convert_dataset.py --dataset c4 --data_subset en --out_root ./my-copy-c4 --splits train --concat_tokens 2048 --tokenizer gpt2 --eos_text '<|endoftext|>'

# For any of the above commands, you can also choose to compress the .mds files.
# This is useful if your plan is to store these in object store after conversion.
# python ../common/convert_dataset.py ... --compression zstd
```

### Test the Dataloader

To verify that the dataloader works, run a quick test on your `val` split like so:

```bash
# This will construct a `StreamingTextDataset` dataset from your `val` split,
# pass it into a PyTorch Dataloader, and iterate over it and print samples.
# Since we only provide a local path, no streaming/copying takes place.
python ../common/text_data.py --local_path ./my-copy-c4

# This will do the same thing, but stream data to {local} from {remote}.
# The remote path can be a filesystem or object store URI.
python ../common/text_data.py --local_path /tmp/cache-c4 --remote_path ./my-copy-c4  # stream from filesystem, e.g. a slow NFS volume to fast local disk
# python ../common/text_data.py --local_path /tmp/cache-c4 --remote_path s3://my-bucket/my-copy-c4  # stream from object store
```

# How to start training

Now that you've installed dependencies and built a local copy of the C4 dataset, let's start training!

**Please remember** to edit the `data_local` and (optionally) `data_remote` paths in your YAML.
Our streaming dataloader always streams to `data_local` <- from <- `data_remote`, and if the remote path is missing, the files are expected to be present in `data_local`.

**Also remember** that if you only downloaded the `train_small` split, you need to make sure your train_loader uses that split. Just change `split: train` to `split: train_small` in your YAML, [e.g. here](./yamls/mosaic_gpt/125m.yaml#L40). Or alternatively, pass it in via CLI arg: `composer main.py ... train_loader.dataset.split=train_small`.


### Single-Node training
We run the `main.py` script using our `composer` launcher, which generates N processes (1 per device).


If training on a single node, the `composer` launcher will autodetect the number of devices, so all you need to do is:

```bash
composer main.py yamls/mosaic_gpt/125m.yaml
```

To train with high performance on multi-node clusters, the easiest way is with the MosaicML platform ;) Check out the `mcloud/` folder for examples!

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
by using [Composer's logging integrations](https://docs.mosaicml.com/en/stable/trainer/logging.html).

```bash
[batch=1/100]:
         Train LanguageCrossEntropy: 10.9736
         Train Perplexity: 58312.0586
         Train loss/train/total: 10.9736
[batch=2/100]:
         Train LanguageCrossEntropy: 10.9724
         Train Perplexity: 58243.8086
         Train loss/train/total: 10.9724
[batch=3/100]:
         Train LanguageCrossEntropy: 10.9745
         Train Perplexity: 58365.8047
         Train loss/train/total: 10.9745
[batch=4/100]:
         Train LanguageCrossEntropy: 10.6459
         Train Perplexity: 42018.5508
         Train loss/train/total: 10.6459
```

# How many GPUs do I need to train a LLM?
This is a complicated question in general, but if we assume that you are using FSDP with `FULL_SHARD`,
activation checkpointing, and `DecoupledAdamW`, then a good rule of thumb is:

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

If you are running with a CUDA-compatible GPU and have installed the LLM requirements, we turn on by default a kernel fusion optimization for the Cross Entropy loss function at the end of the model. This should not affect your model convergence, but if you would like to disable this, you can set `model.loss_fn=torch_crossentropy`. To re-enable, set `model.loss_fn=fused_crossentropy` or omit it from your YAML.

On devices with more / less GPU memory, you may wish to edit the `device_train_microbatch_size` or `fsdp_config` values.
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

you want train LLMs on the MosaicML platform, reach out to us at [llm-early-access@mosaicml.com](mailto:llm-early-access@mosaicml.com)!
