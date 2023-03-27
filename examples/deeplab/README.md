<br />
<p align="center">
   <img src="https://assets-global.website-files.com/61fd4eb76a8d78bc0676b47d/6375dfb52e6aae05f4ceacc2_Screen%20Shot%202022-11-17%20at%202.13.48%20AM.png" width="50%" height="50%"/>
</p>

<h2><p align="center">The most efficient recipes for training DeepLabV3+ on ADE20K</p></h2>

<h3><p align='center'>
<a href="https://www.mosaicml.com">[Website]</a>
- <a href="https://docs.mosaicml.com/">[Composer Docs]</a>
- <a href="https://docs.mosaicml.com/en/stable/method_cards/methods_overview.html">[Methods]</a>
- <a href="https://www.mosaicml.com/team">[We're Hiring!]</a>
</p></h3>

<p align="center">
    <a href="https://join.slack.com/t/mosaicml-community/shared_invite/zt-w0tiddn9-WGTlRpfjcO9J5jyrMub1dg">
        <img alt="Chat @ Slack" src="https://img.shields.io/badge/slack-chat-2eb67d.svg?logo=slack">
    </a>
    <a href="https://github.com/mosaicml/examples/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-green.svg?logo=slack">
    </a>
</p>
<br />

# Mosaic DeepLabV3+

This folder contains starter code for training [mmsegmentation DeepLabV3+ architectures](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/deeplabv3plus) using our most efficient training recipes (see our [benchmark blog post](https://www.mosaicml.com/blog/behind-the-scenes) or [recipes blog post](https://www.mosaicml.com/blog/mosaic-image-segmentation) for details). These recipes were developed to hit baseline accuracy on [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/) 5x faster or to maximize ADE20K mean Intersection-over-Union (mIoU) over long training durations. Although these recipes were developed for training DeepLabV3+ on ADE20k, they could be used to train other segmentation models on other datasets. Give it a try!

The files in this folder are:

- `model.py` - A [ComposerModel](https://docs.mosaicml.com/en/stable/composer_model.html) that wraps an mmsegmentation DeepLabV3+ model
- `data.py` - A [MosaicML streaming dataset](https://streaming.docs.mosaicml.com/en/stable/) for ADE20K and a PyTorch dataset for a local copy of ADE20K
- `transforms.py` - Torchvision transforms for ADE20K
- `download_ade20k.py` - A helper script for downloading ADE20K locally
- `main.py` - The training script that builds a Composer [Trainer](https://docs.mosaicml.com/en/stable/api_reference/generated/composer.Trainer.html#trainer) using the data and model
- `tests/` - A suite of tests to check each training component
- `yamls/`
  - `deeplabv3.yaml` - Configuration for a DeepLabV3+ training run to be used as the first argument to `main.py`
  - `mcloud_run.yaml` - yaml to use if running on the [MosaicML platform](https://www.mosaicml.com/blog/introducing-mosaicml-cloud)

Now that you have explored the code, let's jump into the prerequisites for training.

## Prepare your data

This benchmark assumes that [ADE20k Dataset](https://groups.csail.mit.edu/vision/datasets/ADE20K/) is already stored on your local machine or stored in an S3 bucket after being processed into a streaming dataset. ADE20K can be downloaded by running the command below. This takes up about 1GB of storage and will default to storing the dataset in `./ade20k`.

```bash
# download ADE20k to specified local directory
python download_ade20k.py
```

To convert ADE20k to a [streaming format](https://github.com/mosaicml/streaming) for efficient training from an object store like S3, use [this script](https://github.com/mosaicml/streaming/blob/main/streaming/vision/convert/ade20k.py).

The below commands will test if your data is set up appropriately:

```bash
# Test locally stored dataset
python data.py path/to/data

# Test remote storage dataset
python data.py s3://my-bucket/my-dir/data /tmp/path/to/local
```

## Get started with the MosaicML platform

If you're using the MosaicML platform, all you need to install is [`mcli`](https://mcli.docs.mosaicml.com/):

```bash
pip install --upgrade mosaicml-cli
mcli init
```

Then, just fill in a few fields in [yamls/mcloud_run.yaml](./yamls/mcloud_run.yaml):

```yaml
cluster: # Add the name of the cluster to use for this run
gpu_type: # Type of GPU to use; usually a100_40gb
integrations:
  - integration_type: git_repo
    git_repo: mosaicml/examples # Replace with your fork to use custom code
    git_branch: main # Replace with your branch to use custom code
    ssh_clone: false # Should be true if using a private repo
```

These tell `mcli` where to get your code and what cluster your organization is using.
If you are using a private github repository, you'll need to set up [github secrets](https://mcli.docs.mosaicml.com/en/latest/secrets/ssh.html#git-ssh-secrets)

You'll also need to tell the default training configuration file [(resnet50.yaml)](./yamls/deeplabv3.yaml) where your dataset lives:

```yaml
train_dataset:
    ...
    path: # Fill in with path to local data directory or cloud bucket

eval_dataset:
    ...
    path:  # Fill in with path to local data directory or cloud bucket
```

With this information provided, you can now run the code in this directory on a remote machine like so:

```bash
mcli run -f yamls/mcloud_run.yaml
```

You're done. You can skip the rest of the instructions except [using Mosaic recipes](#using-mosaic-recipes).

## Get started without the MosaicML platform

### Prerequisites

Here's what you need to get started:

- Docker image with PyTorch 1.12+, e.g. [MosaicML's PyTorch image](https://hub.docker.com/r/mosaicml/pytorch/tags)
  - Recommended tag: `mosaicml/pytorch_vision:1.12.1_cu116-python3.9-ubuntu20.04`
  - The image comes pre-configured with the following dependencies:
    - PyTorch Version: 1.12.1
    - CUDA Version: 11.6
    - MMCV Version: 1.4.8
    - mmsegmentation Version: 0.22.0
    - Python Version: 3.9
    - Ubuntu Version: 20.04
- [ADE20k Dataset](https://groups.csail.mit.edu/vision/datasets/ADE20K/) must be stored either locally (see `download_ade20k.py`) or uploaded to an S3 bucket after converting to a [streaming format](https://github.com/mosaicml/streaming) using [this script](https://github.com/mosaicml/streaming/blob/main/streaming/vision/convert/ade20k.py)
- System with NVIDIA GPUs

### Installation

To get started, clone this repo and install the requirements:

```bash
git clone https://github.com/mosaicml/examples.git
cd examples
pip install -e ".[deeplab]"  # or pip install -e ".[deeplab-cpu]" if no NVIDIA GPU
cd examples/deeplab
# Note: mmcv-full is not in requirements.txt since it is difficult to install automatically
# If you are not using the suggested docker image, install mmcv using the instructions at https://mmcv.readthedocs.io/en/latest/get_started/installation.html#install-with-pip (we use version 1.4.8)
```

---

### How to start training

Now that you've installed dependencies and tested your dataset, let's start training!

**Please remember**: for both `train_dataset` and `eval_dataset`, edit the `path` and (if streaming) `local` arguments in `deeplabv3.yaml` to point to your data.

#### Single-Node training

We run the `main.py` script using our `composer` launcher, which generates a process for each device in a node.

If training on a single node, the `composer` launcher will autodetect the number of devices, so all you need to do is:

```bash
composer main.py yamls/deeplabv3.yaml
```

To train with high performance on multi-node clusters, the easiest way is with the MosaicML platform ;)

But if you really must try this manually on your own cluster, then just provide a few variables to `composer`
either directly via CLI, or via environment variables that can be read. Then launch the appropriate command on each node:

#### Multi-Node via CLI args

```bash
# Using 2 nodes with 8 devices each
# Total world size is 16
# IP Address for Node 0 = [0.0.0.0]

# Node 0
composer --world_size 16 --node_rank 0 --master_addr 0.0.0.0 --master_port 7501 main.py yamls/deeplabv3.yaml

# Node 1
composer --world_size 16 --node_rank 1 --master_addr 0.0.0.0 --master_port 7501 main.py yamls/deeplabv3.yaml
```

#### Multi-Node via environment variables

```bash
# Using 2 nodes with 8 devices each
# Total world size is 16
# IP Address for Node 0 = [0.0.0.0]

# Node 0
# export WORLD_SIZE=16
# export NODE_RANK=0
# export MASTER_ADDR=0.0.0.0
# export MASTER_PORT=7501
composer main.py yamls/deeplabv3.yaml

# Node 1
# export WORLD_SIZE=16
# export NODE_RANK=1
# export MASTER_ADDR=0.0.0.0
# export MASTER_PORT=7501
composer main.py yamls/deeplabv3.yaml
```

### Results

You should see logs being printed to your terminal like below. You can also easily enable other experiment trackers like Weights and Biases or CometML,
by using [Composer's logging integrations](https://docs.mosaicml.com/en/stable/trainer/logging.html).

```bash
[epoch=0][batch=16/625]: wall_clock/train: 17.1607
[epoch=0][batch=16/625]: wall_clock/val: 10.9666
[epoch=0][batch=16/625]: wall_clock/total: 28.1273
[epoch=0][batch=16/625]: lr-DecoupledSGDW/group0: 0.0061
[epoch=0][batch=16/625]: trainer/global_step: 16
[epoch=0][batch=16/625]: trainer/batch_idx: 16
[epoch=0][batch=16/625]: memory/alloc_requests: 38424
[epoch=0][batch=16/625]: memory/free_requests: 37690
[epoch=0][batch=16/625]: memory/allocated_mem: 6059054353408
[epoch=0][batch=16/625]: memory/active_mem: 1030876672
[epoch=0][batch=16/625]: memory/inactive_mem: 663622144
[epoch=0][batch=16/625]: memory/reserved_mem: 28137488384
[epoch=0][batch=16/625]: memory/alloc_retries: 3
[epoch=0][batch=16/625]: trainer/grad_accum: 2
[epoch=0][batch=16/625]: loss/train/total: 7.1292
[epoch=0][batch=16/625]: metrics/train/MulticlassAccuracy: 0.0005
[epoch=0][batch=17/625]: wall_clock/train: 17.8836
[epoch=0][batch=17/625]: wall_clock/val: 10.9666
[epoch=0][batch=17/625]: wall_clock/total: 28.8502
[epoch=0][batch=17/625]: lr-DecoupledSGDW/group0: 0.0066
[epoch=0][batch=17/625]: trainer/global_step: 17
[epoch=0][batch=17/625]: trainer/batch_idx: 17
[epoch=0][batch=17/625]: memory/alloc_requests: 40239
[epoch=0][batch=17/625]: memory/free_requests: 39497
[epoch=0][batch=17/625]: memory/allocated_mem: 6278452575744
[epoch=0][batch=17/625]: memory/active_mem: 1030880768
[epoch=0][batch=17/625]: memory/inactive_mem: 663618048
[epoch=0][batch=17/625]: memory/reserved_mem: 28137488384
[epoch=0][batch=17/625]: memory/alloc_retries: 3
[epoch=0][batch=17/625]: trainer/grad_accum: 2
[epoch=0][batch=17/625]: loss/train/total: 7.1243
[epoch=0][batch=17/625]: metrics/train/MulticlassAccuracy: 0.0010
train          Epoch   0:    3%|▋                        | 17/625 [00:17<07:23,  1.37ba/s, loss/train/total=7.1292]
```

## Using Mosaic Recipes

As described in our [Segmentation blog post](https://www.mosaicml.com/blog/mosaic-image-segmentation), we cooked up three recipes to train DeepLabV3+ faster and with higher accuracy:

- The **Mild** recipe is for short training runs
- The **Medium** recipe is for longer training runs
- The **Hot** recipe is for the longest training runs, intended to maximize accuracy

<img src="https://assets-global.website-files.com/61fd4eb76a8d78bc0676b47d/6375c40a1de1101f791bc2d7_Recipe%20Final%20(18).png" width="50%" height="50%"/>

To use a recipe, specify the name using the the `recipe_name` argument. Specifying a recipe will change the duration of the training run to the optimal value for that recipe. Feel free to change these in `deeplabv3.yaml` to better suite your model and/or dataset.

Here is an example command to run the mild recipe on a single node:

```bash
composer main.py yamls/basline.yaml recipe_name=mild
```

---

## Saving and Loading checkpoints

At the bottom of `yamls/deeplabv3.yaml`, we provide arguments for saving and loading model weights. Please specify the `save_folder` or `load_path` arguments if you need to save or load checkpoints!

## On memory constraints

In previous blog posts ([1](https://www.mosaicml.com/blog/farewell-oom), [2](https://www.mosaicml.com/blog/billion-parameter-gpt-training-made-easy))
we demonstrated Auto Grad Accum. This allows Composer to automatically execute each batch as multiple microbatches to save memory. This means the same configuration can be run on different hardware or on fewer devices without manually tuning the batch size or (significantly) changing the optimization. This feature is thoroughly tested, but if there are any issues, you can manually set `grad_accum` to your desired value.

## Contact Us

If you run into any problems with the code, please file Github issues directly to this repo.
