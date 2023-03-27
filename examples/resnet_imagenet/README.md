<br />
<p align="center">
   <img src="https://assets.website-files.com/61fd4eb76a8d78bc0676b47d/62a185326fcd73061ab9aaf9_Hero%20Image%20Final.svg" width="50%" height="50%"/>
</p>

<h2><p align="center">The most efficient recipes for training ResNets on ImageNet</p></h2>

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

# Mosaic ResNet

This folder contains starter code for training [torchvision ResNet architectures](https://pytorch.org/vision/stable/models.html) using our most efficient training recipes (see our [short blog post](https://www.mosaicml.com/blog/mosaic-resnet) or [detailed blog post](https://www.mosaicml.com/blog/mosaic-resnet-deep-dive) for details). These recipes were developed to hit baseline accuracy on [ImageNet](https://www.image-net.org/) 7x faster or to maximize ImageNet accuracy over long training durations. Although these recipes were developed for training ResNet on ImageNet, they could be used to train other image classification models on other datasets. Give it a try!

The specific files in this folder are:

- `model.py` - Creates a [ComposerModel](https://docs.mosaicml.com/en/stable/composer_model.html) from a torchvision ResNet model
- `data.py` - Provides a [MosaicML streaming dataset](https://streaming.docs.mosaicml.com/en/stable/) for ImageNet and a PyTorch dataset for a local copy of ImageNet
- `main.py` - Trains a ResNet on ImagNet using the [Composer](https://github.com/mosaicml/composer) [Trainer](https://docs.mosaicml.com/en/stable/api_reference/generated/composer.Trainer.html#trainer).
- `tests/` - A suite of tests to check each training component
- `yamls/`
  - `resnet50.yaml` - Configuration for a ResNet50 training run to be used as the first argument to `main.py`
  - `mcloud_run.yaml` - yaml to use if running on the [MosaicML platform](https://www.mosaicml.com/blog/introducing-mosaicml-cloud)

Now that you've explored the code, let's get training.

## Prepare your data

If you want to train on ImageNet or any other dataset, you'll need to either make it a [streaming dataset](https://github.com/mosaicml/streaming) using [this script](https://github.com/mosaicml/streaming/blob/86a9b95189e8b292a8c7880a1c49dc55d1895544/streaming/vision/convert/imagenet.py) or a local [torchvision ImageFolder](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html). If you are planning to train on ImageNet, download instructions can be found [here](https://www.image-net.org/download.php).

The below command will test if your data is set up appropriately:

```bash
# Test dataset stored locally
python data.py path/to/data

# Test datast stored remotely
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

You'll also need to tell the default training configuration file [(resnet50.yaml)](./yamls/resnet50.yaml) where your dataset lives:

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

Here's what you need to train:

- Docker image with PyTorch 1.12+, e.g. [MosaicML's PyTorch image](https://hub.docker.com/r/mosaicml/pytorch/tags)
  - Recommended tag: `mosaicml/pytorch:1.12.1_cu116-python3.9-ubuntu20.04`
  - The image comes pre-configured with the following dependencies:
    - PyTorch Version: 1.12.1
    - CUDA Version: 11.6
    - Python Version: 3.9
    - Ubuntu Version: 20.04
- [ImageNet Dataset](http://www.image-net.org/)
  - Must be stored either locally (see download instructions [here](https://www.image-net.org/download.php)) or uploaded to an S3 bucket after converting to a [streaming format](https://github.com/mosaicml/streaming) using [this script](https://github.com/mosaicml/streaming/blob/86a9b95189e8b292a8c7880a1c49dc55d1895544/streaming/vision/convert/imagenet.py)
- System with NVIDIA GPUs

### Installation

Just clone this repo and install the requirements. If you want to customize the
code, first fork this repo on GitHub and clone your fork instead.

```bash
git clone https://github.com/mosaicml/examples.git
cd examples
pip install -e ".[resnet-imagenet]"  # or pip install -e ".[resnet-imagenet-cpu]" if no NVIDIA GPU
cd examples/resnet_imagenet
```

### How to start training

Now that you've installed dependencies and tested your dataset, let's start training!

**Please remember**: for both `train_dataset` and `eval_dataset`, edit the `path` and (if streaming) `local` arguments in `resnet50.yaml` to point to your data.

#### Single-Node training

We run the `main.py` script using our `composer` launcher, which generates a process for each device in a node.

If training on a single node, the `composer` launcher will autodetect the number of devices, so all you need to do is :

```bash
composer main.py yamls/resnet50.yaml
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
composer --world_size 16 --node_rank 0 --master_addr 0.0.0.0 --master_port 7501 main.py yamls/resnet50.yaml

# Node 1
composer --world_size 16 --node_rank 1 --master_addr 0.0.0.0 --master_port 7501 main.py yamls/resnet50.yaml
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
composer main.py yamls/resnet50.yaml

# Node 1
# export WORLD_SIZE=16
# export NODE_RANK=1
# export MASTER_ADDR=0.0.0.0
# export MASTER_PORT=7501
composer main.py yamls/resnet50.yaml
```

#### Results

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

## Using Mosaic recipes

As described in our [ResNet blog post](https://www.mosaicml.com/blog/mosaic-resnet), we cooked up three recipes to train ResNets faster and with higher accuracy:

- The **Mild** recipe is for short training runs
- The **Medium** recipe is for longer training runs
- The **Hot** recipe is for the longest training runs, intended to maximize accuracy

<img src="https://assets.website-files.com/61fd4eb76a8d78bc0676b47d/62a188a808b39301a7c3550f_Recipe%20Final.svg" width="50%" height="50%"/>

To use a recipe, specify the name using the the `recipe_name` argument, either in the config file or via the command line. Specifying a recipe will change several aspects of the training run such as:

1. Changes the loss function to binary cross entropy instead of standard cross entropy to improve accuracy.
2. Changes the train crop size to 176 (instead of 224) and the evaluation resize size to 232 (instead of 256). The smaller train crop size increases throughput and accuracy.
3. Changes the number of training epochs to the optimal value for each training recipe. Feel free to change these in `resnet50.yaml` to better suite your model and/or dataset.
4. Specifies unique sets of [speedup methods](https://docs.mosaicml.com/en/stable/trainer/algorithms.html) for model training.

Here is an example command to run the mild recipe locally:

```bash
composer main.py yamls/resnet50.yaml recipe_name=mild
```

---

**NOTE**

The ResNet50 and smaller models are dataloader-bottlenecked when training with our recipes on 8x NVIDIA A100s. This means the model's throughput is limited to the speed the data can be loaded. One way to alleviate this bottleneck is to use the [FFCV dataloader format](https://github.com/libffcv/ffcv). This [tutorial](https://docs.mosaicml.com/en/stable/examples/ffcv_dataloaders.html) walks you through creating your FFCV dataloader. We also have example code for an ImageNet FFCV dataloader [here](https://github.com/mosaicml/composer/blob/a0f441537008a1ef2678f1474f3cd5519deb80fa/composer/datasets/imagenet.py#L179).

Our best results use FFCV, so an FFCV version of ImageNet is required to exactly match our best results.

---

## Saving and loading checkpoints

At the bottom of [`yamls/resnet50.yaml`](./yamls/resnet50.yaml), we provide arguments for saving and loading model weights. Please specify the `save_folder` or `load_path` arguments if you need to save or load checkpoints!

## Contact us

If you run into any problems with the code, please file Github issues directly to this repo.
