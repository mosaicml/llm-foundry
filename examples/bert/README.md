# Welcome!

This benchmark covers both pre-training and fine-tuning a BERT model. With this starter code, you'll be able to do Masked Language Modeling (MLM) [pre-training](#mlm-pre-training) on the C4 dataset and classification [fine-tuning](#glue-fine-tuning) on GLUE benchmark tasks. We also provide the source code and recipe behind our [Mosaic BERT](#mosaic-bert) model, which you can train yourself using this repo.

## Contents

You'll find in this folder:

### Pre-training

* `main.py` — A straightforward script for parsing YAMLs, building a [Composer](https://github.com/mosaicml/composer) Trainer, and kicking off an MLM pre-training job, locally or on Mosaic's cloud.
* `yamls/main/` - Pre-baked configs for pre-training both our sped-up Mosaic BERT as well as the reference HuggingFace BERT. These are used when running `main.py`.
* `yamls/test/main.yaml` - A config for quickly verifying that `main.py` runs.

### Fine-tuning
* `sequence_classification.py` - A starter script to simplify fine-tuning with your own dataset on a single classification task, locally or on Mosaic's cloud.
* `glue.py` - A more complex script for parsing YAMLs and orchestrating the numerous fine-tuning training jobs across 8 GLUE tasks (we exclude the WNLI task here), locally or on Mosaic's cloud.
* `src/glue/data.py` - Datasets used by `glue.py` in GLUE fine-tuning.
* `src/glue/finetuning_jobs.py` - Custom classes, one for each GLUE task, instantiated by `glue.py`. These handle individual fine-tuning jobs and task-specific hyperparameters.
* `yamls/finetuning/` - Pre-baked configs for fine-tuning both our sped-up Mosaic BERT as well as the reference HuggingFace BERT. These are used when running `sequence_classification.py` and `glue.py`.
* `yamls/test/sequence_classification.yaml` - A config for quickly verifying that `sequence_classification.py` runs.
* `yamls/test/glue.yaml` - A config for quickly verifying that `glue.py` runs.

### Shared
* `src/hf_bert.py` — HuggingFace BERT models for MLM (pre-training) or classification (GLUE fine-tuning), wrapped in [`ComposerModel`s](https://docs.mosaicml.com/en/stable/api_reference/generated/composer.models.HuggingFaceModel.html) for compatibility with the [Composer Trainer](https://docs.mosaicml.com/en/stable/api_reference/generated/composer.Trainer.html#composer.Trainer).
* `src/mosaic_bert.py` — Mosaic BERT models for MLM (pre-training) or classification (GLUE fine-tuning). See [Mosaic BERT](#mosaic-bert) for more.
* `src/bert_layers.py` — The Mosaic BERT layers/modules with our custom speed up methods built in, with an eye towards HuggingFace API compatibility.
* `src/bert_padding.py` — Utilities for Mosaic BERT that help avoid padding overhead.
* `src/flash_attn_triton.py` - Source code for the [FlashAttention](https://arxiv.org/abs/2205.14135) implementation used in Mosaic BERT.
* `requirements.txt` — All needed Python dependencies.
* This `README.md`

In the [common](../common) folder, you will also find:
* `../common/text_data.py`- a [MosaicML streaming dataset](https://streaming.docs.mosaicml.com/en/stable/) that can be used with a vanilla PyTorch dataloader.

## Quick start

### System recommendations

We recommend the following environment:
* A system with NVIDIA GPU(s)
* A Docker container running [MosaicML's PyTorch base image](https://hub.docker.com/r/mosaicml/pytorch/tags): `mosaicml/pytorch:1.13.1_cu117-python3.10-ubuntu20.04`.

This recommended Docker image comes pre-configured with the following dependencies:
  * PyTorch Version: 1.13.1
  * CUDA Version: 11.7
  * Python Version: 3.10
  * Ubuntu Version: 20.04

### Install

To get started, clone this repo and install the requirements:

```bash
git clone https://github.com/mosaicml/examples.git
cd examples
pip install -e ".[bert]"  # or pip install -e ".[bert-cpu]" if no NVIDIA GPU
cd examples/bert
```

## Prepare your data

*(If you have a small dataset that's stored locally or doesn't take much time to download from cloud storage, you can skip this section).*

In this benchmark, we train BERTs on the [C4: Colossal, Cleaned, Common Crawl dataset](https://huggingface.co/datasets/c4). To run pre-training on C4, you'll need to make yourself a copy of this dataset.

Alternatively, feel free to substitute our dataloader with one of your own in the script [main.py](./main.py#L98).
When you move on to fine-tuning, you can train on your own dataset by adding it into the script [sequence_classification.py](./sequence_classification.py#L63).
For now, let's focus on preparing the C4 data for pre-training.

We first convert the dataset from its native format (a collection of zipped JSONs)
to MosaicML's streaming dataset format (a collection of binary `.mds` files).
Once in `.mds` format, we can store the dataset in a central location (filesystem, S3, GCS, etc.)
and stream the data to any compute cluster, with any number of devices, and any number of CPU workers, and it all just works.
You can read more about the benefits of using mosaicml-streaming [here](https://streaming.docs.mosaicml.com/en/stable/).

### Converting C4 to streaming dataset `.mds` format

To make yourself a copy of C4, use `convert_dataset.py` like so:

```bash
# Download the 'train_small' and 'val' splits and convert to StreamingDataset format
# This will take 20-60 seconds depending on your Internet bandwidth
# You should see two folders: `./my-copy-c4/train_small` and `./my-copy-c4/val` that are each ~0.5GB
# Note: for BERT we are not doing any concatenation of samples, so we do not use the `--concat_tokens`
# option here. Instead, samples will simply get padded or truncated to the max sequence length
# in the dataloader
python ../common/convert_dataset.py --dataset c4 --data_subset en --out_root ./my-copy-c4 --splits train_small val

# Download the 'train' split if you really want to train the model (not just profile)
# This will take 1-to-many hours depending on bandwidth, # CPUs, etc.
# The final folder `./my-copy-c4/train` will be ~800GB so make sure you have space!
# python ../common/convert_dataset.py --dataset c4 --data_subset en --out_root ./my-copy-c4 --splits train

# For any of the above commands, you can also choose to compress the .mds files.
# This is useful if your plan is to store these in object store after conversion.
# python ../common/convert_dataset.py ... --compression zstd
```

If you're planning on doing multiple training runs, you can upload the **local** copy of C4 you just created to a central location. This will allow you to skip the dataset preparation step in the future. Once you have done so, modify the YAMLs in `yamls/main/` so that the `data_remote` field points to the new location. Then you can simply stream the dataset instead of creating a local copy!

### Test the Dataloader

To verify that the dataloader works, run a quick test on your `val` split like so:

```bash
# This will construct a `StreamingTextDataset` dataset from your `val` split,
# pass it into a PyTorch Dataloader, and iterate over it and print samples.
# Since we only provide a local path, no streaming/copying takes place.
python ../common/text_data.py --local_path ./my-copy-c4 --tokenizer bert-base-uncased

# This will do the same thing, but stream data to {local} from {remote}.
# The remote path can be a filesystem or object store URI.
python ../common/text_data.py --local_path /tmp/cache-c4 --remote_path ./my-copy-c4 --tokenizer bert-base-uncased # stream from filesystem, e.g. a slow NFS volume to fast local disk
# python ../common/text_data.py --local_path /tmp/cache-c4 --remote_path s3://my-bucket/my-copy-c4 --tokenizer bert-base-uncased # stream from object store
```

With our data prepared, we can now start training.



### Test pre-training

To verify that pre-training runs correctly, first prepare a local copy of the C4 validation split (see the above section), and then run the `main.py` pre-training script twice using our testing config.
First, with the baseline HuggingFace BERT. Second, with the Mosaic BERT.

```bash
# Run the pre-training script with the test config and HuggingFace BERT
composer main.py yamls/test/main.yaml

# Run the pre-training script with the test config and Mosaic BERT
composer main.py yamls/test/main.yaml model.name=mosaic_bert
```

### Test fine-tuning

To verify that fine-tuning runs correctly, run the fine-tuning script using our testing configs and both the HuggingFace and Mosaic BERT models.
First, verify `sequence_classification.py` with the baseline HuggingFace BERT and again with the Mosaic BERT.

```bash
# Run the fine-tuning script with the test config and HuggingFace BERT
composer sequence_classification.py yamls/test/sequence_classification.yaml

# Run the fine-tuning script with the test config and Mosaic BERT
composer sequence_classification.py yamls/test/sequence_classification.yaml model.name=mosaic_bert
```

Second, verify `glue.py` for both models.

```bash
# Run the GLUE script with the test config and HuggingFace BERT
python glue.py yamls/test/glue.yaml && rm -rf local-finetune-checkpoints

# Run the GLUE script with the test config and Mosaic BERT
python glue.py yamls/test/glue.yaml model.name=mosaic_bert && rm -rf local-finetune-checkpoints
```

## Training for real

Now that you've installed dependencies and built a local copy of the C4 dataset, let's start training! We'll start with MLM pre-training on C4.

**Note:** the YAMLs default to using `./my-copy-c4` as the location of your C4 dataset, following the above examples. Please remember to **edit the `data_remote` and `data_local` paths** in your pre-training YAMLs to reflect any differences in where your dataset lives (e.g., if you used a different folder name or already moved your copy to S3).

**Also remember** that if you only downloaded the `train_small` split, you need to make sure your train_dataloader is pointed at that split.
Just change `split: train` to `split: train_small` in your YAML.
This is already done in the testing YAML `yamls/test/main.py`, which you can also use to test your configuration (see [Test pre-training](#test-pre-training)).

### MLM pre-training

To get the most out of your pre-training budget, we recommend using **Mosaic BERT**! You can read more [below](#mosaic-bert).

We run the `main.py` pre-training script using our `composer` launcher, which generates N processes (1 process per GPU device).
If training on a single node, the `composer` launcher will autodetect the number of devices.

```bash
# This will pre-train a HuggingFace BERT that reaches a downstream GLUE accuracy of about 83.3%.
# It takes about 11.5 hours on a single node with 8 A100_80g GPUs.
composer main.py yamls/main/hf-bert-base-uncased.yaml

# This will pre-train a Mosaic BERT that reaches the same downstream accuracy in roughly 1/3 the time.
composer main.py yamls/main/mosaic-bert-base-uncased.yaml
```

**Please remember** to modify the reference YAMLs (e.g., `yamls/main/mosaic-bert-base-uncased.yaml`) to customize saving and loading locations. Only the YAMLs in `yamls/test/` are ready to use out-of-the-box. See the [configs](#configs) section for more detail.

### Single-task fine-tuning

After pre-training comes fine-tuning. We provide `sequence_classification.py` as a handy starter script to simplify fine-tuning a pre-trained BERT model on your own custom dataset. **Just modify this script by plugging in your dataset, and you can fine-tune your BERT model on the task you care about.** Check the script itself for more detailed instructions.

After modifying the starter script, update the reference YAMLs (e.g., `yamls/finetuning/mosaic-bert-base-uncased.yaml`) to reflect your changes. Use the `composer` launcher when you're ready.

```bash
# Fine-tune your BERT model on your custom classification task!
composer sequence_classification.py yamls/finetuning/mosaic-bert-base-uncased.yaml
```

### GLUE fine-tuning

The GLUE benchmark measures the average performance across 8 NLP classification tasks (again, here we exclude the WNLI task). This performance is typically used to evaluate the quality of the pre-training: once you have a set of weights from your MLM task, you fine-tune those weights separately for each task and then compute the average performance across the tasks, with higher averages indicating higher pre-training quality.

To handle this complicated fine-tuning pipeline, we provide the `glue.py` script.

This script handles parallelizing each of these fine-tuning jobs across all the GPUs on your machine.
That is, the `glue.py` script takes advantage of the small dataset/batch size of the GLUE tasks through *task* parallelism rather than data parallelism. This means that different tasks are trained in parallel, each using one GPU, instead of having one task trained at a time with batches parallelized across GPUs.

**Note:** To get started with the `glue.py` script, you will first need to update each reference YAML so that the starting checkpoint field points to the last checkpoint saved by `main.py`. See the [configs](#configs) section for more detail.

Once you have modified the YAMLs in `yamls/glue/` to reference your pre-trained checkpoint as the GLUE starting point, use non-default hyperparameters, etc., run the `glue.py` script using the standard `python` launcher (we don't use the `composer` launcher here because `glue.py` does its own multi-process orchestration):

```bash
# This will run GLUE fine-tuning evaluation on your HuggingFace BERT
python glue.py yamls/finetuning/glue/hf-bert-base-uncased.yaml

# This will run GLUE fine-tuning evaluation on your Mosaic BERT
python glue.py yamls/finetuning/glue/mosaic-bert-base-uncased.yaml
```

Aggregate GLUE scores will be printed out at the end of the script and can also be tracked using Weights and Biases, if enabled via the YAML.
Any of the other [composer supported loggers](https://docs.mosaicml.com/en/stable/trainer/logging.html#available-loggers) can be added easily as well!

Fair warning: all the processes launched inside of `glue.py` will generate their own printouts during training. So don't be surprised if your console looks a bit chaotic. That means it's working :)

## Configs

This section is to help orient you to the config YAMLs referenced throughout this README and found in `yamls/`.

A quick note on our use of YAMLs:
* We use YAMLs just to make all the configuration explicit.
* You can also configure anything you want directly in the Python files.
* The YAML files don't use any special schema, keywords, or other magic. They're just a clean way of writing a dict for the scripts to read.

In other words, you're free to modify this starter code to suit your project and aren't tied to using YAMLs in your workflow.

### main.py

Before using the configs in `yamls/main/` when running `main.py`, you'll need to fill in:

* `save_folder` - This will determine where model checkpoints are saved. Note that it can depend on `run_name`. For example, if you set `save_folder` to `s3://mybucket/mydir/{run_name}/ckpt` it will replace `{run_name}` with the value of `run_name`. So you should avoid re-using the same run name across multiple training runs.
* `data_remote` - Set this to the filepath of your streaming C4 directory. The default value of `./my-copy-c4` will work if you built a local C4 copy, following the [dataset preparation](#prepare-your-data) instructions. If you moved your dataset to a central location, you simply need to point `data_remote` to that new location.
* `data_local` - This is the path to the local directory where the dataset is streamed to. If `data_remote` is local, you can use the same path for `data_local` so no additional copying is done. The default values of `./my-copy-c4` are set up to work with such a local copy. If you moved your dataset to a central location, setting `data_local` to `/tmp/cache-c4` should work fine.
* `loggers.wandb` (optional) - If you want to log to W&B, fill in the `project` and `entity` fields, or comment out the `wandb` block if you don't want to use this logger.
* `load_path` (optional) - If you have a checkpoint that you'd like to start from, this is how you set that.

### sequence_classification.py

Before using the configs in `yamls/finetuning/` when running `sequence_classification.py`, you'll need to fill in:

* `load_path` (optional) - If you have a checkpoint that you'd like to start from, this is how you set that. If you're fine-tuning a Mosaic BERT, this should not be left empty.
* `save_folder` - This will determine where model checkpoints are saved. Note that it can depend on `run_name`. For example, if you set `save_folder` to `s3://mybucket/mydir/{run_name}/ckpt` it will replace `{run_name}` with the value of `run_name`. So you should avoid re-using the same run name across multiple training runs.
* `loggers.wandb` (optional) - If you want to log to W&B, fill in the `project` and `entity` fields, or comment out the `wandb` block if you don't want to use this logger.
* `algorithms` (optional) - Make sure to include any architecture-modifying algorithms that were applied to your starting checkpoint model before pre-training. For instance, if you turned on `gated_linear_units` in pre-training, make sure to do so during fine-tuning too!

### glue.py

Before using the configs in `yamls/finetuning/glue/` when running `glue.py`, you'll need to fill in:

* `starting_checkpoint_load_path` - This determines which checkpoint you start from when doing fine-tuning. This should look like `<save_folder>/<checkpoint>`, where `<save_folder>` is the location you set in your pre-training config (see above section).
* `loggers.wandb` (optional) - If you want to log to W&B, fill in the `project` and `entity` fields, or comment out the `wandb` block if you don't want to use this logger.
* `base_run_name` (optional) - Make sure to avoid re-using the same name across multiple runs.
* `algorithms` (optional) - Make sure to include any architecture-modifying algorithms that were applied to your starting checkpoint model before pre-training. For instance, if you turned on `gated_linear_units` in pre-training, make sure to do so during fine-tuning too!

## Running on the MosaicML Cloud

If you have configured a compute cluster to work with the MosaicML Cloud, you can use the `yaml/*/mcloud_run*.yaml` reference YAMLs for examples of how to run pre-training and fine-tuning remotely!

Once you have filled in the missing YAML fields (and made any other modifications you want), you can launch pre-training by simply running:

```bash
mcli run -f yamls/main/mcloud_run_a100_80gb.yaml
```

Or, if your cluster has A100 GPUs with 40gb of memory:

```bash
mcli run -f yamls/main/mcloud_run_a100_40gb.yaml
```

Similarly, for sequence classification fine-tuning, just fill in the missing YAML fields (e.g., to use the pre-training checkpoint as the starting point) and run:

```bash
mcli run -f yamls/finetuning/mcloud_run.yaml
```

The same applies for GLUE fine-tuning. Fill in the missing YAML fields and run:

```bash
mcli run -f yamls/finetuning/glue/mcloud_run.yaml
```

### Multi-node training

To train with high performance on *multi-node* clusters, the easiest way is with MosaicML Cloud ;)

But if you want to try this manually on your own cluster, then just provide a few variables to `composer`, either directly via CLI or via environment variables. Then launch the appropriate command on each node.

**Note:** multi-node training will only work with `main.py`; the `glue.py` script handles its own orchestration across devices and is not built to be used with the `composer` launcher.

#### Multi-Node via CLI args

```bash
# Using 2 nodes with 8 devices each
# Total world size is 16
# IP Address for Node 0 = [0.0.0.0]

# Node 0
composer --world_size 16 --node_rank 0 --master_addr 0.0.0.0 --master_port 7501 main.py yamls/main/mosaic-bert-base-uncased.yaml

# Node 1
composer --world_size 16 --node_rank 1 --master_addr 0.0.0.0 --master_port 7501 main.py yamls/main/mosaic-bert-base-uncased.yaml

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
composer main.py yamls/main/mosaic-bert-base-uncased.yaml

# Node 1
# export WORLD_SIZE=16
# export NODE_RANK=1
# export MASTER_ADDR=0.0.0.0
# export MASTER_PORT=7501
composer main.py yamls/main/mosaic-bert-base-uncased.yaml
```

You should see logs being printed to your terminal.
You can also easily enable other experiment trackers like Weights and Biases or CometML by using [Composer's logging integrations](https://docs.mosaicml.com/en/stable/trainer/logging.html).


## Mosaic BERT

Our starter code supports both standard HuggingFace BERT models and our own **Mosaic BERT**. The latter incorporates numerous methods to improve throughput and training.
Our goal in developing Mosaic BERT was to greatly reduce training time while making it easy for you to use on your own problems!

To do this, we employ a number of techniques from the literature:
* [ALiBi (Press et al., 2021)](https://arxiv.org/abs/2108.12409v1)
* [Gated Linear Units (Shazeer, 2020)](https://arxiv.org/abs/2002.05202)
* ["The Unpadding Trick"](https://github.com/mlcommons/training_results_v1.1/blob/main/NVIDIA/benchmarks/bert/implementations/pytorch/fmha.py)
* [FusedLayerNorm (NVIDIA)](https://nvidia.github.io/apex/layernorm.html)
* [FlashAttention (Dao et al., 2022)](https://arxiv.org/abs/2205.14135)

... and get them to work together! To our knowledge, many of these methods have never been combined before.

If you're reading this, we're still profiling the exact speedup and performance gains offered by Mosaic BERT compared to comparable HuggingFace BERT models. Stay tuned for incoming results!


## Contact Us

If you run into any problems with the code, please file Github issues directly to this repo.

If you want to train BERT-style models on MosaicML Cloud, reach out to us at [demo@mosaicml.com](mailto:demo@mosaicml.com)!
