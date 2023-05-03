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

At all model scales, we are training the exact same [vanilla PyTorch GPT model](./llmfoundry/models/mosaic_gpt/mosaic_gpt.py), with no special parallelism strategies.
Composer + FSDP does all the heavy lifting to make sure we can scale up without running out of memory and while maintaining high performance.

Feel free to edit any or all of these files, and get a feel for using the LLM stack!
In `llmfoundry/models/mosaic_gpt/mosaic_gpt.py` you can see how easy it is to modify the architecture and replace a layer like `torch.nn.MultiheadAttention` with
a new one like `FlashMHA`. If you want to try and change the FSDP wrapping strategy (e.g. wrap all `GPTMLP` layers in addition to `GPTBlock`),
go ahead and edit it directly in `mosaic_gpt.py`! You'll find a full guide on how to build custom models for Composer + FSDP under [llmfoundry/README.md](./llmfoundry/README.md).

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
git clone https://github.com/mosaicml/llmfoundry.git
cd llmfoundry
pip install -e ".[gpu]"  # or pip install -e . if no NVIDIA GPU
```


# Quickstart

Here is a simple end-to-end workflow for preparing a subset of the C4 dataset, training an MPT-1B model for 10 batches, converting the model to HuggingFace format, and generating responses to prompts.

(Note this is just a quickstart to get a feel for the tools. To get good responses, the model must be trained for longer than 10 batches :)

```bash
# Convert C4 dataset to StreamingDataset format
python data_prep/convert_dataset.py \
  --dataset c4 --data_subset en \
  --out_root my-copy-c4 --splits train_small val \
  --concat_tokens 2048 --tokenizer gpt2 --eos_text '<|endoftext|>'

# Train an MPT-1B model for 10 batches
composer train/train.py \
  train/yamls/mosaic_gpt/1b.yaml \
  data_local=my-copy-c4 \
  train_loader.dataset.split=train_small \
  max_duration=10ba \
  eval_interval=0 \
  save_folder=mpt-1b

# Convert the model to HuggingFace format
python inference/convert_composer_to_hf.py \
  --composer_path mpt-1b/ep0-ba10-rank0.pt \
  --hf_output_path mpt-1b/hf \
  --output_precision bf16

# Generate responses to prompts
python inference/hf_generate.py \
  --name_or_path mpt-1b/hf \
  --max_new_tokens 256 \
  --prompts \
    "The answer to life, the universe, and happiness is" \
    "Here's a quick recipe for baking chocolate chip cookies: Start by"
```

# Contact Us
If you run into any problems with the code, please file Github issues directly to this repo.

If you want to train LLMs on the MosaicML platform, reach out to us at [llm-early-access@mosaicml.com](mailto:llm-early-access@mosaicml.com)!
