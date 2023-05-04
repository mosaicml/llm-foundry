<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./assets/loss-curve-dark.png">
    <img alt="Compute-optimal training curves for LLMs of various sizes (125M -> 3B)." src="./assets/loss-curve-light.png" width="75%">
  </picture>
</p>

# LLM Foundry

This repo contains code for training, finetuning, evaluating, and deploying LLMs for inference with Composer and the MosaicML platform.

You'll find in this repo:
* `llmfoundry/` - source code for models, datasets, callbacks, utilities, etc.
* `scripts/` - scripts to run LLM workloads
  * `data_prep/` - convert text data from original sources to StreamingDataset format
  * `train/` - train or finetune HuggingFace and MPT models from 125M - 70B parameters
    * `train/benchmarking` - profile training throughput and MFU
  * `inference/` - convert models to HuggingFace or ONNX format, and generate responses
    * `inference/benchmarking` - profile inference latency and throughput
  * `eval/` - evaluate LLMs on academic (or custom) in-context-learning tasks
* `mcli/` - launch any of these workloads using [MCLI](https://docs.mosaicml.com/projects/mcli/en/latest/) and the [MosaicML platform](https://www.mosaicml.com/platform)


# Latest News
* [Blog: Benchmarking LLMs on H100](https://www.mosaicml.com/blog/coreweave-nvidia-h100-part-1)
* [Blog: Blazingly Fast LLM Evaluation](https://www.mosaicml.com/blog/llm-evaluation-for-icl)
* [Blog: GPT3 Quality for $500k](https://www.mosaicml.com/blog/gpt-3-quality-for-500k)
* [Blog: Billion parameter GPT training made easy](https://www.mosaicml.com/blog/billion-parameter-gpt-training-made-easy)



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
git clone https://github.com/mosaicml/llm-foundry.git
cd llm-foundry
pip install -e ".[gpu]"  # or pip install -e . if no NVIDIA GPU
```


# Quickstart

Here is a simple end-to-end workflow for preparing a subset of the C4 dataset, training an MPT-1B model for 10 batches, converting the model to HuggingFace format, and generating responses to prompts.

(Remember this is just a quickstart to demonstrate the tools -- To get good responses, the model must be trained for longer than 10 batches :)

<!--pytest.mark.skip-->
```bash
cd scripts

# Convert C4 dataset to StreamingDataset format
python data_prep/convert_dataset.py \
  --dataset c4 --data_subset en \
  --out_root my-copy-c4 --splits train_small val_small \
  --concat_tokens 2048 --tokenizer EleutherAI/gpt-neox-20b --eos_text '<|endoftext|>'

# Train an MPT-1B model for 10 batches
composer train/train.py \
  train/yamls/mpt/1b.yaml \
  data_local=my-copy-c4 \
  train_loader.dataset.split=train_small \
  eval_loader.dataset.split=val_small \
  max_duration=10ba \
  eval_subset_num_batches=1 \
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
