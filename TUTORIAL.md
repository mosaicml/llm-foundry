# LLM Foundry Tutorial

- [Intro](#intro)
  - [How this repo is structured](#how-this-repo-is-structured)
  - [Key components](#key-components)
  - [How the YAMLs work](#how-the-yamls-work)
- [Example Workflows](#example-workflows)
  - [Workflow 1: I want to play with a HF model like MPT-7B locally](#workflow-1-i-want-to-play-with-a-hf-model-like-mpt-7b-locally)
  - [Workflow 2: I want to deploy an inference endpoint with a HF model like MPT-7B](#workflow-2-i-want-to-deploy-an-inference-endpoint-with-a-hf-model-like-mpt-7b)
  - [Workflow 3: I want to fine-tune a HF model like MPT-7B](#workflow-3-i-want-to-fine-tune-a-hf-model-like-mpt-7b)
  - [Workflow 4: I want to train a new HF model from scratch](#workflow-4-i-want-to-train-a-new-hf-model-from-scratch)
- [FAQs](#faqs)
  - [Common installation issues](#common-installation-issues)
  - [Why is the script only using 1 out of N GPUs?](#why-is-the-script-only-using-1-out-of-n-gpus)
  - [I’m running into an Out-Of-Memory (OOM) error. What do I do?](#im-running-into-an-out-of-memory-oom-error-what-do-i-do)
  - [What hardware can I train on?](#what-hardware-can-i-train-on)
  - [What hardware can I run eval on?](#what-hardware-can-i-run-eval-on)
  - [What hardware can I run inference on?](#what-hardware-can-i-run-inference-on)
  - [What is the MosaicML Training platform?](#what-is-the-mosaicml-training-platform)
  - [What is the MosaicML Inference platform?](#what-is-the-mosaicml-inference-platform)
  - [What is FSDP?](#what-is-fsdp)
  - [What are the different attention options `torch` / `flash` / `triton`  for MPT and which one should I use?](#what-are-the-different-attention-options-torch--flash--triton-for-mpt-and-which-one-should-i-use)
  - [Can I finetune using PEFT / LORA?](#can-i-finetune-using-peft--lora)
  - [Can I quantize these models and/or run on CPU?](#can-i-quantize-these-models-andor-run-on-cpu)
  - [How do I deploy with ONNX/FasterTransformer?](#how-do-i-deploy-with-onnxfastertransformer)
  - [How expensive is it to build LLMs?](#how-expensive-is-it-to-build-llms)



# Intro

## How this repo is structured

`llm-foundry` is divided broadly into source code and the scripts that use the source code. Here is an overview:

- `llmfoundry/` - The pip installable source code
  - `data/` - Dataloaders used throughout training and evaluation
  - `models/` - Model source code and classes for training, evaluation, and inference
  - `callbacks/` - A collection of useful Composer callbacks
  - `optim/` - Custom optimizers, e.g., Decoupled LionW
- `scripts/` - Scripts for each stage of the model forging lifecycle
  - `data_prep/` - Convert local and HuggingFace datasets to our Streaming MDS format with support for remote uploading
  - `train/` - Contains the main training script for pretraining/finetuning your model, as well as example workloads
  - `eval/` - Contains a dedicated script for evaluating your trained model, as well as example workloads
  - `inference/` - Scripts to load and query trained, HuggingFace-formatted models
- `mcli/` - A collection of example workload configurations that could be launched on the MosaicML Platform via the `mcli` command line interface.

A few notes:

- The various directories under `scripts` contain their own README files.
- We use YAML files heavily. For example, the main training script `scripts/train/train.py` takes in a configuration YAML and will interpret that YAML to determine how to build the dataloader, model, optimizer, etc. used for training.
- **We are actively building documentation to help explain these YAMLs** but the best way to understand them is to walk through the script itself to follow how the config YAML is interpreted.

## Key components

There are 3 key libraries (all from MosaicML) that power `llm-foundry` and which you'll see throughout. These are worth covering a bit more, so in this section we'll briefly go over [Composer](https://docs.mosaicml.com/projects/composer/en/latest/), our distributed training engine, [Streaming](https://docs.mosaicml.com/projects/streaming/en/stable/), which enables streaming datasets, and [MCLI](https://docs.mosaicml.com/projects/mcli/en/latest/), which you can use to train on the MosaicML Platform.

### Composer

The Composer library is the workhorse of our training and evaluation scripts.
If you dig into those scripts, you'll notice that they are basically just very configurable wrappers around the Composer [Trainer](https://docs.mosaicml.com/projects/composer/en/latest/trainer/using_the_trainer.html).
The Trainer is a pytorch-native object that composes your model, dataset(s), optimizer, and more into a cohesive training pipeline with all the bells and whistles.
Spending some time understanding the Composer Trainer is a great way to form a deeper understanding of what the train and eval scripts are doing under the hood.

Composer also comes packaged with the `composer` launcher.
If you go through our docs, you'll notice that we instruct you to launch the train script (`scripts/train/train.py`) and eval script (`scripts/eval/eval.py`) using the launcher, like so,

```bash
cd scripts/train
composer train.py <path/to/your/training/yaml>
```

The `composer` launcher puts all your GPUs to work by launching the script on a separate process for each device. The Trainer handles the rest.

### StreamingDataset

The training script contains logic for building a few different types of dataloaders used for different training tasks.
Each of these dataloaders are built to work with **streaming datasets**.
There are a number of benefits that come from using streaming datasets, from fast, deterministic resumption to easily loading from a mixture of streams at once.

The scripts in `scripts/data_prep/` are your one-stop-shop for converting a local dataset or a dataset on the Hugging Face Hub to our streaming MDS format.
These conversion scripts also allow you to upload your converted datasets directly to remote storage like s3, which our streaming datasets can read from.

### MCLI

`mcli` (short for MosaicML platform's Command Line Interface) is your gateway to scaling up training, eval, and inference on the MosaicML Platform. Access to the Platform is available to MosaicML customers (which you will need to set up separately). The `mcli/` directory includes several example YAMLs that demonstrate running various `llm-foundry` workloads on a remote cluster using `mcli`.

## How the YAMLs work

There are YAMLs in three locations: `scripts/train/yamls`, `scripts/eval/yamls`, and `mcli/`.
The `train` YAMLs pass arguments to `scripts/train/train.py`, and the `eval` YAMLs pass arguments to `scripts/train/eval.py`.
Both of these scripts, `train.py` and `eval.py`, wrap a `composer` Trainer in an opinionated way to make it easy to train and evaluate (respectively) LLMs.

The scripts in `mcli/` are used to submit a training job to the MosaicML platform using our MosaicML CLI.
Sign up [here](https://forms.mosaicml.com/demo?utm_source=home&utm_medium=mosaicml.com&utm_campaign=always-on).

# Example Workflows

## Workflow 1: I want to play with a HF model like MPT-7B locally

The quickest way to get started is to use the `transformers` library to download one of our MPT-7B models ([base](https://huggingface.co/mosaicml/mpt-7b), [chat](https://huggingface.co/mosaicml/mpt-7b-chat), [instruct](https://huggingface.co/mosaicml/mpt-7b-instruct)) and running a `text-generation` pipeline. You may see some UserWarnings appear due to MPT being a custom model, but those warnings can be safely ignored.

<!--pytest.mark.skip-->
```python
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, pipeline

name = 'mosaicml/mpt-7b'

# Download config
config = AutoConfig.from_pretrained(name, trust_remote_code=True)
# (Optional) Use `triton` backend for fast attention. Defaults to `torch`.
# config.attn_config['attn_impl'] = 'triton'
# (Optional) Change the `max_seq_len` allowed for inference
# config.max_seq_len = 4096

# Download model source and weights
model = AutoModelForCausalLM.from_pretrained(
    name,
    config=config,
    torch_dtype=torch.bfloat16,  # or torch.float32
    trust_remote_code=True)

# Download tokenizer
tokenizer = AutoTokenizer.from_pretrained(name)

# Run text-generation pipeline
pipe = pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer,
    device='cuda:0',  # (Optional) to run on GPU 0
)
print(
    pipe('Here is a recipe for vegan banana bread:\n',
         max_new_tokens=100,
         do_sample=True,
         use_cache=True))
```

To play with more features like batching and multi-turn chat, check out our example scripts `scripts/inference/hf_generate.py` and `scripts/inference/hf_chat.py`, with instructions in the [inference README](https://github.com/mosaicml/llm-foundry/blob/main/scripts/inference/README.md).

## Workflow 2: I want to deploy an inference endpoint with a HF model like MPT-7B

This site is under construction :)

## Workflow 3: I want to fine-tune a HF model like MPT-7B

### Supervised FineTuning and Instruction FineTuning

We have two resources for supervised finetuning:

1. [**LLM Finetuning from a Local Dataset: A Concrete Example**](https://github.com/mosaicml/llm-foundry/blob/main/scripts/train/finetune_example/README.md)
2. [The YAML which should replicate the process of creating MPT-7B-Instruct from MPT-7b](https://github.com/mosaicml/llm-foundry/blob/main/scripts/train/yamls/finetune/mpt-7b_dolly_sft.yaml) — You can point this at your own dataset by [following these instructions](https://github.com/mosaicml/llm-foundry/blob/main/scripts/train/README.md#Usage)

### Domain Adaptation and Sequence Length Adaptation

> **Note**
> Finetuning MPT-7B requires ≥ 4x40GB A100s, and a similarly sized model without flash attention may take 8 or more, depending on your sequence length. Use a smaller model if you do not have enough GPUs.

Domain and Sequence Length Adaptation are two similar cases that do not fit neatly into the pretraining/finetuning taxonomy. For the purposes of LLM-Foundry, it is more instructive to consider them “continued pretraining”, as our setup will more resemble pretraining than it does Supervised Fine Tuning. In particular, we will employ the same dataloader and data preparation strategy as used in pretraining.

For the purposes of this example, we will assume you are fine-tuning MPT-7B on a longer sequence length, but the same process would work for a new style of text (e.g. getting MPT-7B to work on, say, legal text). Note that the bigger the change, the more tokens you want to continue training on: extending the sequences to 4,096 does not require as many training steps as extending to 65,536. Similarly, adapting MPT-7B to code (which made up a significant fraction of its training data) does not require as many steps as adapting to legal documents in Hindi (which made up ~0% of its training data).

#### Data

First we need to pre-tokenize our data and concatenate it to fill up each sequence, as this keeps us from wasting any compute on pad tokens. The canonical reference for this is `scripts/data_prep/README.md`

If you are doing Sequence Length Adaptation, remember to adapt the above example to use your longer sequence length. Since we are using ALiBi, you can train on shorter sequences than you plan to use for evaluation; you can go somewhere between 20% and 100% longer, depending on how long your sequences are and the nuances of your data. For this example, suppose you want to do inference on sequences that are around 6,000 tokens long; for this it makes sense to train on 4,096 and then rely on ALiBi’s zero-shot length extrapolation at inference time.

Output the processed data to `./my-adaptation-data`. Note that we use smaller subsets of C4 as an example; you may have different data you want to use. Following [the data preparation README](https://github.com/mosaicml/llm-foundry/blob/main/scripts/data_prep/README.md), we convert C4 as follows:

<!--pytest.mark.skip-->
```bash
python scripts/data_prep/convert_dataset_hf.py \
  --dataset c4 --data_subset en \
  --out_root my-adaptation-data --splits train_small val_small \
  --concat_tokens 4096 --tokenizer EleutherAI/gpt-neox-20b --eos_text '<|endoftext|>' \
  --compression zstd
```

#### Modeling

Now that we have our data ready, we can slightly modify `scripts/yamls/pretrain/mpt-7b.yaml` to fit our purposes, changing `max_seq_len` to `4096` and the directory `data_local` to `./my-adaptation-data`. We have already created this YAML for you:

<!--pytest.mark.skip-->
```bash
composer scripts/train/train.py scripts/yamls/finetune/mpt-7b_domain_adapt.yaml
```

You will see some info logs including your configs, and then training will start.

After you're done training, you probably want to convert your Composer checkpoint to HuggingFace/ONNX/FasterTransformer format. To do that, check out the [inference README](https://github.com/mosaicml/llm-foundry/blob/main/scripts/inference/README.md).

## Workflow 4: I want to train a new HF model from scratch

> **Note**
> Pretraining for 10s of billions of tokens is a large job even for a smaller model; you’ll want multiple A100s for this example.

It is conceivable that you would like to train a model *with the same architecture* as a model available in HuggingFace `transformers` but without using those same weights; for example, if you have a large amount of proprietary data, or want to change something about the model that is hard to change after the fact. So, as an example, let’s say you want a version of `gpt2`  but with longer sequence length, say 2048. Using the MPT architecture would give us Flash Attention and ALiBi, allowing us to go much longer; but for this example we stick with 2048. And of course, let’s use 150 tokens/parameter, which is the ratio that MPT-7B used, getting us to 17.55B tokens for our 117M param model.

The first step to training from scratch is to get your pretraining data prepared.  Following [the data preparation README](https://github.com/mosaicml/llm-foundry/blob/main/scripts/data_prep/README.md), we convert C4 as follows:

<!--pytest.mark.skip-->
```bash
python convert_dataset_hf.py \
  --dataset c4 --data_subset en \
  --out_root my-copy-c4 --splits train_small val_small \
  --concat_tokens 2048 --tokenizer gpt2 \
  --eos_text '<|endoftext|>' \
  --compression zstd
```

Now we kick off a training using the configuration located at `scripts/yamls/pretrain/gpt2-small.yaml`:

<!--pytest.mark.skip-->
```bash
composer scripts/train/train.py scripts/yamls/pretrain/gpt2-small.yaml \
    max_seq_len=2048 \
    train_loader.dataset.split=train_small \
    eval_loader.dataset.split=val_small \
```

After you're done training, you probably want to convert your Composer checkpoint to HuggingFace/ONNX/FasterTransformer format. To do that, check out the [inference README](https://github.com/mosaicml/llm-foundry/blob/main/scripts/inference/README.md).

# FAQs

### Common installation issues
- TODO…

### Why is the script only using 1 out of N GPUs?
- Make sure you are using the `composer` launcher instead of the `python` launcher:

   ✅ `composer train/train.py ...`

   ❌ `python train/train.py ...`

  The `composer` launcher is responsible for detecting the available GPUs and launching N processes with the correct distributed environment details. See [the launcher script here](https://github.com/mosaicml/composer/blob/dev/composer/cli/launcher.py) for more details.

### I’m running into an Out-Of-Memory (OOM) error. What do I do?
- Hardware limitations may simply prevent some training/inference configurations, but here are some steps to troubleshooting OOMs.
- First, confirm that you are running with the `composer` launcher, e.g. `composer train/train.py ...`, and using all N GPUs? If not, you may be running into OOMs because your model is not being FSDP-sharded across N devices.
- Second, confirm that you have turned on FSDP for model sharding. For example, YAMLs for the `train.py` script should have a `fsdp_config` section. And you need to use `fsdp_config.sharding_strategy: FULL_SHARD` to enable parameter sharding.
- Third, confirm that you are using mixed precision, for example by setting `precision: amp_bf16`.
- If you are still seeing OOMs, reduce the `device_train_microbatch_size` or `device_eval_batch_size` which will reduce the live activation memory.
- If OOMs persist with `device_train_microbatch_size: 1` and `device_eval_batch_size: 1`, you may need to use activation checkpointing `fsdp_config.activation_checkpointing: true` (if you are not already) and, as a last resort, activation CPU offloading `fsdp_config.activation_cpu_offload: true`.

### What hardware can I train on?
- In general, this repo should work on any system with NVIDIA GPUs. Checkout the `scripts/train/README.md` for more [details on GPU memory requirements]([https://github.com/mosaicml/llm-foundry/tree/main/scripts/train#how-many-gpus-do-i-need-to-train-a-llm](https://github.com/mosaicml/llm-foundry/tree/main/scripts/train#how-many-gpus-do-i-need-to-train-a-llm)). Keep in mind you may run into issues with `Triton` support on some GPU types. In that situation, you can fall back to `attn_impl: torch` or raise an issue in the [Triton github repo](https://github.com/openai/triton).

### What hardware can I run eval on?
- Similar to above…

### What hardware can I run inference on?
- Similar to above…

### What is the MosaicML Training platform?
- This is

### What is the MosaicML Inference platform?
- This is

### What is FSDP?
- TODO

### What are the different attention options `torch` / `flash` / `triton`  for MPT and which one should I use?
- TODO

### Can I finetune using PEFT / LORA?
- The LLM Foundry codebase does not directly have examples of PEFT or LORA workflows. However, our MPT model is a subclass of HuggingFace `PretrainedModel`, and we are working on adding the remaining features to enable HuggingFace’s [PEFT](https://huggingface.co/docs/peft/index) / [LORA](https://huggingface.co/docs/peft/conceptual_guides/lora) workflows for MPT.

### Can I quantize these models and/or run on CPU?
- The LLM Foundry codebase does not directly have examples of quantization or limited-resource inference. But you can check out [GGML](https://github.com/ggerganov/ggml) (same library that powers llama.cpp) which has built support for efficiently running MPT models on CPU!

### How do I deploy with ONNX/FasterTransformer?
- Check out the `scripts/inference` directory for instructions and scripts.

### How expensive is it to build LLMs?
- Check out our blog post [GPT3-Quality for <$500k](https://www.mosaicml.com/blog/gpt-3-quality-for-500k) for guidance on LLM training times and costs.

  You can also check out our `scripts/train/benchmarking` folder for up-to-date information on the training throughput of MPT models using LLM Foundry. This datasheet can be used to answer questions like: “If I want to train an MPT-13B with context length 8k on 128xA100-40GB, what training throughput in tokens/sec should I expect?”
