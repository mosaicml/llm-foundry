# LLM Foundry Tutorial



**Hello!** We’ve put together this “tutorial” to help you develop a stronger familiarity with `llm-foundry`, the kinds of things you can do with it, and how to use it to meet your needs.

Forging LLMs can be quite complicated — you have to get your data prepared, set up your environment correctly with adequate hardware in order to train, evaluate your model once it’s trained, and finally get it ready to be served. That’s a lot of moving parts! And that’s exactly why we decided to create and release `llm-foundry`. This repo aims to simplify each of those pieces of the pipeline into an easy-to-use toolkit.

This tutorial will provide a brief intro to the repo’s structure and underlying tools (all courtesy of MosaicML, of course), will go over a few example workflows and point you to the related resources within the repo, and will finally cover a number of FAQs that we have encountered since release.

- [LLM Foundry Tutorial](#llm-foundry-tutorial)
- [Intro](#intro)
  - [How this repo is structured](#how-this-repo-is-structured)
  - [Key components](#key-components)
    - [Composer](#composer)
    - [StreamingDataset](#streamingdataset)
    - [MCLI](#mcli)
  - [How the YAMLs work](#how-the-yamls-work)
- [Example Workflows](#example-workflows)
  - [Workflow 1: I want to play with a HF model like MPT-7B locally](#workflow-1-i-want-to-play-with-a-hf-model-like-mpt-7b-locally)
  - [Workflow 2: I want to deploy an inference endpoint with a HF model like MPT-7B](#workflow-2-i-want-to-deploy-an-inference-endpoint-with-a-hf-model-like-mpt-7b)
  - [Workflow 3: I want to finetune a HF model like MPT-7B](#workflow-3-i-want-to-finetune-a-hf-model-like-mpt-7b)
    - [Supervised FineTuning and Instruction FineTuning](#supervised-finetuning-and-instruction-finetuning)
    - [Domain Adaptation and Sequence Length Adaptation](#domain-adaptation-and-sequence-length-adaptation)
      - [Data](#data)
      - [Modeling](#modeling)
  - [Workflow 4: I want to train a new HF model from scratch](#workflow-4-i-want-to-train-a-new-hf-model-from-scratch)
- [FAQs](#faqs)
    - [Why is the script only using 1 out of N GPUs?](#why-is-the-script-only-using-1-out-of-n-gpus)
    - [I’m running into an Out-Of-Memory (OOM) error. What do I do?](#im-running-into-an-out-of-memory-oom-error-what-do-i-do)
    - [What hardware can I train on?](#what-hardware-can-i-train-on)
    - [What hardware can I run eval on?](#what-hardware-can-i-run-eval-on)
    - [What hardware can I run inference on?](#what-hardware-can-i-run-inference-on)
    - [What is FSDP?](#what-is-fsdp)
    - [What are the different attention options `torch` / `flash` / `triton`  for MPT and which one should I use?](#what-are-the-different-attention-options-torch--flash--triton--for-mpt-and-which-one-should-i-use)
      - [Limitations](#limitations)
      - [What is `triton-pre-mlir`?](#what-is-triton-pre-mlir)
      - [Known issue with sm86+ GPUs](#known-issue-with-sm86-gpus)
      - [Support for FlashAttention-2](#support-for-flashattention-2)
    - [What kinds of positional embeddings does LLM Foundry support?](#what-kinds-of-positional-embeddings-does-llm-foundry-support)
    - [Can I finetune using PEFT / LoRA?](#can-i-finetune-using-peft--lora)
    - [Can I quantize these models and/or run on CPU?](#can-i-quantize-these-models-andor-run-on-cpu)
    - [How do I deploy with ONNX/FasterTransformer?](#how-do-i-deploy-with-onnxfastertransformer)
    - [TransformerEngine and amp\_fp8 support](#transformerengine-and-amp_fp8-support)
    - [How expensive is it to build LLMs?](#how-expensive-is-it-to-build-llms)
    - [Common installation issues](#common-installation-issues)

Let’s get started!



# Intro

This section establishes some basics that will provide useful context when navigating `llm-foundry` and digging into the provided scripts, YAMLs, etc. The goals here are to give you a clear sense of the general layout, orient you to the core MosaicML tools that this repo builds on, and introduce the way we use YAMLs to configure some of the more complex scripts.

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

Each of the above directories has its own `README.md` which goes into more depth about how to use the code it contains. To get the fullest picture on how `llm-foundry` works, make sure to give those a read too.

## Key components

There are 3 key libraries (all from MosaicML) that power `llm-foundry` and which you'll see throughout. These are worth covering a bit more, so in this section we'll briefly go over [Composer](https://docs.mosaicml.com/projects/composer/en/latest/), our distributed training engine, [Streaming](https://docs.mosaicml.com/projects/streaming/en/stable/), which enables streaming datasets, and [MCLI](https://docs.mosaicml.com/projects/mcli/en/latest/), which you can use to train on the MosaicML Platform.

### Composer

The Composer library is the workhorse of our training and evaluation scripts.
If you dig into those scripts, you'll notice that they are basically just very configurable wrappers around the Composer [Trainer](https://docs.mosaicml.com/projects/composer/en/latest/trainer/using_the_trainer.html).
The Trainer is a pytorch-native object that composes your model, dataset(s), optimizer, and more into a cohesive training pipeline with all the bells and whistles.
Spending some time understanding the Composer Trainer is a great way to form a deeper understanding of what the train and eval scripts are doing under the hood.

Composer also comes packaged with the `composer` launcher.
If you go through our docs, you'll notice that we instruct you to launch the training script (`scripts/train/train.py`) and eval script (`scripts/eval/eval.py`) using the launcher, like so,

<!--pytest.mark.skip-->
```bash
cd scripts/train
composer train.py <path/to/your/training/yaml>
```

The `composer` launcher puts all your GPUs to work by launching the script on a separate process for each device. The Trainer handles the rest.

### StreamingDataset

The training script contains logic for building a few different types of dataloaders used for different training tasks.
Each of these dataloaders is built to work with **streaming datasets**.
There are a number of benefits that come from using streaming datasets, from fast, deterministic resumption to easily loading from a mixture of streams at once.

The scripts in `scripts/data_prep/` are your one-stop-shop for converting a local dataset or a dataset on the Hugging Face Hub to our streaming MDS format.
These conversion scripts also allow you to upload your converted datasets directly to remote storage like S3, which our streaming datasets can read from.

### MCLI

`mcli` (short for MosaicML platform's Command Line Interface) is your gateway to scaling up training, eval, and inference on the MosaicML Platform. Access to the Platform is available to MosaicML customers (which you will need to set up separately). The `mcli/` directory includes several example YAMLs that demonstrate running various `llm-foundry` workloads on a remote cluster using `mcli`.

## How the YAMLs work

You'll find a lot of YAMLs in this repo. That's because they are a convenient tool for managing configs, which is what we use them for.

Config YAMLs are used as inputs to `scripts/train/train.py` and `scripts/eval/eval.py`, and are the main way we configure runs launched with `mcli`.

Both of the above scripts, `train.py` and `eval.py`, wrap a Composer Trainer in an opinionated way to make it easy to train and evaluate (respectively) LLMs. The bulk of each script essentially just interprets the config YAML it receives to build the appropriate inputs to the Trainer.

**We strive to keep the names of the YAML fields as closely related as possible to the kwargs of the function/class they direct to.** For instance, here's an example snippet for the `model` portion:
```yaml
model:
  name: hf_causal_lm
  pretrained: true
  pretrained_model_name_or_path: mosaicml/mpt-7b
```
If you dig into `train.py`, you'll find that `model.name: hf_causal_lm` instructs the model builder to create a [ComposerHFCausalLM](https://github.com/mosaicml/llm-foundry/blob/main/llmfoundry/models/hf/hf_causal_lm.py) object. The fields `pretrained` and `pretrained_model_name_or_path` correspond to the same kwargs used by the Hugging Face constructors that class builds from.

The YAMLS in `mcli/` are used to submit a training job to the MosaicML platform using our MosaicML CLI.
Sign up [here](https://forms.mosaicml.com/demo?utm_source=home&utm_medium=mosaicml.com&utm_campaign=always-on).
You can find more info about how to configure mcli YAMLs [here](https://docs.mosaicml.com/projects/mcli/en/latest/).

# Example Workflows

In this section, we’ll give a brief overview of 4 different workflows. You can treat them as independent — that is, you don’t need to go through each in any particular order. Instead, the goal here is to give you a sense of how you might approach each of these different situations using `llm-foundry` and related tooling.

## Workflow 1: I want to play with a HF model like MPT-7B locally

The quickest way to get started is to use the `transformers` library to download one of our MPT-7B models ([base](https://huggingface.co/mosaicml/mpt-7b), [chat](https://huggingface.co/mosaicml/mpt-7b-chat), [instruct](https://huggingface.co/mosaicml/mpt-7b-instruct)) and running a `text-generation` pipeline. You may see some UserWarnings appear due to MPT being a custom model, but those warnings can be safely ignored.

<!--pytest.mark.skip-->
```python
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, pipeline

name = 'mosaicml/mpt-7b'

# Download config
config = AutoConfig.from_pretrained(name, trust_remote_code=True)
# (Optional) Use `flash` (preferred) or `triton` backend for fast attention. Defaults to `torch`.
# config.attn_config['attn_impl'] = 'flash'
# (Optional) Change the `max_seq_len` allowed for inference
# config.max_seq_len = 4096

dtype = torch.bfloat16  # or torch.float32

# Download model source and weights
model = AutoModelForCausalLM.from_pretrained(
    name,
    config=config,
    torch_dtype=dtype,
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
with torch.autocast('cuda', dtype=dtype):
    print(
        pipe('Here is a recipe for vegan banana bread:\n',
            max_new_tokens=100,
            do_sample=True,
            use_cache=True))

```

Note: when running Torch modules in lower precision, it is best practice to use the [torch.autocast context manager](https://pytorch.org/docs/stable/amp.html).

To play with more features like batching and multi-turn chat, check out our example scripts `scripts/inference/hf_generate.py` and `scripts/inference/hf_chat.py`, with instructions in the [inference README](https://github.com/mosaicml/llm-foundry/blob/main/scripts/inference/README.md).

## Workflow 2: I want to deploy an inference endpoint with a HF model like MPT-7B

This site is under construction :)

Please check back soon for more info.

## Workflow 3: I want to finetune a HF model like MPT-7B

We address two possible versions of “finetuning” here. For both, you’ll want to be familiar with the material covered in `scripts/train/README.md`. The first finetuning workflow applies if you have **labeled data** — where you want to train the model to produce a target output given some input. The second workflow applies if you have additional **unlabeled data** that you want to adapt the model to.

### Supervised FineTuning and Instruction FineTuning

`scripts/train/` already includes some resources for supervised finetuning. If that’s what you’re interested in check out

1. [**LLM Finetuning from a Local Dataset: A Concrete Example**](https://github.com/mosaicml/llm-foundry/blob/main/scripts/train/finetune_example/README.md)
2. [The YAML which should replicate the process of creating MPT-7B-Instruct from MPT-7b](https://github.com/mosaicml/llm-foundry/blob/main/scripts/train/yamls/finetune/mpt-7b_dolly_sft.yaml) — You can point this at your own dataset by [following these instructions](https://github.com/mosaicml/llm-foundry/blob/main/scripts/train/README.md#Usage)

### Domain Adaptation and Sequence Length Adaptation

> **Note**
> Finetuning MPT-7B requires ≥ 4x40GB A100s, and a similarly sized model without flash attention may take 8 or more, depending on your sequence length. Use a smaller model if you do not have enough GPUs.

Domain and Sequence Length Adaptation are two similar cases that do not fit neatly into the pretraining/finetuning taxonomy. For the purposes of LLM Foundry, it is more instructive to consider them “continued pretraining”, as our setup will more resemble pretraining than it does Supervised Fine Tuning. In particular, we will employ the same dataloader and data preparation strategy as used in pretraining.

For the purposes of this example, we will assume you are "fine-tuning" MPT-7B on a longer sequence length, but the same process would work for a new style of text (e.g. getting MPT-7B to work on, say, legal text). Note that the bigger the change, the more tokens you want to continue training on: extending the sequences to 4,096 does not require as many training steps as extending to 65,536. Similarly, adapting MPT-7B to code (which made up a significant fraction of its training data) does not require as many steps as adapting to legal documents in Hindi (which made up ~0% of its training data).

#### Data

First we need to pre-tokenize our data and concatenate it to fill up each sequence, as this keeps us from wasting any compute on pad tokens. The canonical reference for this is `scripts/data_prep/README.md`.

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

Now that we have our data ready, we can slightly modify `scripts/train/yamls/finetune/mpt-7b_domain_adapt.yaml` to fit our purposes, changing `max_seq_len` to 4096 and the directory data_local to `./my-adaptation-data`. We *could* create a new YAML to do this, then point the trainer to it, but there is no need to. We can change these values as we kick off the training by supplying the override values as additional arguments:

<!--pytest.mark.skip-->
```bash
composer scripts/train/train.py scripts/train/yamls/finetune/mpt-7b_domain_adapt.yaml max_seq_len=4096 ...
```
> Note that this override where we set `max_seq_len=4096` in the above command works because of how the whole YAML is set up. Importantly, the YAML is configured with `model.config_overrides.max_seq_len: ${max_seq_len}`, which tells the MPT model to override its default max sequence length with the value set for `max_seq_len`.

You will see some info logs including your configs, and then training will start.

After you're done training, you probably want to convert your Composer checkpoint to HuggingFace/ONNX/FasterTransformer format. To do that, check out the [inference README](https://github.com/mosaicml/llm-foundry/blob/main/scripts/inference/README.md).

## Workflow 4: I want to train a new HF model from scratch

> **Note**
> Pretraining for 10s of billions of tokens is a large job even for a smaller model; you’ll want multiple A100s for this example.

It is conceivable that you would like to train a model *with the same architecture* as a model available in HuggingFace `transformers` but without using those same weights; for example, if you have a large amount of proprietary data, or want to change something about the model that is hard to change after the fact. So, as an example, let’s say you want a version of `gpt2`  but with a longer sequence length, say 2048. Using the MPT architecture would give us Flash Attention and ALiBi, allowing us to go much longer; but for this example we stick with 2048. And of course, let’s use 150 tokens/parameter, which is the ratio that MPT-7B used, getting us to 17.55B tokens for our 117M param model.

The first step to training from scratch is to get your pretraining data prepared.  Following [the data preparation README](https://github.com/mosaicml/llm-foundry/blob/main/scripts/data_prep/README.md), we convert C4 as follows:

<!--pytest.mark.skip-->
```bash
python scripts/data_prep/convert_dataset_hf.py \
  --dataset c4 --data_subset en \
  --out_root my-copy-c4 --splits train_small val_small \
  --concat_tokens 2048 --tokenizer gpt2 \
  --eos_text '<|endoftext|>' \
  --compression zstd
```

Now we kick off a training using the configuration located at `scripts/train/yamls/pretrain/gpt2-small.yaml`:

<!--pytest.mark.skip-->
```bash
composer scripts/train/train.py scripts/train/yamls/pretrain/gpt2-small.yaml \
    max_seq_len=2048 \
    train_loader.dataset.split=train_small \
    eval_loader.dataset.split=val_small \
```

After you're done training, you probably want to convert your Composer checkpoint to HuggingFace/ONNX/FasterTransformer format. To do that, check out the [inference README](https://github.com/mosaicml/llm-foundry/blob/main/scripts/inference/README.md).

# FAQs

The purpose of this section is probably pretty self-evident. You’ve got questions and we’ve (hopefully) got answers. Here are some of the more common ones we’ve seen. Before filing an issue, please see if your question is addressed in one of these FAQs or in the READMEs.

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
- In general, this repo should work on any system with NVIDIA GPUs. Checkout the `scripts/train/README.md` for more [details on GPU memory requirements]([https://github.com/mosaicml/llm-foundry/tree/main/scripts/train#how-many-gpus-do-i-need-to-train-a-llm](https://github.com/mosaicml/llm-foundry/tree/main/scripts/train#how-many-gpus-do-i-need-to-train-a-llm)). We recommend using `Flash` attention instead of `Triton` attention, unless you're training Prefix Language Models (in which case use `Triton`). Keep in mind you may run into issues with `Flash` or `Triton` support on some GPU types. In that situation, you can fall back to `attn_impl: torch`, or raise an issue in the [Flash Attention github repo](https://github.com/Dao-AILab/flash-attention).

### What hardware can I run eval on?
- Similar to above…

### What hardware can I run inference on?
- Similar to above…

### What is FSDP?
- [Fully Sharded Data Parallel (FSDP)](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/) is a PyTorch implementation of the [Zero Redundancy Optimizer (ZeRO)](https://arxiv.org/abs/1910.02054). FSDP shards networks parameters and the optimizer state across all GPUs. This enables users to train models with large parameter counts which do not fit into a single GPUs memory.

### What are the different attention options `torch` / `flash` / `triton`  for MPT and which one should I use?
- **Short answer:** `torch` is the native pytorch attention implementation, and `flash` and `triton` are different implementations of the much more optimized [Flash Attention](https://arxiv.org/abs/2205.14135) method. `triton` and `flash` will be faster (and use less GPU memory) than `torch`, but they might not work with all hardware and environment setups.

  Our training setups typically use `flash`.

- **Long answer:** In NLP, Softmax Attention operates on a sequence. It is an all to all graph operation where, during training, the memory complexity is quadratic with respect to the length of the sequence. Furthermore, on GPUs, naive implementations of Softmax Attention are bandwidth (BW) limited.
[Rabe et al. (2021)](https://arxiv.org/abs/2112.05682) and [Dao et al. (2022)](https://arxiv.org/abs/2205.14135) showed that fusing all operations in Softmax Attention can make the operation much less BW limited.
Furthermore, integrating a recomputation schema decreases the sequence length memory complexity from *quadratic* to *linear*, thereby supporting much longer sequence lengths.

  - Setting `attn_config.attn_impl=torch` enables a naive Softmax Attention written using base torch operations.
  - Setting `attn_config.attn_impl=flash` enables Flash Attention [implemented by Dao et al in the Dao-AILab repo using CUDA](https://github.com/Dao-AILab/flash-attention). This will have linear memory complexity (enabling larger batch sizes) and will run much faster.
  - Setting `attn_config.attn_impl=triton` enables a Flash Attention [implemented using Triton](https://github.com/mosaicml/llm-foundry/blob/main/llmfoundry/models/layers/flash_attn_triton.py). We recommend using `flash` attention instead of `triton` attention, unless you're training Prefix Language Models (in which case use `Triton`).

<!-- In NLP, Softmax Attention operates on a sequence. It is an all to all graph operation where, during training, the memory complexity is quadratic with respect to the length of the sequence. Furthermore, on GPUs, naive implementations of Softmax Attention are BW limited.
[Rabe et al. (2021)](https://arxiv.org/abs/2112.05682) and [Dao et al. (2022)](https://arxiv.org/abs/2205.14135) noted that fusing all operations in Softmax Attention can make the operation much less BW limited.
Furthermore, integrating a recomputation schema decreases the sequence length memory complexity from quadratic to linear enabling practitioners to train transformer networks using much longer sequence lengths.

Setting `attn_config.attn_impl=torch` enables a naive Softmax Attention written using base torch operations.
Setting `attn_config.attn_impl=flash` enables flash attention [implemented by Dao et al in the HazyResearch repo using CUDA](https://github.com/HazyResearch/flash-attention). This will have linear memory complexity (enabling larger batch sizes) and will run much faster.
Setting `attn_config.attn_impl=triton` enables a flash attention [implemented using Triton](https://github.com/mosaicml/llm-foundry/blob/main/llmfoundry/models/layers/flash_attn_triton.py). In our experience, `triton` is slightly faster than `flash`.
The majority of our training setups use `triton`. -->

#### Limitations
- For training, `torch` uses a lot of memory and is slow.
- `flash` and `triton` cannot return attention weights and therefore cannot be used with methods that require it.
- `flash` cannot accept an attention bias. However, it still allows the use of ALiBi positional bias.

#### What is `triton-pre-mlir`?
- Torch2 installs and requires a specific version of [Triton](https://openai.com/research/triton).
  `attn_config.attn_impl=triton` requires a different version of triton.
  As a result, you can either use torch2 or `attn_impl=triton`.
  To enable both, we fork triton and make it pip installable as `triton-pre-mlir`.
  `attn_impl=triton` can then use `triton-pre-mlir` leaving the version of triton required for torch2 intact.

#### Known issue with sm86+ GPUs
- Under the hood, part of `triton-pre-mlir` compile path uses LLVM11.
  H100 GPUs (sm90 GPUs) are not formally supported until LLVM15 (technically it doesn't support anything sm86+).
  Updating the LLVM version used by `triton-pre-mlir` to LLVM13 seems to be relatively easy.
  Updating to LLVM14 (or LLVM15) cannot be done because there are breaking changes.
  What is the result of this? Although sm89+ is not **formally** supported until LLVM15, our testing on H100 GPUs shows that `attn_impl=triton` still works well and still runs fast. The only issue is that when the network is starting to run, LLVM might throw a warning like: `'sm_90' is not a recognized processor for this target (ignoring processor)`. This warning does not seem to affect performance.

#### Support for FlashAttention-2
- [FlashAttention-2](https://arxiv.org/pdf/2307.08691.pdf) improves upon FlashAttention to get even faster attention computation. LLM Foundry supports FlashAttention-2. Please follow the instructions [here](https://github.com/mosaicml/llm-foundry/tree/main/scripts/train#flashattention).

### What kinds of positional embeddings does LLM Foundry support?
Currently we support [Learned Positional Embeddings](https://arxiv.org/pdf/1706.03762.pdf), [Attention with Linear Biases (ALiBi)](https://arxiv.org/pdf/2108.12409.pdf), and [Rotary Positional Embeddings (RoPE)](https://arxiv.org/pdf/2104.09864.pdf). There is also an option to switch off all of these embeddings to get [No Positional Embedding](https://arxiv.org/pdf/2203.16634.pdf).

| Name                               | YAML Config                                                       | Training MFU on MPT-7B trained on 8 A100 80GB GPUs | Notes                                                                                                                                                                       |
|:-----------------------------------|:------------------------------------------------------------------|:---------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Learned Positional Embeddings      | <pre>model:<br>     learned_pos_emb:&nbsp;True</pre>| 65.7                                                |                                                                                                                                                                             |
| ALiBi                              | <pre>model:<br>     attn_config:<br>         alibi:&nbsp;True</pre>| 64.5                                                |  Requires Flash (v2.4.2 or higher) or Triton or Torch attention.                                                                                                                                        |
| RoPE (Dao-AILab Implementation)    | <pre>model:<br>     attn_config:<br>         rope:&nbsp;True<br>         rope_impl:&nbsp;dail</pre>| 64.5                                                | Requires a CUDA GPU and the [flash-attn library](https://github.com/Dao-AILab/flash-attention) v2.0.1 or higher to be installed. Please see the instructions in the [paragraph above](#support-for-flashattention-2) on how to install flash-attn v2. Note that the attention implementation can still be `torch`, `triton`, or `flash`. |
| RoPE (Hugging<code>&nbsp;</code>Face Implementation)  | <pre>model:<br>     attn_config:<br>         rope:&nbsp;True<br>         rope_impl:&nbsp;hf</pre>| 62.3                                                |                                                                                                                                                                             |

### Can I finetune using PEFT / LoRA?
- LLM Foundry does support LoRA via an integration with the [PEFT](https://github.com/huggingface/peft) library. Within LLM Foundry, run (`scripts/train/train.py`), adding `peft_config` arguments to the `model` section of the config `.yaml`, like so:
<!--pytest.mark.skip-->
```yaml
model:
  ...
  peft_config:
      r: 16
      peft_type: LORA
      task_type: CAUSAL_LM
      lora_alpha: 32
      lora_dropout: 0.05
      target_modules:
      - q_proj
      - k_proj
    target_modules: 
    - 'Wqkv'
```
- For efficiency, The MPT model concatenates the `Q`, `K`, and `V` matrices in each attention block into a single `Wqkv` matrix that is three times wider. Currently, LoRA supports a low-rank approximation to this `Wqkv` matrix.
- When evaluating with PEFT / LoRA separated weight, just set `pretrained_lora_id_or_path` in `model`(Find an example [here](scripts/eval/yamls/hf_lora_eval.yml#L19)).

### Can I quantize these models and/or run on CPU?
- The LLM Foundry codebase does not directly have examples of quantization or limited-resource inference. But you can check out [GGML](https://github.com/ggerganov/ggml) (same library that powers llama.cpp) which has built support for efficiently running MPT models on CPU! You _can_ load your model in 8-bit precision for inference using the [bitsandbytes library](https://github.com/TimDettmers/bitsandbytes) and Hugging Face's [accelerate](https://huggingface.co/docs/accelerate/index) via `load model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map="auto", trust_remote_code=True)`, although we have not extensively benchmarked the performance (see the Hugging Face [quantization documentation](https://huggingface.co/docs/transformers/main/main_classes/quantization) for more detail).

### How do I deploy with ONNX/FasterTransformer?
- Check out the `scripts/inference` directory for instructions and scripts.

### TransformerEngine and amp_fp8 support
Once [installed](https://github.com/mosaicml/llm-foundry/tree/main#TransformerEngine-and-amp_fp8-support), if you are using an H100, you can use fp8 with te layers by setting eg:
<!--pytest.mark.skip-->
```yaml
precision: amp_fp8

model:
  fc_type: te
```
in the training yaml.

Setting
<!--pytest.mark.skip-->
```yaml
model:
  ffn_config_defaults:
    ffn_type: te_ln_mlp
```
enables [TransformerEngine's LayerNormMLP](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/pytorch.html#transformer_engine.pytorch.LayerNormMLP) layer which enables sequence parallelism if configured correctly.

WARNING: `state_dicts` generated with `ffn_type: te_ln_mlp` will NOT directly map to `state_dicts` generated using the default network configurations. We do not have control over how `te.LayerNormMLP` is implemented and therefore cannot readily reconcile it with the default implementation (or any other implementation).

### How expensive is it to build LLMs?
- Check out our blog post [GPT3-Quality for <$500k](https://www.mosaicml.com/blog/gpt-3-quality-for-500k) for guidance on LLM training times and costs.

  You can also check out our `scripts/train/benchmarking` folder for up-to-date information on the training throughput of MPT models using LLM Foundry. This datasheet can be used to answer questions like: “If I want to train an MPT-13B with context length 8k on 128xA100-40GB, what training throughput in tokens/sec should I expect?”

### Common installation issues
- We're still putting this section together. In the meantime, please see the top-level README for our recommended installation.
