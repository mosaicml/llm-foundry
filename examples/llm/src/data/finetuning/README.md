# LLM Finetuning

This directory contains utilities for Seq2Seq finetuning LLMs, for example, Supervised Finetuning (SFT) (aka Instruction(Fine)Tuning (IFT)), or finetuning a base LLM to focus on a specific task like summarization.

You can use the `main.py` script in the LLM examples directory to do finetuning. If you are unfamiliar with that script, or the LLM directory in general, you should first go through the README located at `examples/examples/llm/README.md`.

In this README, we'll cover how to use the finetuning utilities.

## Usage

You activate finetuning via the `train_loader` and `eval_loader` fields in your configuration YAML. We include some reference examples inside `llm/yamls/mosaic_gpt/finetuning/`.

There are 3 different types of data sources you can use for finetuning: (1) [the HuggingFace Hub](#1-using-a-dataset-on-the-huggingface-hub), (2) [a local dataset](#2-using-a-local-dataset), and (3) [a local or remote streaming dataset](#3-using-an-mds-formatted-dataset-locally-or-in-an-object-store). We'll cover these more below, but first will describe some important steps for all 3.

In all cases, you'll activate the finetuning dataloading codepath by setting `{train,eval}_loader.name` to `finetuning`, and providing the `dataset.name` like so:
```yaml
train_loader:
    name: finetuning
    dataset:
        name: my-finetuning-task
        ...
```

**IMPORTANT:** The subfield `dataset.name` has a special meaning. It tells the finetuning dataloader what function to use when tokenizing each example into the input and output sequences.
- "input" refers to the text that you feed into the model, e.g. *Tell me a few facts about dogs.*
- "output" refers to the text that the model is trained to produce in response to the input, e.g. *Dogs are great pets. They love to play fetch. They are better than cats...*

`dataset.name` must refer to a function in `tasks.py` that you have registered under that name. For example:
```python
@dataset_constructor.register('my-finetuning-task')
def my_tokenization_function(example: Dict, tokenizer: Tokenizer):
    """Map the input/output fields to the correct tokenizer kwargs."""
    # `text` is the text the model receives (i.e. the prompt)
    # `text_target` is the target output the model should produce
    return tokenizer(
        text=example["input_text"],
        text_target=example["output_text"],,
    )
```

These tokenization functions simply serve to handle dataset-specific formatting, where different field names are used to represent the input/output, the input/output need to be split out of a single text sequence, and so on. **You should look through `tasks.py` to see other examples and to build a clearer intuition.**

Now that we've covered that concept, we'll describe the 3 different usage patterns...

### 1) Using a dataset on the HuggingFace Hub

Let's say you want to finetune using a dataset available on the HuggingFace Hub. We'll pretend this dataset on the Hub is called `hf-hub/identifier`.

1. In `tasks.py`, write a tokenization function for processing the dataset, to split it into prompt and response
1. Register this function using `@dataset_constructor.register('hf-hub/identifier')` -- the registered name ("hf-hub/identifier") needs to match the name of the model on the Hub
1. Reference this in a training yaml, such as the one in `yamls/mosaic_gpt/finetune/7b_dolly_sft.yaml`
```yaml
train_loader:
    name: finetuning
    dataset:
        name: hf-hub/identifier
        split: train
        ...
```

### 2) Using a local dataset

Let's say you have your finetuning dataset stored in local `jsonl` files.

1. In `tasks.py`, write a function for processing the dataset, to split it into prompt and response
1. Register this function using `@dataset_constructor.register('some_name')` -- you can register this under any name you want, just set `dataset.name` in your yaml to have the same name
1. Reference this in a training yaml, such as the one in `yamls/mosaic_gpt/finetune/1b_local_data_sft.yaml`
```yaml
train_loader:
    name: finetuning
    dataset:
        name: some_name
        kwargs:
            data_files:
                train: /path/to/train.jsonl
        split: train
        ...
```

### 3) Using an MDS-formatted (streaming) dataset -- locally or in an object store

Let's say you made an [MDS-formatted dataset](https://github.com/mosaicml/streaming) (which you totally should -- they're amazing). For example, maybe you used the `convert_finetuning_dataset.py` script to convert a large HuggingFace dataset into a streaming format and saved it to S3.

1. In `tasks.py`, write a function for processing the dataset, to split it into prompt and response
1. Register this function using `@dataset_constructor.register('some_name')` -- you can register this under any name you want, just set `dataset.name` in your yaml to have the same name
1. Set the `dataset.remote` and `dataset.local` values in your YAML
```yaml
train_loader:
    name: finetuning
    dataset:
        name: some_name
        remote: s3://path/to/mds/dataset/
        local: /tmp/mds-cache/
        split: train
        ...
```
Note: `convert_finetuning_dataset.py` is a handy script for converting a HuggingFace dataset to MDS format. If you already wrote/registered a tokenization function for that dataset, you can skip steps 1 & 2 and continue using the name of the HuggingFace dataset for `dataset.name`. Setting `dataset.remote` and `dataset.local` will activate the streaming codepath.
