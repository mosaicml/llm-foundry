# LLM Finetuning from a Local Dataset: A Concrete Example

> **Note**
> Before going through this please make sure you have read the LLM Finetuning section of the main README (see parent directory). This guide assumes you have already done so!

The contents of this directory provide a concrete example of finetuning an LLM on a local dataset using the `llm-foundry` training tools.


***

## Contents

Here, we have a minimal example that includes all the necessary pieces:
- `train.jsonl`: Our local dataset. (It is actually just a 100-example snippet of the ARC Easy ICL evaluation set, so it's not something we'd want train on for real.)
- `preprocessing.py`: A python file that defines the "preprocessing function" we will use to format our dataset into the required "prompt"/"response" structure.
- `gpt2-arc-easy--cpu.yaml`: The configuration YAML for finetuning a pretrained gpt2 model on our local ARC Easy snippet with our custom preprocessing function. You can run this toy example on CPU in 3-4 minutes.
- `mpt-7b-arc-easy--gpu.yaml`: The configuration YAML for finetuning MPT-7B on our local ARC Easy snippet with our custom preprocessing function. This requires GPU(s).

## Quick start

<!--pytest.mark.skip-->
```bash
cd llm-foundry/scripts/train
composer train.py finetune_example/gpt2-arc-easy--cpu.yaml
```
That's it :)

***

# What's actually going on here?

This guide aims to provide an intuition for how to correctly set up finetuning when using a local dataset. You may also find it useful if training from a dataset on the Hugging Face Hub that requires a custom preprocessing function.

## Our local dataset

In this example, our data lives in the JSON file `train.jsonl`. Our finetuning code will load this data using the Hugging Face `datasets` library. Let's take a closer look.

> **Note**
> The code below assumes our working directory is `llm-foundry/scripts/train`

<!--pytest.mark.skip-->
```python
from datasets import load_dataset
my_dataset = load_dataset('json', data_dir='./finetune_example', split='train')
my_dataset[0]
>>> {'query': 'Question: Which statement best explains why photosynthesis is the foundation of most food webs?\n',
     'choices': [
        'Sunlight is the source of energy for nearly all ecosystems.',
        'Most ecosystems are found on land instead of in water.',
        'Carbon dioxide is more available than other gases.',
        'The producers in all ecosystems are plants.'],
     'gold': 0}
```

This dataset is currently formatted for a type of multiple-choice evaluation. It has a question `query`, a list of possible answers to the question `choices` (only one of which is correct), and the index of the correct choice `gold`.

LLM finetuning works by showing the model examples of input *prompts* and the *responses* it should generate. Given a prompt, the model is trained to produce the target response.

Therefore, we need to "preprocess" our dataset to ensure that each example has the required `{"prompt": ..., "response": ...}` format.

For example, let's say we want the model to take in prompts that include a question and a set of possible answers, and have it respond by outputting the correct answer. For example:

<!--pytest.mark.skip-->
```python
prompt = """Question: Which statement best explains why photosynthesis is the foundation of most food webs?

Options:
 - Sunlight is the source of energy for nearly all ecosystems.
 - Most ecosystems are found on land instead of in water.
 - Carbon dioxide is more available than other gases.
 - The producers in all ecosystems are plants.
Answer: """
response = "Sunlight is the source of energy for nearly all ecosystems."
```

## Our preprocessing function
The LLM finetuning tools can handle arbitrarily formatted datasets, such as our local dataset, as long as you also supply a **preprocessing function** that takes converts an example to the required format.

We have implemented such a preprocessing function in `preprocessing.py` (within this directory). The function is named `multiple_choice` since it maps the multiple-choice format of our starting dataset to the required prompt/response format. Let's take a closer look.

> **Note**
> The code below assumes our working directory is `llm-foundry/scripts/train`

<!--pytest.mark.skip-->
```python
# We'll load our local dataset as in the example above
from datasets import load_dataset
my_dataset = load_dataset('json', data_dir='./finetune_example', split='train')

# This time we'll invoke our preprocessing function
from finetune_example.preprocessing import multiple_choice

multiple_choice(my_dataset[0])
>>> {'prompt': 'Question: Which statement best explains why photosynthesis is the foundation of most food webs?\n\nOptions:\n - Sunlight is the source of energy for nearly all ecosystems.\n - Most ecosystems are found on land instead of in water.\n - Carbon dioxide is more available than other gases.\n - The producers in all ecosystems are plants.\nAnswer: ',
     'response': 'Sunlight is the source of energy for nearly all ecosystems.'}
```

Now we have a local dataset and a preprocessing function that will map examples in our dataset to the desired prompt/response format. All that's left is to configure our training YAML to use them.

## Our training YAML

This is already taken care of in the example YAMLs, specifically in the `train_loader` section that controls how the  train dataloader is built in the `train.py` script. (If you also have an accompanying validation dataset, you'd make similar changes in the `eval_loader` section). Let's take a closer look.

<!--pytest.mark.skip-->
```yaml
train_loader:
  name: finetuning
  dataset:
    ############
    hf_name: json
    hf_kwargs:
      # Note: absolute paths for data_dir are more reliable;
      # relative paths will be interpreted relative to whatever your
      # working directory is when you run `train.py`
      data_dir: finetune_example
    # Note: `scripts/train` will be the working directory when resolving
    # the preprocessing_fn import path
    preprocessing_fn: finetune_example.preprocessing:multiple_choice
    split: train
```

With this configuration, the finetuning code will *effectively* run the following code to construct the tokenized dataset used by the training dataloader:
<!--pytest.mark.skip-->
```python
from datasets import load_dataset
# (from "preprocessing_fn" in the config)
from finetune_example.preprocessing import multiple_choice
# (from "hf_name", "hf_kwargs", and "split" in the config)
dataset = load_dataset('json', data_dir='finetune_example', split='train')

def format_and_tokenize(example):
   formatted_example = multiple_choice(example)
   return tokenizer(text=formatted_example['prompt'], text_target=formatted_example['response'])

tokenized_dataset = dataset.map(format_and_tokenize)
```

Hopefully this clarifies how the dataloader settings in the YAML affect dataset construction!

As you can see, these config settings mainly just determine the inputs to `load_dataset`, from Hugging Face's `datasets` library. As such, we encourage you to review [their documentation](https://huggingface.co/docs/datasets/loading) to get a better sense of what our tooling will let you do!

A couple notes on that:
- You can use either the `data_dir` or `data_file` kwarg to tell `load_dataset` where your data live.
- `data_dir` tends to be a bit more reliable, but requires that your actual data files within the directory are named `train`, `validation`, and/or `test`.
- Whenever possible, use **absolute paths** instead of relative paths. Relative paths will depend on the working directory you are in when running the `train.py` script.

## ARC data

The data in `train.jsonl` is from the ARC corpus.

This work uses data from the AI2 Reasoning Challenge (ARC) 2018, created by and copyright [AI2 and Aristo​](https://allenai.org/data/arc)​. The data consists of 7,787 science exam questions and is intended for non-commercial, research purposes only. We use the first 100 examples of the Easy Test split, consisting of questions and answers, which we do not modify. The original data is available at https://allenai.org/data/arc and is licensed under a [Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/).
