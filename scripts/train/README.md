# LLM Pretraining

## Installation

If you haven't already, make sure to install the requirements:

```bash
git clone https://github.com/mosaicml/llm-foundry.git
cd llm-foundry
pip install -e ".[gpu]"  # or pip install -e . if no NVIDIA GPU
```

## Dataset preparation
To run pretraining, you'll need to make yourself a copy of a pretraining dataset. Check out the `llm-foundry/data_prep` folder for detailed instructions.

As a quickstart, here is how to prepare the [C4: Colossal, Cleaned, Common Crawl dataset](https://huggingface.co/datasets/c4).
We first convert the dataset from its native format (a collection of zipped JSONs)
to MosaicML's streaming dataset format (a collection of binary `.mds` files).
Once in `.mds` format, we can store the dataset in a central location (filesystem, S3, GCS, etc.)
and stream the data to any compute cluster, with any number of devices, and any number of CPU workers, and it all ~ just works ~ .
You can read more about the benefits of using mosaicml-streaming [here](https://streaming.docs.mosaicml.com/en/stable/).

NOTE: If you only want to profile these LLMs, we recommend that you **download and prepare the `train_small` and `val_small` splits**,
and skip the full `train` and `val` splits. You'll just need to replace `split: train` with `split: train_small`
and `split: val` with `split: val_small` in your run YAML's dataloader config.
You can also accomplish this in your CLI command like so: `composer train.py ... train_loader.dataset.split=train_small eval_loader.dataset.split=val_small`
Alternatively, feel free to substitute our dataloader with one of your own in `train.py`.

### Converting C4 to streaming dataset `.mds` format
To make yourself a copy of C4, use `convert_dataset.py` like so:
<!--pytest.mark.skip-->
```bash
# Download the 'train_small' and 'val_small' splits and convert to StreamingDataset format
# This will take 20-60 seconds depending on your Internet bandwidth
# You should see two folders: `./my-copy-c4/train_small` and `./my-copy-c4/val` that are each ~0.5GB
# Note: We are using the `--concat_tokens` option to pre tokenize our samples to be of the max sequence length without padding
python ../data_prep/convert_dataset.py --dataset c4 --data_subset en --out_root ./my-copy-c4 --splits train_small val_small --concat_tokens 2048 --tokenizer gpt2 --eos_text '<|endoftext|>'

# Download the 'train' and 'val' splits if you really want to train the model (not just profile)
# This will take 1-to-many hours depending on bandwidth, # CPUs, etc.
# The final folder `./my-copy-c4/train` will be ~800GB so make sure you have space!
# python ../data_prep/convert_dataset.py --dataset c4 --data_subset en --out_root ./my-copy-c4 --splits train val --concat_tokens 2048 --tokenizer gpt2 --eos_text '<|endoftext|>'

# For any of the above commands, you can also choose to compress the .mds files.
# This is useful if your plan is to store these in object store after conversion.
# python ../data_prep/convert_dataset.py ... --compression zstd
```

### Test the Dataloader

To verify that the dataloader works, run a quick test on your `val_small` split like so:
<!--pytest.mark.skip-->
```bash
# This will construct a `StreamingTextDataset` dataset from your `val` split,
# pass it into a PyTorch Dataloader, and iterate over it and print samples.
# Since we only provide a local path, no streaming/copying takes place.
python ../data_prep/text_data.py --local_path ./my-copy-c4 --split val_small

# This will do the same thing, but stream data to {local} from {remote}.
# The remote path can be a filesystem or object store URI.
python ../data_prep/text_data.py --local_path /tmp/cache-c4 --remote_path ./my-copy-c4  --split val_small # stream from filesystem, e.g. a slow NFS volume to fast local disk
# python ../data_prep/text_data.py --local_path /tmp/cache-c4 --remote_path s3://my-bucket/my-copy-c4  # stream from object store
```

## How to start training

Now that you've installed dependencies and built a local copy of the C4 dataset, let's start training!

**Please remember** to edit the `data_local` and (optionally) `data_remote` paths in your YAML.
Our streaming dataloader always streams to `data_local` <- from <- `data_remote`, and if the remote path is missing, the files are expected to be present in `data_local`.

**Also remember** that if you only downloaded the `train_small` split, you need to make sure your train_loader uses that split. Just change `split: train` to `split: train_small` in your YAML.
And similarly for `val` and `val_small`.


### Single-Node training
We run the `train.py` script using our `composer` launcher, which generates N processes (1 per device).


If training on a single node, the `composer` launcher will autodetect the number of devices, so all you need to do is:
<!--pytest.mark.skip-->
```bash
composer train.py yamls/mosaic_gpt/125m.yaml
```

To train with high performance on multi-node clusters, the easiest way is with the MosaicML platform ;) Check out the `mcloud/` folder for examples!

But if you really must try this manually on your own cluster, then just provide a few variables to `composer`
either directly via CLI, or via environment variables that can be read. Then launch the appropriate command on each node:

### Multi-Node via CLI args

<!--pytest.mark.skip-->
```bash
# Using 2 nodes with 8 devices each
# Total world size is 16
# IP Address for Node 0 = [0.0.0.0]

# Node 0
composer --world_size 16 --node_rank 0 --master_addr 0.0.0.0 --master_port 7501 train.py yamls/mosaic_gpt/125m.yaml

# Node 1
composer --world_size 16 --node_rank 1 --master_addr 0.0.0.0 --master_port 7501 train.py yamls/mosaic_gpt/125m.yaml

```

### Multi-Node via environment variables

<!--pytest.mark.skip-->
```bash
# Using 2 nodes with 8 devices each
# Total world size is 16
# IP Address for Node 0 = [0.0.0.0]

# Node 0
# export WORLD_SIZE=16
# export NODE_RANK=0
# export MASTER_ADDR=0.0.0.0
# export MASTER_PORT=7501
composer train.py yamls/mosaic_gpt/125m.yaml

# Node 1
# export WORLD_SIZE=16
# export NODE_RANK=1
# export MASTER_ADDR=0.0.0.0
# export MASTER_PORT=7501
composer train.py yamls/mosaic_gpt/125m.yaml
```

You should see logs being printed to your terminal like so.
You can also easily enable other experiment trackers like Weights and Biases or CometML,
by using [Composer's logging integrations](https://docs.mosaicml.com/en/stable/trainer/logging.html).

<!--pytest.mark.skip-->
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


# LLM Finetuning

This repo also contains utilities for Seq2Seq finetuning for LLMs, for example, Supervised Finetuning (SFT) (aka Instruction(Fine)Tuning (IFT)), or finetuning a base LLM to focus on a specific task like summarization.

You can use the same `train.py` script to do finetuning.
If you are unfamiliar with that script, or the LLM Foundry in general, you should first go through the instructions above.

In this section, we'll cover how to use the finetuning utilities.

## Data formatting

You activate finetuning via the `train_loader` and `eval_loader` fields in your configuration YAML.
We include some reference examples inside `llm/yamls/mpt/finetuning/`.

There are 3 different types of data sources you can use for finetuning:
(1) [the HuggingFace Hub](#1-using-a-dataset-on-the-huggingface-hub),
(2) [a local dataset](#2-using-a-local-dataset), and
(3) [a local or remote streaming dataset](#3-using-an-mds-formatted-dataset-locally-or-in-an-object-store).
We'll cover these more below, but first will describe an important consideration for all 3: data formatting!

The finetuning dataloader requires training examples to be formatted as dictionaries with the following key-value structure:
<!--pytest.mark.skip-->
```python
formatted_example = {'prompt': <prompt_text>, 'response': <response_text>}
```
- "prompt" refers to the text that you feed into the model, e.g. *Tell me a few facts about dogs.*
- "response" refers to the text that the model is trained to produce in response to the prompt, e.g. *Dogs are great pets. They love to play fetch. They are better than cats...*

**How to ensure that your data follows that format.**
Our tooling attempts to simplify any reformatting by making it easy to insert **preprocessing functions** into the data pipeline. Here's a (simplified) example of a preprocessing function in `llmfoundry.data.finetuning.tasks`:
<!--pytest.mark.skip-->
```python
@dataset_constructor.register('tatsu-lab/alpaca')
def alpaca_preprocessing_function(inp: Dict):
    """Split out prompt/response from text."""
    prompt, response = inp['text'].split('### Response:')
    prompt +=  '### Response:'
    return {'prompt': prompt, 'response': response}
```

### Pre-defined preprocessing functions

As shown above, the preprocessing functions in `llmfoundry.data.finetuning.tasks` use the `register()` decorator to connect them to the HuggingFace datasets they apply to.
To get a list of all the HuggingFace datasets that already have preprocessing functions registered for them, you can run:

<!--pytest.mark.skip-->
```bash
python -c "from llmfoundry.data.finetuning.tasks import dataset_constructor; dataset_constructor.print_registered_tasks()"
```

### Custom data preprocessing

If the dataset you want to use is in that list, or if it already has the "prompt"/"response" format, you're in luck!
You can skip the rest of this section.

If not, you just need to write your own processing function.
You can write the function wherever is convenient for you, as long as it importable.

Let's say you want to finetune on a HuggingFace dataset named `mosaicml/doge-facts` (which, sadly, is made up for this example), and it contains examples that look like this:
<!--pytest.mark.skip-->
```python
>>> import datasets
>>> dogefacts = datasets.load_dataset('mosaicml/doge-facts', split='train')
>>> dogefacts[0]
{'question': 'What doge is the best doge?', 'answer': 'All of them!'}
>>> dogefacts[1]
{'question': 'When was the first doge?', 'answer': 'The original doge meme is based on a photograph taken in 2010.'}
```

The only preprocessing required here is to map "question"-->"prompt" and "answer"-->"response".
<!--pytest.mark.skip-->
```python
def dogefacts_prep_fn(inp: Dict):
    return {'prompt': inp['question'], 'response': inp['answer']}
```
For this example, let's say we add this function to a file that we can import from. For example, with
`from my_data.formatting import dogefacts_prep_fn`

## Usage

Now we'll cover the different ways you can use the finetuning utilities. This will mostly focus on how to configure your YAML, assuming you have already prepared any custom preprocessing functions as described above.
### **1) Using a dataset on the HuggingFace Hub**

Let's say you want to finetune using a dataset available on the HuggingFace Hub.
If the dataset has a [pre-defined preprocessing function](#pre-defined-preprocessing-functions), e.g., `tatsu-lab/alpaca`, or if the dataset already has the "prompt"/"response" format, simply point the dataloader to that dataset.
```yaml
train_loader:
    name: finetuning
    dataset:
        hf_name: tatsu-lab/alpaca
        split: train
        ...
```

If the dataset requires a [custom preprocessing function](#custom-data-preprocessing), such as in the example described above, use `preprocessing_fn` to tell the dataloader where it should import the function from.
```yaml
train_loader:
    name: finetuning
    dataset:
        hf_name: mosaiml/doge-facts
        preprocessing_fn: my_data.formatting:dogefacts_prep_fn
        split: train
        ...
```

### **2) Using a local dataset**

Let's say you have your finetuning dataset stored in local `jsonl` files.
Reference this in your YAML, such as the one in `yamls/mpt/finetune/1b_local_data_sft.yaml`
```yaml
train_loader:
    name: finetuning
    dataset:
        hf_name: my-local-dataset
        hf_kwargs:
            data_files:
                train: /path/to/train.jsonl
        preprocessing_fn: my.import.path:my_preprocessing_fn
        split: train
        ...
```
As before, if your local dataset already has the "prompt"/"response" format, you don't need to include `preprocessing_fn` since no preprocessing is needed.

### **3) Using an MDS-formatted (streaming) dataset -- locally or in an object store**

To enable streaming, you must first use the `convert_finetuning_dataset.py` script to convert a HuggingFace dataset into an [MDS-formatted dataset](https://github.com/mosaicml/streaming) (which you totally should -- they're amazing).

<!--pytest.mark.skip-->
```bash
python ../data_prep/convert_finetuning_dataset.py \
    --dataset tatsu-lab/alpaca \
    --splits train \
    --out_root s3://my-bucket/my-copy-alpaca
```

> **Note**
> Streaming datasets *must* follow the required "prompt"/"response" format, but you can preprocess during conversion by setting the `--preprocessor` argument.
> <!--pytest.mark.skip-->
> ```bash
> python ../data_prep/convert_finetuning_dataset.py \
>     --dataset mosaicml/doge-facts \
>     --preprocessor my_data.formatting:dogefacts_prep_fn \
>     --splits train \
>     --out_root s3://my-bucket/my-copy-doge-facts
>```

Once you have converted your HuggingFace dataset to a streaming dataset, just update your YAML like so:
```yaml
train_loader:
    name: finetuning
    dataset:
        remote: s3://my-bucket/my-copy-doge-facts
        local: /tmp/mds-cache/
        split: train
        ...
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

Given these two values, our code automatically adjusts the # of gradient accumulation steps based on the # of devices,
so you should be able to run the exact same YAML on 8 or 16 or 256 GPUs and get the same training results (within numerics).
This is nice because it means you can write device-count-agnostic training configs,
and not worry about OOM-ing or accidentally changing the optimization math.

In previous blogs ([1](https://www.mosaicml.com/blog/farewell-oom), [2](https://www.mosaicml.com/blog/billion-parameter-gpt-training-made-easy))
we also demonstrated Auto Grad Accum, which takes things a step further by letting Composer determine the `device_train_microbatch_size` on its own.
This makes our configs not only device-count-agnostic, but hardware-agnostic too!
You can try out this feature by setting `device_train_microbatch_size: auto`, but bear in mind that FSDP support is still in alpha mode
and may not always work with Auto Grad Accum (but we are working on it!).
