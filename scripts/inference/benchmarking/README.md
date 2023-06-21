# Inference Benchmarking

This folder provides scripts for benchmarking the inference performance of deep learning models. Currently, we support benchmarking with Deepspeed and Huggingface generate.

## Scripts

The repository includes the benchmark.py script, along with associated `.yaml files,` to run benchmarking. The script takes a `.yaml` file as input and outputs the latency (in seconds) and tokens per second for each run. We average over `num_batches=5`, which is defined in the `.yaml` file. Additionally, we iterate over various `batch_sizes`, `input_lengths`, and `output_lengths` to produce varying throughput metrics.

## Usage

To use the `benchmark.py` script, you need to provide a `.yaml` file that specifies the model configuration and other parameters such as the path to the model checkpoint and the input data. You can modify the default `.yaml` files provided in the repository or create your own `.yaml` file.

To run the benchmarking script, use the following command:

`python benchmark.py yamls/1b.yaml`

To run the scripts on [The MosaicML platform](https://www.mosaicml.com/blog/mosaicml-cloud-demo) we've also included scripts and associated `.yaml files` in the `mcli` folder.

## LLM Inference Overview and Results

### How do I reason about inference in LLMs?

LLM inference consists of two stages: _prefill_ and _decode_. It's important to understand the difference between these stages as latency scales differently for these two stages.

During _prefill_, the model processes the input tokens/prompt/context. This is done in a single forward pass, making this stage fast, with excellent use of GPU hardware (ie. high Model Flop Utilization aka [MFU](https://github.com/mosaicml/llm-foundry/tree/main/scripts/train/benchmarking#mfu)). Typically, if people talk about LLM inference being slow, this is _not_ the stage that they are referring to. As part of the prefill stage, the KV cache is populated, which stores the calculated values for the keys and queries for the model across the model layers, avoiding re-processing these tokens during the rest of generation.
To set up the KV cache for benchmarking, set `use_cache: true` in the `.yaml` file.

During _decode_, the model generates output tokens one at a time, ie. autoregressively. This requires making N forward passes of the model for N tokens. This stage is slow and inefficient, because it requires moving gigabytes of model weights and pre-filled values for every single forward pass. Here, latency scales linearly with the number of output tokens.

