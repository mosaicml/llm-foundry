# Inference Benchmarking

This folder provides scripts for benchmarking the inference performance of deep learning models. Currently, we support benchmarking with Deepspeed and Huggingface generate.

We also have a dedicated inference service which uses an optimized stack, leading to improved performance compared to the scripts in this repository. The benchmarks in this README show performance of LLMs performing generations on target hardware, but _do not_ have the same functionality and performance as our full inference service for real-world deployments.

To learn more about our inference service, visit [mosaicml.com/inference](https://www.mosaicml.com/inference).

## Scripts

The repository includes the benchmark.py script, along with associated `.yaml` files to run benchmarking. The script takes a `.yaml` file as input and outputs the latency (in seconds) and tokens per second for each run. We average over `num_batches=5`, which is defined in the `.yaml` file. Additionally, we iterate over various `batch_sizes`, `input_lengths`, and `output_lengths` to produce varying throughput metrics.

## Usage

To use the `benchmark.py` script, you need to provide a `.yaml` file that specifies the model configuration and other parameters such as the path to the model checkpoint and the input data. You can modify the default `.yaml` files provided in the repository or create your own `.yaml` file.

To run the benchmarking script, use the following command:

`python benchmark.py yamls/1b.yaml`

To run the scripts on [The MosaicML platform](https://www.mosaicml.com/blog/mosaicml-cloud-demo) we've also included scripts and associated `.yaml` files in the `mcli` folder.

## LLM Inference Overview and Results

### Background

LLM inference consists of two stages: _prefill_ and _decode_. It's important to understand the difference between these stages as latency scales differently in each stage.

During _prefill_, the model processes the input tokens/prompt/context. This is done in a single forward pass, making this stage fast, with excellent use of GPU hardware (ie. high Model Flop Utilization aka [MFU](https://github.com/mosaicml/llm-foundry/tree/main/scripts/train/benchmarking#mfu)). Typically, if people talk about LLM inference being slow, this is _not_ the stage that they are referring to.

During _decode_, the model generates output tokens one at a time, i.e. autoregressively. This requires making N forward passes of the model for N tokens. This stage is slow and inefficient, because it requires moving gigabytes of model weights and pre-filled values for every single forward pass. Here, latency scales (mostly) linearly with the number of output tokens. Why mostly linear? When generating long sequences, the quadratic memory and compute complexity of the attention operation become more prominant.

##### KV cache

One of the important elements of LLM inference is a Key-Value (KV) cache. As part of the prefill stage, the key and value tensors are computed and cached for each token in the prompt across the model layers. This is helpful because the initial prompt does not change during inference, which allows the model layers to simply re-use the cached values, rather than spending compute on re-computing the tokens for the context. \
To enable KV cache for benchmarking, set `use_cache: true` in the `.yaml` file.


### Results
Now that we have the necessary background to reason about inference, we can move onto the fun stuff, ie. real-world benchmark results.

### I want to run generation with MPT-[X], how long does it take to process different size inputs?

#### Setup
We use a single A100 80GB for inference, running with precision `bf16` and the `torch` implementation of attention. We include the highest possible batch size that is able to produce 1000 output tokens without running out of GPU memory.

Here we show how latency changes for a given input prompt length, while varying batch size and output length.
This gives a rule of thumb of how fast you can expect MPT to be based on different generation parameters.

For the output length sweep, we use output lengths of 1, 11, 101, 1001 instead of 1, 10, 100, 1000 as this makes it easier to compute the overhead of the first output token in the prefill stage.
#### Technical Details
Hardware Setup:
- 1 x NVIDIA A100 80GB


Benchmark Setup:
- Input Length: [128, 2048]
- Output Length: [1, 11, 101, 1001]
#### Short Inputs (128 input tokens) on MPT-7B
![assets](assets/Latency-for-MPT-7B,-n_input_tok=128.svg)
#### Long Inputs (2048 input tokens) on MPT-7B
![assets](assets/Latency-for-MPT-7B,-n_input_tok=2048.svg)
#### Short Inputs (128 input tokens) on MPT-30B
![assets](assets/Latency-for-MPT-30B,-n_input_tok=128.svg)
#### Long Inputs (2048 input tokens) on MPT-30B
![assets](assets/Latency-for-MPT-30B,-n_input_tok=2048.svg)

Our real-world results match the theory! The latency grows nearly linearly with the number of output tokens, which is the _decode_ stage time. For large batch sizes and output lengths, the latency looks more quadratic, which shows the quadratic compute complexity of the attention operation.

For longer input lengths and batch sizes, the _prefill_ stage starts to become more important, given that the model has to process a lot of input tokens in the forward pass.
Despite the _prefill_ stage being highly efficient, the model still needs to perform a lot of computation in one forward pass, which results in the higher latency when increasing batch size and input length.

In general, latency tends to be dominated by long sequences rather than larger batch sizes, so whenever possible, it's valuable to increase batch sizes as this allows for processing a lot more inputs without (much) additional cost.

### Which hardware system should I use?

While the previous section focused on providing a guideline on how latency changes for different generation parameters, folks who are deploying inference systems often care about satisfying certain latency constraints.
That is to say, there is a certain target latency by which a user _must_ receive the model's output.

Hence, an effective way to compare different inference systems is by plotting their latency vs. throughput, which says how many tokens/second the model can serve given a specific latency constraint.

Typically, the latency is a hard constraint, while throughput can be increased by using more GPUs/alternative hardware setups. However, batch size is usually not a constraint, and so it can be chosen (statically or dynamically) depending on the latency/throughput budget.

A model or setup whose curve is strictly above another's will be able to achieve higher throughput for a given target latency.

To generate these curves, we vary the batch size for a fixed input length (512) and fixed output length (64), and calculate the associated latencies and throughputs. The batch sizes swept are 1, 2, 4, 8, 16, 32, 64, unless the GPU ran out of memory, in which case that point is not shown.

#### Technical Details
Hardware Setup:
- 1, 2, 4 x NVIDIA A100 80GB
- 1, 2, 4 x NVIDIA A100 40GB


Benchmark Setup:
- Input Length: 512
- Output Length: 64
- Batch Sizes: 1, 2, 4, 8, 16, 32, 64

#### Different HW setups for MPT-7B
![assets](assets/Latency-vs.-Throughput,-MPT-7B-(n_input_tok=512,-n_output_tok=64).svg)
#### Different HW setups for MPT-30B
![assets](assets/Latency-vs.-Throughput,-MPT-30B-(n_input_tok=512,-n_output_tok=64).svg)

These plots show how using multiple A100s with 40GB and 80GB cards and Tensor Parallelism (TP) using `deepspeed` compare with single GPUs.
Note that there are OOMs at larger batch sizes for some of the cards, so the plots present the achievable latency/throughputs for a given GPU setup.

As expected, using more GPUs for inference increases the throughput. TP also allows MPT-30B to be supported across multiple A100 40G, which is not possible on a single 40G card.
Still, the communication overhead becomes significant with more GPUs, which is why MPT-7B shows comparable results when using both 4 x A100 40GB and 4 x A100 80GB GPUs.

### Comparing MPT with other open-source models

#### Technical Details
Hardware Setup:
- 1 x NVIDIA A100 80GB


Models benchmarked:
- MPT-[7, 30]B
- LLAMA-[7, 13, 30]B
- Falcon-7B


Benchmark Setup:
- Input Length: 512
- Output Length: 64
- Batch Sizes: 1, 2, 4, 8, 16, 32, 64

![assets](assets/Latency-vs.-Throughput-(n_input_tok=512,-n_output_tok=64).svg)

Here, we perform a similar benchmark to the previous section, but compare inference performance for different open-source models.
The benchmark script supports calling models directly from huggingface (using `hf.generate`), which is done to keep the comparison amongst the models fair.
The analysis is done on a single A100 80GB GPU, with input length 512, and output length 64, while varying the batch size. As in previous sections, the batch sizes swept are 1, 2, 4, 8, 16, 32, 64, unless the GPU ran out of memory, in which case that point is not shown.

As seen here, both MPT-7B and MPT-30B are among the fastest for inference in the open-source community, with MPT-30B being faster than the respective LLAMA-30B model.
Among the 7B models, Falcon-7B tends to have higher througput at higher latencies than MPT-7B, though MPT-7B has higher throughput at lower latencies.
Previously, we found that Falcon-7b was significantly slower than both MPT-7B and LLAMA-7B. This slow speed was due to the KV-cache not being used properly during generation, however this appears to be [fixed](https://huggingface.co/tiiuae/falcon-7b/tree/main) as of July 13, 2022.
