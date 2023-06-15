# Inference Benchmarking

This folder provides scripts for benchmarking the inference performance of deep learning models. Currently, we support benchmarking with Deepspeed and Huggingface generate.

## Scripts

The repository includes the benchmark.py script, along with associated `.yaml files,` to run benchmarking. The script takes a `.yaml` file as input and outputs the latency (in seconds) and tokens per second for each run. We average over `num_batches=5`, which is defined in the `.yaml` file. Additionally, we iterate over various `batch_sizes`, `input_lengths`, and `output_lengths` to produce varying throughput metrics.

## Usage

To use the `benchmark.py` script, you need to provide a `.yaml` file that specifies the model configuration and other parameters such as the path to the model checkpoint and the input data. You can modify the default `.yaml` files provided in the repository or create your own `.yaml` file.

To run the benchmarking script, use the following command:

`python benchmark.py yamls/1b.yaml`

To run the scripts on [The MosaicML platform](https://www.mosaicml.com/blog/mosaicml-cloud-demo) we've also included scripts and associated `.yaml files` in the `mcli` folder.
