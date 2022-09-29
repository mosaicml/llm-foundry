# benchmarks

Fast reference benchmarks for training ML models with recipes. Designed to be easily forked and modified.

## ResNet-50
<img src="https://assets-global.website-files.com/61fd4eb76a8d78bc0676b47d/62a12d1e4eb9b83915be37a6_r50_overall_pareto.png" alt="drawing" width="500"/>

*Figure 1: Comparison of MosaicML recipes against other results, all measured on 8x A100s on MosaicML Cloud.*


Train the MosaicML ResNet, the fastest ResNet50 implementation that yields a :sparkles: 7x :sparkles: faster time-to-train compared to a strong baseline. See our [blog](https://www.mosaicml.com/blog/mosaic-resnet) for more details and recipes. Our recipes were also demonstrated at [MLPerf](https://www.mosaicml.com/blog/mlperf-2022), a cross industry ML benchmark.

:rocket: Get started with the code [here](./resnet/).

## GPT-3

Simple yet feature complete implementation of GPT-3, that scales to 175B parameters while maintaining similar GPU utilization as other approaches. Flexible code, written in vanilla pytorch, that uses PyTorch [FSDP](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/) and some recent efficiency improvements.

:rocket: Get started with the code [here](./llm/).

