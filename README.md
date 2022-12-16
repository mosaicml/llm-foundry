# Examples

Fast reference examples for training ML models with recipes. Designed to be easily forked and modified.
## ResNet-50 + ImageNet
<img src="https://assets-global.website-files.com/61fd4eb76a8d78bc0676b47d/62a12d1e4eb9b83915be37a6_r50_overall_pareto.png" alt="drawing" width="500"/>

*Figure 1: Comparison of MosaicML recipes against other results, all measured on 8x A100s on MosaicML Cloud.*


Train the MosaicML ResNet, the fastest ResNet50 implementation that yields a :sparkles: 7x :sparkles: faster time-to-train compared to a strong baseline. See our [blog](https://www.mosaicml.com/blog/mosaic-resnet) for more details and recipes. Our recipes were also demonstrated at [MLPerf](https://www.mosaicml.com/blog/mlperf-2022), a cross industry ML benchmark.

:rocket: Get started with the code [here](./resnet/).

## DeepLabV3 + ADE20k
<img src="https://assets-global.website-files.com/61fd4eb76a8d78bc0676b47d/637512d993030157b04ad4f8_Frame%2010%20(1).png" alt="drawing" width="500"/>

Train the MosaicML DeepLabV3 that yields a :sparkles:5x:sparkles: faster time-to-train compared to a strong baseline. See our [blog](https://www.mosaicml.com/blog/mosaic-image-segmentation) for more details and recipes.

:rocket: Get started with the code [here](./deeplab/).

## Large Language Models (LLMs)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./llm/assets/loss-curve-dark.png">
  <img alt="Training curves for various LLM sizes." src="./llm/assets/loss-curve-light.png" width=500px>
</picture>

A simple yet feature complete implementation of GPT, that scales to 70B parameters while maintaining high performance on GPU clusters. Flexible code, written with vanilla PyTorch, that uses [PyTorch FSDP](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/) and some recent efficiency improvements.

:rocket: Get started with the code [here](./llm/).
