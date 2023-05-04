
# Source code
This folder contains all the packaged source code for LLM Foundry. In this folder you'll find:
* `llmfoundry/models/mpt/` - a simple PyTorch GPT model, wrapped in `ComposerModel`, that can scale up to 70B+ parameters
* `llmfoundry/utils/builders.py`- A collection of convenient string-to-object mappings used to create objects that get passed to `Trainer`.
* `llmfoundry/data/text_data.py`- a [MosaicML streaming dataset](https://streaming.docs.mosaicml.com/en/stable/) that can be used with a vanilla PyTorch dataloader.
