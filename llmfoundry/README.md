
# Source code

LLMFoundry is a Python package for training, finetuning, evaluating, and serving large scale LLM models on distributed compute infrastructure using MosaicML's Composer with PyTorch

At a granular level, LLMFoundry is a library that consists of the following components:

* `llmfoundry.models.mpt.MPTModel` - a simple PyTorch GPT model, wrapped in `ComposerModel`, that can scale up to 70B+ parameters
* `llmfoundry.models.layers` - a collection of layers used in the MPTModel
* `llmfoundry.models.hf` - a collection of tools which enables training / finetuning huggingface models with `../scripts/train/train.py`
* `llmfoundry.data.text_data.StreamingTextDataset`- a [MosaicML streaming dataset](https://streaming.docs.mosaicml.com/en/stable/) that can be used with a vanilla PyTorch dataloader.
* `llmfoundry.data.finetuning.collator.Seq2SeqFinetuningCollator`- a dataloader for different finetuning tasks
* `llmfoundry.optim`- a collection of optimizers used for training LLMs (PyTorch and Composer optimizers are also compatible)
* `llmfoundry.utils.builders`- a collection of convenient string-to-object mappings used to create objects that get passed to the [Composer Trainer](https://docs.mosaicml.com/projects/composer/en/stable/api_reference/generated/composer.Trainer.html).
