# How to finetune a new Hugging Face model
Using the [DBRX yaml](https://github.com/mosaicml/llm-foundry/blob/main/scripts/train/yamls/finetune/dbrx-full-ft.yaml) as a template, change the following fields for the new model:
- `max_seq_len`
- `model/pretrained_model_name_or_path`
- `tokenizer/name`
Open a GitHub issue if these steps don't work. Most new causal LMs should work out of the box.
