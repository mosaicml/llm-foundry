# Data preparation

This folder contains scripts for converting text data from original sources (HF, JSON) to StreamingDataset format for consumption by our training scripts.


## Converting a pretraining dataset
Using the `convert_dataset_hf.py` script...

<!--pytest.mark.skip-->
```bash
# Convert C4 dataset to StreamingDataset format
python convert_dataset_hf.py \
  --dataset c4 --data_subset en \
  --out_root my-copy-c4 --splits train_small val_small \
  --concat_tokens 2048 --tokenizer EleutherAI/gpt-neox-20b --eos_text '<|endoftext|>'
```

Using the `convert_dataset_json.py` script...

<!--pytest.mark.skip-->
```bash
# Convert C4 dataset to StreamingDataset format
python convert_dataset_hf.py \
  --path ./example_data/arxiv.json \
  --out_root my-copy-arxiv --split train \
  --concat_tokens 2048 --tokenizer EleutherAI/gpt-neox-20b --eos_text '<|endoftext|>'
```

Where `--path` can be a single json file, or a folder containing json files, and split the intended split (hf defaults to train).

## Converting a finetuning dataset
Using the `convert_finetuning_dataset.py` script...
