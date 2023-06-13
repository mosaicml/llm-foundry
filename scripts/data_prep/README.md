# Data preparation

This folder contains scripts for converting text data from original sources (HF, JSON) to [StreamingDataset](https://github.com/mosaicml/streaming) format for consumption by our training scripts.


## Converting a pretraining dataset
Using the `convert_dataset_hf.py` script...

<!--pytest.mark.skip-->
```bash
# Convert C4 dataset to StreamingDataset format
python convert_dataset_hf.py \
  --dataset c4 --data_subset en \
  --out_root my-copy-c4 --splits train_small val_small \
  --concat_tokens 2048 --tokenizer EleutherAI/gpt-neox-20b --eos_text '<|endoftext|>' \
  --compression zstd
```

Using the `convert_dataset_json.py` script...

<!--pytest.mark.skip-->
```bash
# Convert json dataset to StreamingDataset format
python convert_dataset_json.py \
  --path ./example_data/arxiv.jsonl \
  --out_root my-copy-arxiv --split train \
  --concat_tokens 2048 --tokenizer EleutherAI/gpt-neox-20b --eos_text '<|endoftext|>' \
  --compression zstd
```

Where `--path` can be a single json file, or a folder containing json files, and split the intended split (hf defaults to train).

## Converting a finetuning dataset
Using the `convert_finetuning_dataset.py` script you can run a command such as:
<!--pytest.mark.skip-->
```bash
python convert_finetuning_dataset.py --dataset "Muennighoff/P3" \
--splits "train" "validation" \
--preprocessor "llmfoundry.data.finetuning.tasks:p3_preprocessing_function"\
 --out_root "/path/to/your/output_directory"
```

This example assumes:

- You are running the script in the terminal or command prompt where `python` command is recognized.
- `"Muennighoff/P3"` is the dataset you want to convert. Substitute "Muennighoff/P3" with the name or path of your dataset.
- `train` and `validation` are the splits of the dataset to convert.
- `llmfoundry.data.finetuning.tasks:p3_preprocessing_function` is a string that provides the name or import path of the function used to preprocess the dataset. Substitute it with your actual preprocessor. See [tasks](https://github.com/mosaicml/llm-foundry/blob/main/llmfoundry/data/finetuning/tasks.py) for available functions and examples.
- `s3://<bucket>/muennighoff-p3` is the root path of your output directory where MDS shards will be stored. Replace this with the actual path to your output directory.

Please note that you need to fill in actual values for "your_preprocessing_function" and "/path/to/your/output_directory" in the command above for it to work correctly.

Also, if you want to keep a local copy of the output when `out_root` is remote, you can use the `--local` argument:
<!--pytest.mark.skip-->
```bash
python convert_finetuning_dataset.py --dataset "squad" --splits "train" "validation" --preprocessor "your_preprocessing_function" --out_root "s3://your_bucket/output_directory" --local "/path/to/local/directory"
```

Remember that all these command line arguments should be filled with your actual dataset name/path, preprocessing function, and output directories.
