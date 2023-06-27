
set -x
# Convert C4 dataset to StreamingDataset format
python3 data_prep/convert_dataset_hf.py \
  --dataset /dataset/home/liaoxingyu/datasets/c4 --data_subset suben \
  --out_root /dataset/home/liaoxingyu/datasets/my-copy-c4 --splits train_small val_small \
  --concat_tokens 2048 --tokenizer EleutherAI/gpt-neox-20b --eos_text '<|endoftext|>'
