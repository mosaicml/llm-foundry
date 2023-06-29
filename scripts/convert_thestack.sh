
set -x
# Convert TheStackDedup dataset to StreamingDataset format
python3 data_prep/convert_dataset_hf.py \
  --dataset /dataset_goosefs/cos_shanghai_1/ready_datasets/code/the_stack_dedup \
  --data_subset v1/data/javascript \
  --out_root /dataset/home/liaoxingyu/datasets/my-copy-the-stack-javascript-v1 --splits train \
  --concat_tokens 2048 --tokenizer /dataset/home/liaoxingyu/models/starcoderbase --eos_text '<|endoftext|>'
