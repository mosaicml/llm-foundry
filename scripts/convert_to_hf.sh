set -x
# Convert the model to HuggingFace format
python inference/convert_composer_to_hf.py \
  --composer_path output/mpt-125m-stack/latest-rank0.pt \
  --hf_output_path mpt-125m-hf-stack \
  --output_precision bf16
