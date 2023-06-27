set -x
# Convert the model to HuggingFace format
python convert_composer_to_hf.py \
  --composer_path ../train/output/mpt-125m/ep3-ba1000-rank0.pt \
  --hf_output_path mpt-125m-hf \
  --output_precision bf16
