
set -x

composer -n 8 eval/eval.py \
  eval/yamls/hf_eval.yaml \
  model_name_or_path=/dataset/home/liaoxingyu/models/starcoderplus \
  precision=amp_fp16
