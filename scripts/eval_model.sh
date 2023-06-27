set -x

# Evaluate the model on Winograd
python eval/eval.py \
  eval/yamls/hf_eval.yaml \
  icl_tasks=eval/yamls/winograd.yaml \
  model_name_or_path=inference/mpt-125m-hf
