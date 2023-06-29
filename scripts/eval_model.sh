set -x

composer eval/eval.py \
  eval/yamls/hf_eval.yaml \
  model_name_or_path=/dataset/home/liaoxingyu/Llama-X/output/alpaca_v1_code_evol_v1-starcoderplus-15b-fp16-zero_dp-plr2e-5-mlr0-mbsz16-gbsz256-ctxlen2048-tokn90k_piece-ep3-wmup30/checkpoint-100
