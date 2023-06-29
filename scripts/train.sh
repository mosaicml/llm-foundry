
# Train an MPT-125m model for 10 batches
composer train/train.py \
  train/yamls/pretrain/mpt-125m-stack.yaml \
  eval_loader.dataset.split=val_small \
  max_duration=1000ba \
  eval_interval=500ba
