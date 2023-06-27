
# Train an MPT-125m model for 10 batches
composer train.py \
  yamls/pretrain/mpt-125m.yaml \
  data_local=../data_prep/my-copy-c4 \
  train_loader.dataset.split=train_small \
  eval_loader.dataset.split=val_small \
  max_duration=1000ba \
  eval_interval=0 \
  save_folder=output/mpt-125m
