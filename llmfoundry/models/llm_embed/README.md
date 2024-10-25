# MPT-Embed

_Q: What is a contrastive loss?_

The contrastive loss can be thought of as a loss that creates a high similarity score between two similar samples, and a low score between two very different samples. Formally, this can be achieved by using the cross-entropy loss for an N-way softmax classifier. Some motivation for the contrastive loss can be found in [this blogpost](https://ankeshanand.com/blog/2020/01/26/contrative-self-supervised-learning.html). This has become the dominant method for training embedding/retrieval methods.

_Q: How does the data need to be formatted?_

The data simply needs to come in "pairs." This can either be in the form of:
1. positive pairs such as "query: is snowboarding allowed at the alta ski resort?" "passage: the alta ski resort does not allow snowboarding," or
2. tuples of positive pairs with curated hard negative pairs.

In the first case, the InfoNCE Loss treats the rest of the samples in the batch as "soft negatives" (this scenario also is most successful with _very_ large global batch sizes on the order of 16-32k). In the second scenario, the InfoNCE uses the hard negatives in the denominator of the loss and much smaller global batch sizes (e.g. 32).

_Q: How do you get a vector embedding out of a decoder? I thought you could only do that with encoders?_

Before the final "logit" layer of the decoder, the tensor still has dimensions _batch size_ x _sequence length_ x _hidden dimension_. In order to get a single vector representation for single sample, we can simple average along the sequence length dimension (i.e. average the vectors representing each token). Alternatively, you can append an `<|endoftext|>` token to the end of each sample and extract the vector for this token alone (this seems to work well for RepLlama).

The main additions are as follows:

* The class `ContrastiveModel(HuggingFaceModel)`, which implements the InfoNCE Loss in the `.loss()` function. `mpt_embed` is treated as a separate model folder from `mpt`.
* A naive dataloader for contrastive pairs `build_pairs_dataloader()`. This can handle positive pairs formatted as `text_a` and `text_b`, or positive pairs with hard negatives formatted as `query`,`passage` and a list of `hard_negatives`.

## Example YAML

```yaml
name: <your run name>
image: <the image you wish to use>
scheduling:
  priority: low
  preemptible: true
compute:
  gpus: 16
  cluster: r7z22
  gpu_type: a100_40gb
parameters:
  seed: ${global_seed}
  model:
    name: contrastive_lm
    d_model: 2048
    n_heads: 16
    no_bias: true
    n_layers: 20
    norm_type: low_precision_layernorm
    ffn_config:
      ffn_type: mb_dmoe
      mlp_impl: grouped
      mlp_type: glu
      moe_top_k: 4
      ffn_act_fn:
        name: silu
      moe_jitter_eps: 0.01
      moe_world_size: 8
      ffn_hidden_size: 3584
      moe_lbl_in_fp32: false
      moe_loss_weight: 0.05
      moe_num_experts: 16
      memory_optimized_mlp: true
      moe_weight_parallelism: false
      quantize_inputs_num_bits: 8
      quantize_scatter_num_bits: 8
      uniform_expert_assignment: false
      moe_normalize_expert_weights: 1
      quantize_rematerialize_num_bits: -1
    vocab_size: 100352
    attn_config:
      rope: true
      alibi: false
      clip_qkv: 8
      attn_impl: flash
      attn_type: grouped_query_attention
      kv_n_heads: 8
      rope_theta: 500000
      attn_uses_sequence_id: false
    init_device: meta
    max_seq_len: ${max_seq_len}
    contrastive_config:
      temperature: 0.01
      normalize_output: true
      vector_representation: avg
    fuse_norm_attn_norm: true
    tie_word_embeddings: false
  loggers:
    wandb:
      entity: mosaic-ml
      project: mpt2-1e22-msmarco
    mlflow:
      tags:
        group: ${run_name}
        dataset: ms-marco
      tracking_uri: databricks
      experiment_name: <your experiment name>
  run_name: null
  callbacks:
    lr_monitor: {}
    speed_monitor:
      window_size: 10
    memory_monitor: {}
    runtime_estimator: {}
    mono_checkpoint_saver:
      save_folder: <your save folder>
      batch_interval: 1
  load_path: <you load path>
  optimizer:
    lr: 1.0e-07
    eps: 1.0e-06
    name: decoupled_adamw
    betas:
    - 0.9
    - 0.95
    weight_decay: 1.0e-05
  precision: amp_bf16
  scheduler:
    name: cosine_with_warmup
    alpha_f: 0.02
    t_warmup: 0.06dur
  tokenizer:
    name: tiktoken
    kwargs:
      model_name: gpt-4
  algorithms:
    gradient_clipping:
      clipping_type: norm
      clipping_threshold: 1
  data_local: /tmp/mds-cache/mds-ms_marco/
  eval_first: false
  data_remote: <your dataset>
  fsdp_config:
    verbose: true
    data_parallel_shard_degree: 16
    mixed_precision: PURE
    state_dict_type: sharded
    use_orig_params: true
    forward_prefetch: true
    backward_prefetch: BACKWARD_PRE
    limit_all_gathers: true
    sharding_strategy: FULL_SHARD
    activation_cpu_offload: true
    forward_prefetch_limit: 1
    backward_prefetch_limit: 2
    activation_checkpointing: true
    activation_checkpointing_reentrant: false
  global_seed: 17
  max_seq_len: 128
  save_folder: <your save folder>
  dist_timeout: 6000
  max_duration: 1ba
  progress_bar: false
  train_loader:
    name: contrastive_pairs
    dataset:
      local: ${data_local}
      split: null
      remote: ${data_remote}
      shuffle: true
      max_seq_len: ${max_seq_len}
      shuffle_seed: ${global_seed}
      prepend_query: 'query: '
      prepend_passage: 'passage: '
      append_eos_token: true
    drop_last: true
    num_workers: 8
  eval_interval: 1
  save_interval: 100ba
  log_to_console: true
  load_weights_only: true
  console_log_interval: 1ba
  device_eval_batch_size: 2
  eval_subset_num_batches: -1
  global_train_batch_size: 128
  device_train_microbatch_size: 2
  save_num_checkpoints_to_keep: 1
integrations:
- integration_type: git_repo
  git_repo: mosaicml/llm-foundry
  ssh_clone: true
  git_branch: main
  pip_install: .[gpu]
env_variables:
  NCCL_DEBUG: INFO
command: |-
  cd llm-foundry/scripts
  composer train/train.py /mnt/config/parameters.yaml
```
