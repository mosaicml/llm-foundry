# Embedding models

_Q: What is a contrastive loss?_

The contrastive loss can be thought of as a loss that creates a high similarity score between two similar samples, and a low score between two very different samples. Formally, this can be achieved by using the cross-entropy loss for an N-way softmax classifier. Some motivation for the contrastive loss can be found in [this blogpost](https://ankeshanand.com/blog/2020/01/26/contrative-self-supervised-learning.html). This has become the dominant method for training embedding/retrieval methods.

_Q: How does the data need to be formatted?_

The data simply needs to come in "pairs." This can either be in the form of:
1. positive pairs such as "query: is snowboarding allowed at the alta ski resort?" "passage: the alta ski resort does not allow snowboarding," or
2. tuples of positive pairs with curated hard negative pairs.

In the first case, the InfoNCE Loss treats the rest of the samples in the batch as "soft negatives" (this scenario also is most successful with _very_ large global batch sizes on the order of 16-32k). In the second scenario, the InfoNCE uses the hard negatives in the denominator of the loss and much smaller global batch sizes (e.g. 32).

_Q: How do you get a vector embedding out of a decoder? I thought you could only do that with encoders?_

Before the final "logit" layer of the decoder, the tensor still has dimensions _batch size_ x _sequence length_ x _hidden dimension_. In order to get a single vector representation for a single sample, we can average along the sequence length dimension (i.e. average the vectors representing each token). Alternatively, you can append an `<|endoftext|>` token to the end of each sample and extract the vector for this token alone (this seems to work well for RepLlama).

The main additions are as follows:

* The class `ContrastiveModel(HuggingFaceModel)`, which implements the InfoNCE Loss in the `.loss()` function.
* A dataloader for contrastive pairs `build_pairs_dataloader()`. This can handle positive pairs formatted as `text_a` and `text_b`, or positive pairs with hard negatives formatted as `query`,`passage` and a list of `hard_negatives`.

## Example YAML

```yaml
variables:
  data_local: <your_dataset_location>
  data_remote:  # If blank, files must be present in data_local
  max_seq_len: 2048
  global_seed: 17

  # Run Name
  run_name:  # If left blank, will be read from env var $RUN_NAME

max_seq_len: ${variables.max_seq_len}
run_name: ${variables.run_name}

# Model
model:
  name: contrastive_lm
  init_device: meta
  d_model: 768
  n_heads: 12
  n_layers: 12
  expansion_ratio: 4
  max_seq_len: ${variables.max_seq_len}
  vocab_size: 50368
  attn_config:
    attn_impl: flash

# Tokenizer
tokenizer:
  name: EleutherAI/gpt-neox-20b
  kwargs:
    model_max_length: ${variables.max_seq_len}

# Dataloaders
train_loader:
  name: contrastive_pairs
  dataset:
    local: ${variables.data_local}
    split: null
    remote: ${variables.data_remote}
    shuffle: true
    max_seq_len: ${variables.max_seq_len}
    shuffle_seed: ${variables.global_seed}
    prepend_query: 'query: '
    prepend_passage: 'passage: '
    append_eos_token: true
  drop_last: true
  num_workers: 8
```
