# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

from omegaconf import OmegaConf as om
from transformers import AutoTokenizer


def get_config(conf_path='yamls/mosaic_gpt/125m.yaml'):
    with open(conf_path) as f:
        test_cfg = om.load(f)
    return test_cfg


def test_load_tokenizer():
    test_cfg = get_config(conf_path='yamls/mosaic_gpt/125m.yaml')
    truncation = True
    padding = 'max_length'

    resolved_om_tokenizer_config = om.to_container(test_cfg.tokenizer,
                                                   resolve=True)
    tokenizer_kwargs = resolved_om_tokenizer_config.get(  # type: ignore
        'kwargs', {})
    tokenizer_name = resolved_om_tokenizer_config['name']  # type: ignore
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,
                                              **tokenizer_kwargs)
    tokenizer.pad_token = tokenizer.eos_token
    assert tokenizer.vocab_size == 50257
    assert tokenizer.name_or_path == 'gpt2'

    in_str = 'hello\n\nhello'
    out_token_key = [31373, 198, 198, 31373]

    # test explicitly call tokenizer
    out = tokenizer.encode(in_str)
    assert out == out_token_key

    # tokenizer __call__
    out = tokenizer(in_str)['input_ids']
    assert out == out_token_key

    # tokenizer  __call__ with kwargs
    padded_tokenize = tokenizer(
        in_str,
        truncation=truncation,
        padding=padding,
        max_length=tokenizer.model_max_length)['input_ids']
    out_pad_tokens = out_token_key + [50256] * (tokenizer.model_max_length - 4)
    assert padded_tokenize == out_pad_tokens

    # wrapper class __call__
    out = tokenizer(in_str)['input_ids']
    assert out == out_token_key

    # wrapper class __call__ with kwargs
    padded_tokenize = tokenizer(
        in_str,
        truncation=truncation,
        padding=padding,
        max_length=tokenizer.model_max_length)['input_ids']
    assert padded_tokenize == out_pad_tokens

    # check attn mask
    attention_mask = tokenizer(
        in_str,
        truncation=truncation,
        padding=padding,
        max_length=tokenizer.model_max_length)['attention_mask']
    attn_mask_key = [1, 1, 1, 1] + [0] * (tokenizer.model_max_length - 4)
    assert attention_mask == attn_mask_key
