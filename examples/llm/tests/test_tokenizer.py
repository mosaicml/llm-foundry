# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

from omegaconf import OmegaConf as om

from examples.llm.src.tokenizer import TOKENIZER_REGISTRY


def get_config(conf_path='yamls/mosaic_gpt/125m.yaml'):
    with open(conf_path) as f:
        test_cfg = om.load(f)
    return test_cfg


def test_load_tokenizer():
    test_cfg = get_config(conf_path='yamls/mosaic_gpt/125m.yaml')
    truncation = True
    padding = 'max_length'

    tokenizer = TOKENIZER_REGISTRY[test_cfg.tokenizer.type](
        **test_cfg.tokenizer.args)
    assert tokenizer.tokenizer.vocab_size == 50257
    assert tokenizer.tokenizer.name_or_path == 'gpt2'

    in_str = 'hello\n\nhello'
    out_token_key = [31373, 198, 198, 31373]

    # test explicitly call tokenizer
    out = tokenizer.tokenizer.encode(in_str)
    assert out == out_token_key

    # tokenizer __call__
    out = tokenizer.tokenizer(in_str)['input_ids']
    assert out == out_token_key

    # tokenizer  __call__ with kwargs
    padded_tokenize = tokenizer.tokenizer(
        in_str,
        truncation=truncation,
        padding=padding,
        max_length=tokenizer.max_seq_len)['input_ids']
    out_pad_tokens = out_token_key + [50256] * (tokenizer.max_seq_len - 4)
    assert padded_tokenize == out_pad_tokens

    # wrapper class __call__
    out = tokenizer(in_str)['input_ids']
    assert out == out_token_key

    # wrapper class __call__ with kwargs
    padded_tokenize = tokenizer(in_str,
                                truncation=truncation,
                                padding=padding,
                                max_length=tokenizer.max_seq_len)['input_ids']
    assert padded_tokenize == out_pad_tokens

    # check attn mask
    attention_mask = tokenizer(
        in_str,
        truncation=truncation,
        padding=padding,
        max_length=tokenizer.max_seq_len)['attention_mask']
    attn_mask_key = [1, 1, 1, 1] + [0] * (tokenizer.max_seq_len - 4)
    assert attention_mask == attn_mask_key
