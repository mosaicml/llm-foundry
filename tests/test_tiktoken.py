# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import pathlib

import pytest
import transformers

from llmfoundry import TiktokenTokenizerWrapper
from tests.horrible_strings import HORRIBLE_STRINGS

TEST_STRINGS = [
    'Hello world!', 'def hello_world(input: str):\n    print(input)',
    '0000000000000000000000000000',
    '19234324 asas sf 119aASDFM AW3RAW-AF;;9900', '\n\n\n\nhello\n\t,'
]

TEST_STRINGS += HORRIBLE_STRINGS


@pytest.mark.parametrize('model_name',
                         ['gpt-4', 'gpt-3.5-turbo', 'text-davinci-003'])
def test_tiktoken(model_name: str, tmp_path: pathlib.Path):
    tiktoken = pytest.importorskip('tiktoken')

    # Construction
    wrapped_tokenizer = TiktokenTokenizerWrapper(model_name='gpt-4')
    original_tokenizer = tiktoken.encoding_for_model('gpt-4')

    # Repr works
    _ = wrapped_tokenizer.__repr__()

    # Save and load
    wrapped_tokenizer.save_pretrained(tmp_path)
    reloaded_wrapped_tokenizer = transformers.AutoTokenizer.from_pretrained(
        tmp_path, trust_remote_code=True)

    didnt_match = []
    # Simple tokenization test
    for string in TEST_STRINGS:
        wrapped_output = wrapped_tokenizer(string)
        original_output = original_tokenizer.encode(string)
        reloaded_wrapped_output = reloaded_wrapped_tokenizer(string)
        assert wrapped_output['input_ids'] == original_output
        assert set(wrapped_output.keys()) == {'input_ids', 'attention_mask'}
        assert reloaded_wrapped_output == wrapped_output

    # Round trip
    for string in TEST_STRINGS:
        wrapped_output = wrapped_tokenizer.decode(
            wrapped_tokenizer(string)['input_ids'])
        original_output = original_tokenizer.decode(
            original_tokenizer.encode(string))
        reloaded_wrapped_output = reloaded_wrapped_tokenizer.decode(
            reloaded_wrapped_tokenizer(string)['input_ids'])
        assert wrapped_output == string
        assert original_output == string
        assert reloaded_wrapped_output == string

    # Batched tokenization
    wrapped_output = wrapped_tokenizer(
        ['Hello world!', 'Hello world but longer!'])
    original_output = original_tokenizer.encode_batch(
        ['Hello world!', 'Hello world but longer!'])
    reloaded_wrapped_output = reloaded_wrapped_tokenizer(
        ['Hello world!', 'Hello world but longer!'])
    assert wrapped_output['input_ids'] == original_output
    assert set(wrapped_output.keys()) == {'input_ids', 'attention_mask'}
    assert reloaded_wrapped_output == wrapped_output

    # With padding
    wrapped_tokenizer.pad_token_id = wrapped_tokenizer.eos_token_id
    reloaded_wrapped_tokenizer.pad_token_id = reloaded_wrapped_tokenizer.eos_token_id
    wrapped_output = wrapped_tokenizer(
        ['Hello world!', 'Hello world but longer!'], padding=True)
    original_output = original_tokenizer.encode_batch(
        ['Hello world!', 'Hello world but longer!'])
    reloaded_wrapped_output = reloaded_wrapped_tokenizer(
        ['Hello world!', 'Hello world but longer!'], padding=True)
    for wrapped1, attn_mask, original1 in zip(wrapped_output['input_ids'],
                                              wrapped_output['attention_mask'],
                                              original_output):
        original_length = len(original1)
        assert wrapped1[:original_length] == original1
        assert sum(attn_mask) == original_length

    assert set(wrapped_output.keys()) == {'input_ids', 'attention_mask'}
    assert reloaded_wrapped_output == wrapped_output

    # Get vocab
    wrapped_vocab = wrapped_tokenizer.get_vocab()
    reloaded_wrapped_vocab = reloaded_wrapped_tokenizer.get_vocab()
    assert wrapped_vocab == reloaded_wrapped_vocab

    didnt_match = []
    for key, value in wrapped_vocab.items():
        if original_tokenizer.encode(key, allowed_special='all') == [value]:
            continue
        else:
            didnt_match.append(
                (key, original_tokenizer.encode(key,
                                                allowed_special='all'), value))

    # Decode is lossy because some bytes are not representable in utf-8
    # see https://github.com/openai/tiktoken/blob/39f29cecdb6fc38d9a3434e5dd15e4de58cf3c80/tiktoken/core.py#L245-L247
    assert len(didnt_match) == 77
