# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import pathlib
from typing import TYPE_CHECKING, Optional, Tuple

import pytest
import transformers

from llmfoundry import TiktokenTokenizerWrapper
from tests.horrible_strings import HORRIBLE_STRINGS
from tests.test_hf_conversion_script import check_hf_tokenizer_equivalence

if TYPE_CHECKING:
    from tiktoken.core import Encoding

TEST_STRINGS = [
    'Hello world!', 'def hello_world(input: str):\n    print(input)',
    '0000000000000000000000000000',
    '19234324 asas sf 119aASDFM AW3RAW-AF;;9900', '\n\n\n\nhello\n\t',
    '            hello\n\t\\\\     goodbye!?*#&@!)     ',
    'This is just a normal sentence. And here is another one!',
    'hello<|endoftext|>world', 'hello <|endoftext|> world',
    'hello <|endoftext|>', 'hello <|endoftext|> ', '<|endoftext}>',
    '<|endoftext}> ', ' <|endoftext|>',
    '<|endoftext|><|endoftext|><|endoftext|><|endoftext|>',
    '<|endoftext|> <|endoftext|> <|endoftext|> <|endoftext|>'
]

TEST_STRINGS += HORRIBLE_STRINGS

MODEL_OR_ENCODING_NAME_TO_NON_UTF8_TOKENS = {
    'gpt-4': 77,
    'gpt-3.5-turbo': 77,
    'text-davinci-003': 14,
    'cl100k_base': 77,
}

MODEL_ENCODING_NAME_PARAMETRIZATION = [
    ('gpt-4', None),
    ('gpt-3.5-turbo', None),
    ('text-davinci-003', None),
    (None, 'cl100k_base'),
]


def get_tokenizers_for_testing(
    model_name: Optional[str], encoding_name: Optional[str],
    tmp_path: pathlib.Path
) -> Tuple[TiktokenTokenizerWrapper, TiktokenTokenizerWrapper, 'Encoding']:
    tiktoken = pytest.importorskip('tiktoken')

    # Construction
    wrapped_tokenizer = TiktokenTokenizerWrapper(model_name=model_name,
                                                 encoding_name=encoding_name)
    if model_name is not None:
        original_tokenizer = tiktoken.encoding_for_model(model_name)
    else:
        original_tokenizer = tiktoken.get_encoding(encoding_name)

    # Repr works
    _ = wrapped_tokenizer.__repr__()

    # Save and load
    wrapped_tokenizer.save_pretrained(tmp_path)
    reloaded_wrapped_tokenizer = transformers.AutoTokenizer.from_pretrained(
        tmp_path, trust_remote_code=True)

    return wrapped_tokenizer, reloaded_wrapped_tokenizer, original_tokenizer


@pytest.mark.parametrize('model_name,encoding_name',
                         MODEL_ENCODING_NAME_PARAMETRIZATION)
def test_tiktoken_simple(model_name: Optional[str],
                         encoding_name: Optional[str], tmp_path: pathlib.Path):
    wrapped_tokenizer, reloaded_wrapped_tokenizer, original_tokenizer = get_tokenizers_for_testing(
        model_name, encoding_name, tmp_path)

    # Simple tokenization test
    for string in TEST_STRINGS:
        wrapped_output = wrapped_tokenizer(string)
        original_output = original_tokenizer.encode(string,
                                                    allowed_special='all')
        reloaded_wrapped_output = reloaded_wrapped_tokenizer(string)

        assert wrapped_output['input_ids'] == original_output
        assert set(wrapped_output.keys()) == {'input_ids', 'attention_mask'}
        assert reloaded_wrapped_output == wrapped_output


@pytest.mark.parametrize('model_name,encoding_name',
                         MODEL_ENCODING_NAME_PARAMETRIZATION)
def test_tiktoken_roundtrip(model_name: Optional[str],
                            encoding_name: Optional[str],
                            tmp_path: pathlib.Path):
    wrapped_tokenizer, reloaded_wrapped_tokenizer, original_tokenizer = get_tokenizers_for_testing(
        model_name, encoding_name, tmp_path)

    for string in TEST_STRINGS:
        wrapped_output = wrapped_tokenizer.decode(
            wrapped_tokenizer(string)['input_ids'])
        original_output = original_tokenizer.decode(
            original_tokenizer.encode(string, allowed_special='all'))
        reloaded_wrapped_output = reloaded_wrapped_tokenizer.decode(
            reloaded_wrapped_tokenizer(string)['input_ids'])
        assert wrapped_output == string
        assert original_output == string
        assert reloaded_wrapped_output == string


@pytest.mark.parametrize('model_name,encoding_name',
                         MODEL_ENCODING_NAME_PARAMETRIZATION)
def test_tiktoken_batched(model_name: Optional[str],
                          encoding_name: Optional[str], tmp_path: pathlib.Path):
    wrapped_tokenizer, reloaded_wrapped_tokenizer, original_tokenizer = get_tokenizers_for_testing(
        model_name, encoding_name, tmp_path)

    wrapped_output = wrapped_tokenizer(
        ['Hello world!', 'Hello world but longer!'])
    original_output = original_tokenizer.encode_batch(
        ['Hello world!', 'Hello world but longer!'])
    reloaded_wrapped_output = reloaded_wrapped_tokenizer(
        ['Hello world!', 'Hello world but longer!'])
    assert wrapped_output['input_ids'] == original_output
    assert set(wrapped_output.keys()) == {'input_ids', 'attention_mask'}
    assert reloaded_wrapped_output == wrapped_output
    assert wrapped_tokenizer.batch_decode(
        wrapped_output['input_ids']) == original_tokenizer.decode_batch(
            original_output)
    assert reloaded_wrapped_tokenizer.batch_decode(
        reloaded_wrapped_output['input_ids']
    ) == original_tokenizer.decode_batch(original_output)


@pytest.mark.parametrize('model_name,encoding_name',
                         MODEL_ENCODING_NAME_PARAMETRIZATION)
def test_tiktoken_padding(model_name: Optional[str],
                          encoding_name: Optional[str], tmp_path: pathlib.Path):
    wrapped_tokenizer, reloaded_wrapped_tokenizer, original_tokenizer = get_tokenizers_for_testing(
        model_name, encoding_name, tmp_path)

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


@pytest.mark.parametrize('model_name,encoding_name',
                         MODEL_ENCODING_NAME_PARAMETRIZATION)
def test_tiktoken_vocab(model_name: Optional[str], encoding_name: Optional[str],
                        tmp_path: pathlib.Path):
    wrapped_tokenizer, reloaded_wrapped_tokenizer, original_tokenizer = get_tokenizers_for_testing(
        model_name, encoding_name, tmp_path)

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
    # This means that the str: int vocab mapping doesn't work. Would have to look more into how other HF tokenizers handle this.
    model_or_encoding_name = model_name or encoding_name
    if model_or_encoding_name is not None:
        expected_didnt_match = MODEL_OR_ENCODING_NAME_TO_NON_UTF8_TOKENS.get(
            model_or_encoding_name)
        assert len(didnt_match) == expected_didnt_match
    else:
        raise NotImplementedError(
            'Add the new tokenizer and how many tokens in the vocab are not utf8 representable.'
        )


@pytest.mark.parametrize('model_name,encoding_name',
                         MODEL_ENCODING_NAME_PARAMETRIZATION)
def test_tiktoken_save_from_pretrained(model_name: Optional[str],
                                       encoding_name: Optional[str],
                                       tmp_path: pathlib.Path):
    wrapped_tokenizer, reloaded_wrapped_tokenizer, _ = get_tokenizers_for_testing(
        model_name, encoding_name, tmp_path)
    check_hf_tokenizer_equivalence(wrapped_tokenizer,
                                   reloaded_wrapped_tokenizer)
