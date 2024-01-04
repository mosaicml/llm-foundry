# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import pathlib
from typing import TYPE_CHECKING, List, Optional, Tuple

import pytest
import transformers

from llmfoundry.tokenizers.tiktoken import (TiktokenTokenizerWrapper,
                                            bytes_to_unicode)
from tests.a_scripts.inference.test_convert_composer_to_hf import \
    check_hf_tokenizer_equivalence
from tests.horrible_strings import HORRIBLE_STRINGS

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

MODEL_ENCODING_NAME_PARAMETRIZATION = [
    ('gpt-4', None),
    ('gpt-3.5-turbo', None),
    ('text-davinci-003', None),
    (None, 'cl100k_base'),
    ('gpt2', None),
]

DEFAULT_SYSTEM_PROMPT = """<|im_start|>system\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible."""

MULTI_TURN_CHAT_ML = [
    [{
        'content':
            'Please summarize the goals in this text:\n\nGoing outside has benefits include reducing stress and triggering the relaxation response, which can help us not only feel better mentally, but even heal faster from physical ailments.',
        'role':
            'user'
    }, {
        'content': 'You should go outside and touch grass.',
        'role': 'assistant'
    }],
    [{
        'content':
            'You are a honest and helpful AI language model. Tell the user the truth, the whole truth, and nothing but the truth.',
        'role':
            'system'
    }, {
        'content':
            'Please summarize the goals in this text:\n\nGoing outside has benefits include reducing stress and triggering the relaxation response, which can help us not only feel better mentally, but even heal faster from physical ailments.',
        'role':
            'user'
    }, {
        'content': 'You should go outside and touch grass.',
        'role': 'assistant'
    }]
]

MULTI_TURN_CHAT_STRING_NO_SYSTEM_PROMPT = [
    """<|im_start|>user
Please summarize the goals in this text:

Going outside has benefits include reducing stress and triggering the relaxation response, which can help us not only feel better mentally, but even heal faster from physical ailments.<|im_end|>
<|im_start|>assistant
You should go outside and touch grass.<|im_end|><|endoftext|>""",
    """<|im_start|>system
You are a honest and helpful AI language model. Tell the user the truth, the whole truth, and nothing but the truth.
<|im_start|>user
Please summarize the goals in this text:

Going outside has benefits include reducing stress and triggering the relaxation response, which can help us not only feel better mentally, but even heal faster from physical ailments.<|im_end|>
<|im_start|>assistant
You should go outside and touch grass.<|im_end|><|endoftext|>"""
]

MULTI_TURN_CHAT_STRING_SYSTEM_PROMPT = [
    """<|im_start|>system
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible.
<|im_start|>user
Please summarize the goals in this text:

Going outside has benefits include reducing stress and triggering the relaxation response, which can help us not only feel better mentally, but even heal faster from physical ailments.<|im_end|>
<|im_start|>assistant
You should go outside and touch grass.<|im_end|><|endoftext|>""",
    """<|im_start|>system
You are a honest and helpful AI language model. Tell the user the truth, the whole truth, and nothing but the truth.
<|im_start|>user
Please summarize the goals in this text:

Going outside has benefits include reducing stress and triggering the relaxation response, which can help us not only feel better mentally, but even heal faster from physical ailments.<|im_end|>
<|im_start|>assistant
You should go outside and touch grass.<|im_end|><|endoftext|>"""
]

MULTI_TURN_GENERATE_CHAT_ML = [[{
    'content':
        'Please summarize the goals in this text:\n\nGoing outside has benefits include reducing stress and triggering the relaxation response, which can help us not only feel better mentally, but even heal faster from physical ailments.',
    'role':
        'user'
}]]

MULTI_TURN_GENERATE_STRING = [
    """<|im_start|>system
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible.
<|im_start|>user
Please summarize the goals in this text:

Going outside has benefits include reducing stress and triggering the relaxation response, which can help us not only feel better mentally, but even heal faster from physical ailments.<|im_end|>
<|im_start|>assistant
"""
]


def get_tokenizers_for_testing(
    model_name: Optional[str],
    encoding_name: Optional[str],
    tmp_path: pathlib.Path,
    use_default_system_prompt: bool = False,
    add_bos_token: bool = False,
    add_eos_token: bool = False,
    additional_special_tokens: Optional[List[str]] = None,
) -> Tuple[TiktokenTokenizerWrapper, TiktokenTokenizerWrapper, 'Encoding']:
    tiktoken = pytest.importorskip('tiktoken')

    # Construction
    wrapped_tokenizer = TiktokenTokenizerWrapper(
        model_name=model_name,
        encoding_name=encoding_name,
        add_bos_token=add_bos_token,
        add_eos_token=add_eos_token,
        use_default_system_prompt=use_default_system_prompt,
        additional_special_tokens=additional_special_tokens)
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
def test_tiktoken_tokenize_with_ids(model_name: Optional[str],
                                    encoding_name: Optional[str],
                                    tmp_path: pathlib.Path):
    wrapped_tokenizer, reloaded_wrapped_tokenizer, original_tokenizer = get_tokenizers_for_testing(
        model_name, encoding_name, tmp_path)

    for string in TEST_STRINGS:
        wrapped_output = wrapped_tokenizer.tokenize(string)
        original_output = original_tokenizer.encode(string,
                                                    allowed_special='all')
        reloaded_wrapped_output = reloaded_wrapped_tokenizer.tokenize(string)

        assert all([isinstance(t, str) for t in wrapped_output])
        assert len(wrapped_output) == len(original_output)
        assert wrapped_output == reloaded_wrapped_output

        redone_token_ids = wrapped_tokenizer.convert_tokens_to_ids(
            wrapped_output)
        assert redone_token_ids == original_output
        assert wrapped_tokenizer.convert_ids_to_tokens(
            redone_token_ids) == wrapped_output


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

    for key, value in wrapped_vocab.items():
        # Skip checking the extra ids we pad the vocab with
        if key.startswith('<extra_id') and key.endswith('>'):
            continue

        expected_decoding = ''.join([
            bytes_to_unicode()[ord(char)]
            for char in original_tokenizer.decode_single_token_bytes(
                value).decode('latin-1')
        ])
        assert expected_decoding == key


@pytest.mark.parametrize('model_name,encoding_name',
                         MODEL_ENCODING_NAME_PARAMETRIZATION)
def test_tiktoken_save_from_pretrained(model_name: Optional[str],
                                       encoding_name: Optional[str],
                                       tmp_path: pathlib.Path):
    wrapped_tokenizer, reloaded_wrapped_tokenizer, _ = get_tokenizers_for_testing(
        model_name, encoding_name, tmp_path)
    check_hf_tokenizer_equivalence(wrapped_tokenizer,
                                   reloaded_wrapped_tokenizer)


@pytest.mark.parametrize('model_name,encoding_name',
                         MODEL_ENCODING_NAME_PARAMETRIZATION)
def test_tiktoken_encode_plus(model_name: Optional[str],
                              encoding_name: Optional[str],
                              tmp_path: pathlib.Path):
    # Testing encode_plus which optionally wrap encodes with bos and eos tokens
    wrapped_tokenizer, _, _ = get_tokenizers_for_testing(model_name,
                                                         encoding_name,
                                                         tmp_path,
                                                         add_bos_token=True,
                                                         add_eos_token=True)

    for test_string in TEST_STRINGS:
        encoded_outputs = wrapped_tokenizer.encode_plus(
            test_string,
            add_special_tokens=True,
            return_special_tokens_mask=True)
        encoded_input_ids = encoded_outputs.input_ids
        assert encoded_input_ids[0] == wrapped_tokenizer.bos_token_id
        assert encoded_input_ids[-1] == wrapped_tokenizer.eos_token_id

        encoded_special_mask = encoded_outputs.special_tokens_mask
        assert encoded_special_mask[0] == 1
        assert encoded_special_mask[-1] == 1


@pytest.mark.parametrize('model_name,encoding_name',
                         MODEL_ENCODING_NAME_PARAMETRIZATION)
def test_additional_special_tokens(model_name: Optional[str],
                                   encoding_name: Optional[str],
                                   tmp_path: pathlib.Path):
    special_token_to_add = '<|im_start|>'
    wrapped_tokenizer, _, _ = get_tokenizers_for_testing(
        model_name,
        encoding_name,
        tmp_path,
        add_bos_token=False,
        add_eos_token=False,
        additional_special_tokens=[special_token_to_add])
    encoded_outputs = wrapped_tokenizer(special_token_to_add +
                                        ' hello')['input_ids']

    assert encoded_outputs[0] == wrapped_tokenizer.vocab_size
    assert len(encoded_outputs) == 2


@pytest.mark.parametrize('model_name,encoding_name',
                         MODEL_ENCODING_NAME_PARAMETRIZATION)
def test_chat_formatting(model_name: Optional[str],
                         encoding_name: Optional[str], tmp_path: pathlib.Path):
    special_tokens_to_add = ['<|im_start|>', '<im_end>']
    # Default behavior to not use default system prompt.
    wrapped_tokenizer, _, _ = get_tokenizers_for_testing(
        model_name,
        encoding_name,
        tmp_path,
        add_bos_token=False,
        add_eos_token=False,
        additional_special_tokens=special_tokens_to_add)
    for i, dict_chats in enumerate(MULTI_TURN_CHAT_ML):
        chat_str = wrapped_tokenizer.apply_chat_template(
            dict_chats, tokenize=False, add_generation_prompt=False)
        assert chat_str == MULTI_TURN_CHAT_STRING_NO_SYSTEM_PROMPT[i]
    # Using default system prompt.
    wrapped_tokenizer, _, _ = get_tokenizers_for_testing(
        model_name,
        encoding_name,
        tmp_path,
        use_default_system_prompt=True,
        add_bos_token=False,
        add_eos_token=False,
        additional_special_tokens=special_tokens_to_add)
    for i, dict_chats in enumerate(MULTI_TURN_CHAT_ML):
        chat_str = wrapped_tokenizer.apply_chat_template(
            dict_chats, tokenize=False, add_generation_prompt=False)
        assert chat_str == MULTI_TURN_CHAT_STRING_SYSTEM_PROMPT[i]
    for i, dict_chats in enumerate(MULTI_TURN_GENERATE_CHAT_ML):
        chat_str = wrapped_tokenizer.apply_chat_template(
            dict_chats, tokenize=False, add_generation_prompt=True)
        assert chat_str == MULTI_TURN_GENERATE_STRING[i]
