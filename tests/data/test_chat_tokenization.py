# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest
import transformers

from llmfoundry.data.finetuning.tasks import (_ALLOWED_PROMPT_KEYS,
                                              _ALLOWED_RESPONSE_KEYS,
                                              _slice_chat_formatted_example,
                                              _tokenize_formatted_example)
from llmfoundry.utils.builders import build_tokenizer


def test_tokenize_chat_example_malformed():
    no_content = {'messages': [{'role': 'user'}]}
    ends_with_user_role = {
        'messages': [{
            'role': 'user',
            'content': 'Hello GPT!'
        }, {
            'role': 'assistant',
            'content': 'Hi, User!'
        }, {
            'role': 'user',
            'content': 'user message not followed by an assistant label'
        }]
    }
    no_assistant_message = {
        'messages': [{
            'role': 'user',
            'content': 'Hello GPT!'
        }, {
            'role': 'user',
            'content': 'user message not followed by an assistant label'
        }]
    }
    malformed_chat_examples = [
        no_content, ends_with_user_role, no_assistant_message
    ]
    my_tokenizer = build_tokenizer('mosaicml/mpt-7b-8k-chat', {})
    for example in malformed_chat_examples:
        with pytest.raises(Exception):
            _tokenize_formatted_example(
                example, my_tokenizer
            )  # type: ignore (the typing here is supposed to be malformed)


def test_tokenize_chat_example_well_formed():
    chat_examples = [
        {
            'messages': [{
                'role': 'user',
                'content': 'Hello, GPT'
            }, {
                'role': 'assistant',
                'content': 'this is my response'
            }]
        },  # prompt/response but in chat format
        {
            'messages': [
                {
                    'role': 'user',
                    'content': 'Hello, GPT'
                },
                {
                    'role': 'assistant',
                    'content': 'this is my response'
                },
                {
                    'role': 'user',
                    'content': 'Nice to hear that.'
                },
                {
                    'role': 'assistant',
                    'content': 'multi-way chat works too!'
                },
            ]
        },  # multi-way chat
    ]

    expected = [
        {
            'prompt': 'TODO: fix',
            'response': 'TODO: fix'
        },
        {
            'prompt': 'TODO: fix',
            'response': 'TODO: fix'
        },
    ]

    chat_tokenizer = build_tokenizer('mosaicml/mpt-7b-8k-chat', {})
    assert len(expected) == len(
        chat_examples)  # if we add a new example, zip shouldn't fail silently
    for chat_example, expected_stringification in zip(chat_examples, expected):
        prompt, response = _slice_chat_formatted_example(
            chat_example, chat_tokenizer)
        tokenized_example = _tokenize_formatted_example(chat_example,
                                                        chat_tokenizer)
        assert prompt == expected_stringification['prompt']
        assert response == expected_stringification['response']
        assert 'input_ids' in tokenized_example
        assert 'labels' in tokenized_example


def test_tokenize_instruct_example_malformed():
    no_keys = {}
    no_prompt_key = {'response': 'response'}
    no_response_key = {'prompt': 'prompt'}
    extra_keys_with_prompt = {'prompt': 'prompt', 'extra': 'extra'}
    extra_keys_with_response = {'response': 'response', 'extra': 'extra'}
    multiple_allowed_response_keys = {
        'prompt': 'prompt',
        'response': 'response',
        'completion': 'completion'
    }

    malformed_prompt_response_examples = [
        no_keys, no_prompt_key, no_response_key, extra_keys_with_prompt,
        extra_keys_with_response, multiple_allowed_response_keys
    ]

    for example in malformed_prompt_response_examples:
        with pytest.raises(KeyError):
            _tokenize_formatted_example(example, MagicMock())


def test_tokenize_instruct_example_well_formed():
    tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')

    for prompt_key in _ALLOWED_PROMPT_KEYS:
        for response_key in _ALLOWED_RESPONSE_KEYS:

            example = {prompt_key: 'prompt', response_key: 'response'}
            tokenized_example = _tokenize_formatted_example(example, tokenizer)
            assert 'input_ids' in tokenized_example
            assert 'labels' in tokenized_example
