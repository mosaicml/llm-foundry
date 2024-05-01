# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest
import transformers

from llmfoundry.data.finetuning.tasks import (
    _slice_chat_formatted_example,
    dataset_constructor,
    tokenize_formatted_example,
)
from llmfoundry.utils.builders import build_tokenizer
from llmfoundry.utils.exceptions import (
    ALLOWED_PROMPT_KEYS,
    ALLOWED_RESPONSE_KEYS,
)


def test_tokenize_chat_example_malformed():
    no_content = {'messages': [{'role': 'user'}]}
    too_few_messages = {
        'messages': [{
            'role': 'assistant',
            'content': 'Hi, User!',
        }],
    }
    ends_with_user_role = {
        'messages': [{
            'role': 'user',
            'content': 'Hello GPT!',
        }, {
            'role': 'assistant',
            'content': 'Hi, User!',
        }, {
            'role': 'user',
            'content': 'user message not followed by an assistant label',
        }],
    }
    no_assistant_message = {
        'messages': [{
            'role': 'user',
            'content': 'Hello GPT!',
        }, {
            'role': 'user',
            'content': 'user message not followed by an assistant label',
        }],
    }
    wrong_type = {'messages': 'this is not a list of messages'}
    malformed_chat_examples = [
        too_few_messages,
        no_content,
        ends_with_user_role,
        no_assistant_message,
        wrong_type,
    ]
    my_tokenizer = build_tokenizer('mosaicml/mpt-7b-8k-chat', {})
    for example in malformed_chat_examples:
        with pytest.raises(Exception):
            tokenize_formatted_example(
                example,
                my_tokenizer,
            )  # type: ignore (the typing here is supposed to be malformed)


def test_tokenize_chat_example_well_formed():
    chat_examples = [
        {
            'messages': [{
                'role': 'user',
                'content': 'Hello, GPT',
            }, {
                'role': 'assistant',
                'content': 'this is my response',
            }],
        },  # prompt/response but in chat format
        {
            'messages': [
                {
                    'role': 'user',
                    'content': 'Hello, GPT',
                },
                {
                    'role': 'assistant',
                    'content': 'this is my response',
                },
                {
                    'role': 'user',
                    'content': 'Nice to hear that.',
                },
                {
                    'role': 'assistant',
                    'content': 'multi-way chat works too!',
                },
            ],
        },  # multi-way chat
    ]

    expected = [
        [{
            'prompt':
                '''<|im_start|>system
A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.
<|im_start|>user
Hello, GPT<|im_end|>
<|im_start|>assistant
''',
            'response':
                'this is my response<|im_end|>',
        }],
        [{
            'prompt':
                '''<|im_start|>system
A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.
<|im_start|>user
Hello, GPT<|im_end|>
<|im_start|>assistant
''',
            'response':
                'this is my response<|im_end|>',
        }, {
            'prompt':
                '''
<|im_start|>user
Nice to hear that.<|im_end|>
<|im_start|>assistant
''',
            'response':
                'multi-way chat works too!<|im_end|>',
        }],
    ]

    chat_tokenizer = build_tokenizer('mosaicml/mpt-7b-8k-chat', {})
    assert len(expected) == len(
        chat_examples,
    )  # if we add a new example, zip shouldn't fail silently
    for chat_example, expected_stringification in zip(chat_examples, expected):
        templatized_prompt_response_turns = _slice_chat_formatted_example(
            chat_example,
            chat_tokenizer,
        )
        tokenized_example = tokenize_formatted_example(
            chat_example,
            chat_tokenizer,
        )
        for (prompt, response), exp_str, turn in zip(
            templatized_prompt_response_turns,
            expected_stringification,
            tokenized_example['turns'],
        ):
            assert prompt == exp_str['prompt']
            assert response == exp_str['response']
            assert 'input_ids' in turn
            assert 'labels' in turn


def test_tokenize_instruct_example_malformed():
    no_keys = {}
    no_prompt_key = {'response': 'response'}
    no_response_key = {'prompt': 'prompt'}
    extra_keys_with_prompt = {'prompt': 'prompt', 'extra': 'extra'}
    extra_keys_with_response = {'response': 'response', 'extra': 'extra'}
    multiple_allowed_response_keys = {
        'prompt': 'prompt',
        'response': 'response',
        'completion': 'completion',
    }

    malformed_prompt_response_examples = [
        no_keys,
        no_prompt_key,
        no_response_key,
        extra_keys_with_prompt,
        extra_keys_with_response,
        multiple_allowed_response_keys,
    ]

    for example in malformed_prompt_response_examples:
        with pytest.raises(Exception):
            tokenize_formatted_example(example, MagicMock())


def test_tokenize_instruct_example_well_formed():
    tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')

    for prompt_key in ALLOWED_PROMPT_KEYS:
        for response_key in ALLOWED_RESPONSE_KEYS:

            example = {prompt_key: 'prompt', response_key: 'response'}
            tokenized_example = tokenize_formatted_example(example, tokenizer)
            assert 'input_ids' in tokenized_example['turns'][0]
            assert 'labels' in tokenized_example['turns'][0]


@pytest.mark.parametrize(
    'tokenizer_name',
    ['EleutherAI/gpt-neox-20b', 'HuggingFaceH4/zephyr-7b-beta', 't5-base'],
)
@pytest.mark.parametrize('messages_format', [True, False])
def test_multi_turn_chat_slicing(tokenizer_name: str, messages_format: bool):
    if messages_format:
        convo = [
            {
                'role': 'system',
                'content': 'everyone thinks you are so cool',
            },
            {
                'role': 'user',
                'content': 'hiiii',
            },
            {
                'role': 'assistant',
                'content': 'yassss',
            },
            {
                'role': 'user',
                'content': 'HIIIIII!!!',
            },
            {
                'role': 'assistant',
                'content': 'YASSSSSS',
            },
        ]
    else:
        convo = [
            {
                'from': 'system',
                'value': 'everyone thinks you are so cool',
            },
            {
                'from': 'human',
                'value': 'hiiii',
            },
            {
                'from': 'gpt',
                'value': 'yassss',
            },
            {
                'from': 'tool',
                'value': 'HIIIIII!!!',
            },
            {
                'from': 'gpt',
                'value': 'YASSSSSS',
            },
        ]
        tmp = {'conversations': convo}
        preprocessor = dataset_constructor.get_preprocessing_fn_from_str(
            'teknium/OpenHermes-2.5',
        )
        assert preprocessor is not None
        convo = preprocessor(tmp)['messages']
        assert isinstance(convo, list)

    example = {'messages': convo}

    tok = transformers.AutoTokenizer.from_pretrained(tokenizer_name)

    templated_prompt_response_turns = _slice_chat_formatted_example(
        example,
        tok,
    )

    reconstructed_chat = ''
    for prompt, response in templated_prompt_response_turns:
        reconstructed_chat += prompt + response

    full_chat = tok.apply_chat_template(convo, tokenize=False)
    assert reconstructed_chat == full_chat


def test_tokenize_no_labels_bos_pr():
    # This tokenizer automatically adds bos tokens
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        'ai21labs/Jamba-v0.1',
        add_bos_token=True,
    )

    example = {'prompt': 'prompt', 'response': 'response'}

    assert tokenizer.add_bos_token == True

    tokenized_example = tokenize_formatted_example(example, tokenizer)

    # Extract the first turn
    tokenized_example = tokenized_example['turns'][0]

    assert len(tokenized_example['labels']) == 1
    assert tokenized_example['labels'][0] != tokenizer.bos_token_id
    assert tokenized_example['input_ids'][0] == tokenizer.bos_token_id

    # This tokenizer does not have the add_bos_token attribute
    tokenizer = transformers.AutoTokenizer.from_pretrained('mosaicml/mpt-7b')

    assert not tokenizer.add_bos_token

    tokenized_example = tokenize_formatted_example(example, tokenizer)

    # Extract the first turn
    tokenized_example = tokenized_example['turns'][0]

    assert len(tokenized_example['labels']) == 1
    assert tokenized_example['labels'][0] != tokenizer.bos_token_id
    assert tokenized_example['input_ids'][0] != tokenizer.bos_token_id
