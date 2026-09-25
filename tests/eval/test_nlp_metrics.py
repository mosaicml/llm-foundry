# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
from typing import Any, cast

import pytest
import torch
from transformers import PreTrainedTokenizerBase

from llmfoundry.eval.metrics import (
    InContextLearningGenerationExactMatchAccuracy,
    InContextLearningLMAccuracy,
    InContextLearningMultipleChoiceAccuracy,
)


def test_in_context_learning_lm_accuracy(
    tiny_gpt2_with_pad_tokenizer: PreTrainedTokenizerBase,
):
    contexts = ['The dog is', 'I love to eat', 'I hate', 'The weather is']
    continuations = [' furry', ' pie', ' long lines', ' snowy']
    pad = tiny_gpt2_with_pad_tokenizer.pad_token_id
    assert isinstance(pad, int)
    inputs = [
        cast(Any,
             tiny_gpt2_with_pad_tokenizer(context)['input_ids']) +
        cast(Any,
             tiny_gpt2_with_pad_tokenizer(continuation)['input_ids'])
        for context, continuation in zip(contexts, continuations)
    ]
    inputs = torch.tensor([
        input + [pad] * (2048 - len(input)) for input in inputs
    ])

    cont_idxs = []
    for context, continuation in zip(contexts, continuations):
        start = len(
            cast(Any,
                 tiny_gpt2_with_pad_tokenizer(context)['input_ids']),
        )
        end = start + len(
            cast(Any,
                 tiny_gpt2_with_pad_tokenizer(continuation)['input_ids']),
        )
        cont_idxs.append(torch.tensor(list(range(start, end))))

    batch = {
        'continuation_indices': cont_idxs,
        'labels': inputs.roll(-1),
        'input_ids': inputs,
    }
    logits = torch.nn.functional.one_hot(inputs.roll(-1),
                                         num_classes=pad + 1).float() * 100
    start, end = cont_idxs[1].tolist()[0] - 1, cont_idxs[1].tolist()[-1]
    logits[1][start:end] = logits[0][start:end].clone(
    )  # make one of the answer's continuations incorrect
    metric = InContextLearningLMAccuracy()
    metric.update(batch, logits, batch['labels'])

    assert metric.compute() == 0.75


def test_in_context_learning_qa_accuracy():
    outputs = [
        'Correct but then some more text',
        'Incorrect',
        ' the CORREct with weird casing and spacing',
    ]
    labels = [['Correct'], ['blah', 'blah2'], ['blah', 'correct']]
    batch = {'cot_delimiter': '', 'labels': labels}
    metric = InContextLearningGenerationExactMatchAccuracy()
    metric.update(batch, outputs, labels)

    assert metric.compute() == (2 / 3)


def test_in_context_learning_qa_cot_accuracy():
    outputs = [
        'chain of thought ### Correct but then some more text\n\nanother chain of thought ### Incorrect answer this time',
        'Incorrect',
        'chain of thought ### the CORREct with weird casing and spacing',
        'incorrect chain of thought delimiter ## Correct but wrong delimiter',
    ]
    labels = [['Correct'], ['blah', 'blah2'], ['blah', 'correct'], ['correct']]
    batch = {
        'cot_delimiter': ' ### ',
        'labels': labels,
        'do_normalization': True,
        'stopping_criteria': '\n\n',
    }
    metric = InContextLearningGenerationExactMatchAccuracy()
    metric.update(batch, outputs, labels)

    assert metric.compute() == (2 / 4)


@pytest.mark.parametrize(
    'output,labels,do_normalization,expected',
    [
        # A label that is empty once cleaned (a bare article, punctuation
        # only, whitespace only) must not match arbitrary output.
        pytest.param(
            'Some unrelated text',
            ['The'],
            True,
            0.0,
            id='bare_article',
        ),
        pytest.param('garbage', ['A'], True, 0.0, id='bare_article_a'),
        pytest.param('garbage', ['...'], True, 0.0, id='punctuation_only'),
        pytest.param(
            'I cannot answer that.',
            ['The'],
            True,
            0.0,
            id='refusal',
        ),
        pytest.param(
            'London',
            ['the', 'Paris'],
            True,
            0.0,
            id='empty_alias_next_to_real_alias',
        ),
        pytest.param(
            'garbage',
            ['   '],
            False,
            0.0,
            id='whitespace_only_no_normalization',
        ),
        # Controls that must behave the same before and after the fix.
        pytest.param('London', ['Paris'], True, 0.0, id='plain_miss'),
        pytest.param('', ['Paris'], True, 0.0, id='empty_output_plain_label'),
        pytest.param(
            'Paris is the capital',
            ['Paris'],
            True,
            1.0,
            id='plain_prefix_hit',
        ),
        pytest.param(
            'paris',
            ['the', 'Paris'],
            True,
            1.0,
            id='real_alias_still_matches',
        ),
        pytest.param(
            'The Who were a band',
            ['The Who'],
            True,
            1.0,
            id='article_inside_label',
        ),
        pytest.param('The The The', ['The Who'], True, 0.0, id='article_miss'),
        pytest.param(
            'The cat',
            ['The'],
            False,
            1.0,
            id='no_normalization_prefix_hit',
        ),
        pytest.param(
            'the cat',
            ['The'],
            False,
            0.0,
            id='no_normalization_case_miss',
        ),
        # A correct answer: the gold is a bare article and the model says
        # exactly that. Dropping empty labels would score this as wrong.
        pytest.param('The', ['The'], True, 1.0, id='bare_article_exact'),
        pytest.param('the.', ['The'], True, 1.0, id='bare_article_punct'),
        pytest.param(
            'A',
            ['a', 'letter a'],
            True,
            1.0,
            id='bare_article_alias',
        ),
    ],
)
def test_in_context_learning_qa_accuracy_empty_normalized_label(
    output: str,
    labels: list[str],
    do_normalization: bool,
    expected: float,
):
    batch = {
        'cot_delimiter': '',
        'labels': [labels],
        'do_normalization': do_normalization,
    }
    metric = InContextLearningGenerationExactMatchAccuracy()
    metric.update(batch, [output], [labels])

    assert metric.compute() == expected


def test_in_context_learning_qa_accuracy_empty_normalized_label_batch():
    outputs = ['The', 'garbage', 'Paris is the capital', 'London']
    labels = [['The'], ['the'], ['the', 'Paris'], ['Paris']]
    batch = {'cot_delimiter': '', 'labels': labels}
    metric = InContextLearningGenerationExactMatchAccuracy()
    metric.update(batch, outputs, labels)

    assert metric.compute() == 0.5


def test_in_context_learning_mc_accuracy(
    tiny_gpt2_with_pad_tokenizer: PreTrainedTokenizerBase,
):
    contexts = [
        'Q: How do you cook a cake?',
        'Q: How do you cook a cake?',
        'Q: How old is the earth?',
        'Q: How old is the earth?',
    ]
    continuations = [
        ' A: turn on the oven',
        ' A: do a backflip',
        ' A: 2 minutes',
        ' A: 4.5 billion years',
    ]
    gold_indices = [0, 1]
    choice_groupings = [(0, 2), (2, 4)]
    pad = tiny_gpt2_with_pad_tokenizer.pad_token_id
    assert isinstance(pad, int)
    inputs = [
        cast(Any,
             tiny_gpt2_with_pad_tokenizer(context)['input_ids']) +
        cast(Any,
             tiny_gpt2_with_pad_tokenizer(continuation)['input_ids'])
        for context, continuation in zip(contexts, continuations)
    ]
    inputs = torch.tensor([
        input + [pad] * (2048 - len(input)) for input in inputs
    ])
    attention_mask = ~(inputs == pad)

    cont_idxs = []
    for context, continuation in zip(contexts, continuations):
        start = len(
            cast(Any,
                 tiny_gpt2_with_pad_tokenizer(context)['input_ids']),
        )
        cont_iids = cast(
            Any,
            tiny_gpt2_with_pad_tokenizer(continuation)['input_ids'],
        )
        end = start + len(
            cont_iids,
        )
        cont_idxs.append(torch.tensor(list(range(start, end))))

    batch = {
        'continuation_indices': cont_idxs,
        'labels': inputs.roll(-1),
        'input_ids': inputs,
        'attention_mask': attention_mask,
        'gold_indices': gold_indices,
        'choice_groupings': choice_groupings,
    }
    logits = torch.nn.functional.one_hot(inputs.roll(-1),
                                         num_classes=pad + 1).float()

    # for the first two, the correct answer is continuation 0
    # make the answer correct by making continuation 0 more likely for both answers
    start, end = cont_idxs[1].tolist()[0] - 1, cont_idxs[1].tolist()[-1]
    logits[1][start:end] = logits[0][start:end].clone()

    # for the last two, the correct answer is continuation 3
    # make the answer incorrect by making continuation 2 more likely for both answers
    start, end = cont_idxs[3].tolist()[0], cont_idxs[3].tolist()[-1]
    logits[3][start:end] = logits[2][start:end].clone()

    metric = InContextLearningMultipleChoiceAccuracy()

    metric.update(batch, logits, batch['labels'])
    assert metric.compute() == 0.5
