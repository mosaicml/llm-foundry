# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Dict
from unittest.mock import patch

import pytest
from omegaconf import DictConfig, ListConfig

from llmfoundry.models.inference_api_wrapper import (OpenAICausalLMEvalWrapper,
                                                     OpenAIChatAPIEvalWrapper)
from llmfoundry.tokenizers import TiktokenTokenizerWrapper
from llmfoundry.utils.builders import build_icl_evaluators
from openai.types.chat.chat_completion import ChatCompletion

HUMAN_EVAL_PROMPT = 'from typing import List\n\n\ndef separate_paren_groups(paren_string: str) -> List[str]:\n    """ Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n    separate those group into separate strings and return the list of those.\n    Separate groups are balanced (each open brace is properly closed) and not nested within each other\n    Ignore any spaces in the input string.\n    >>> separate_paren_groups(\'( ) (( )) (( )( ))\')\n    [\'()\', \'(())\', \'(()())\']\n    """\n'
CANONICAL_SOLN = """    result = []\n    current_string = []\n    current_depth = 0\n\n    for c in paren_string:\n        if c == '(':\n            current_depth += 1\n            current_string.append(c)\n        elif c == ')':\n            current_depth -= 1\n            current_string.append(c)\n\n            if current_depth == 0:\n                result.append(''.join(current_string))\n                current_string.clear()\n\n    return result\n"""
    

@pytest.fixture(scope='module')
def openai_api_key_env_var() -> str:
    # os.environ['OPENAI_API_KEY'] = 'dummy'
    return os.environ['OPENAI_API_KEY']


def load_icl_config():
    return DictConfig({
        'icl_tasks':
            ListConfig([
                DictConfig({
                    'label':
                        'jeopardy',
                    'dataset_uri':
                        'scripts/eval/local_data/world_knowledge/jeopardy_all.jsonl',
                    'num_fewshot': [0, 1],
                    'icl_task_type':
                        'language_modeling',
                    'continuation_delimiter':
                        '\nAnswer: ',
                    'has_categories':
                        True
                }),
                DictConfig({
                    'label':
                        'triviaqa',
                    'dataset_uri':
                        'scripts/eval/local_data/world_knowledge/triviaqa_small.jsonl',
                    'num_fewshot': [0, 1],
                    'icl_task_type':
                        'question_answering',
                    'continuation_delimiter':
                        '',
                }),
                DictConfig({
                    'label':
                        'human_eval',
                    'dataset_uri':
                        'scripts/eval/local_data/programming/human_eval.jsonl',
                    'num_fewshot': [0],
                    'icl_task_type':
                        'code_evaluation',
                    'pass_at_k': 1,
                    'generation_kwargs': {
                        'generations_per_sample': 5,
                    },
                    'continuation_delimiter':
                        '',
                })
            ])
    })


class MockTopLogProb:

    def __init__(self, expected_token: str) -> None:
        self.top_logprobs = [{expected_token: 0}]


class MockTokenOutput:

    def __init__(self, expected_token: str) -> None:
        self.logprobs = MockTopLogProb(expected_token)
        self.text = expected_token


class MockCompletion:

    def __init__(self, expected_token: str) -> None:
        self.choices = [MockTokenOutput(expected_token)]


class MockContent:

    def __init__(self, expected_token: str) -> None:
        setattr(self, 'content', expected_token)


class MockMessage:

    def __init__(self, expected_token: str) -> None:
        setattr(self, 'message', MockContent(expected_token))


class MockChatCompletion:

    def __init__(self, expected_token: str) -> None:
        setattr(self, 'choices', [MockMessage(expected_token)])


def mock_create_jeopardy(**kwargs: Dict[str, str]):
    prompt = kwargs['prompt']
    assert isinstance(prompt, str)
    if prompt.endswith(
            'AMERICAN HISTORY: On May 29, 1765 Patrick Henrys Stamp Act protest was interrupted with this one word\nAnswer:'
    ):  # pyright: ignore[reportUnnecessaryComparison]
        return MockCompletion(' Tre')

    elif prompt.endswith(
            'AMERICAN HISTORY: On May 29, 1765 Patrick Henrys Stamp Act protest was interrupted with this one word\nAnswer: Tre'
    ):  # pyright: ignore[reportUnnecessaryComparison]
        return MockCompletion('ason')

    elif prompt.endswith(
            'AMERICAN HISTORY: On May 29, 1765 Patrick Henrys Stamp Act protest was interrupted with this one word\nAnswer: Treason'
    ):  # pyright: ignore[reportUnnecessaryComparison]
        return MockCompletion('!')

    else:
        # dummy token to make sure the model is incorrect on any other prompt
        return MockCompletion(' ')

def mock_create_human_eval(**kwargs: Dict[str, str]) :
    prompt = kwargs['prompt']

    
    assert isinstance(prompt, str)
    if prompt.endswith(HUMAN_EVAL_PROMPT):
        return MockCompletion(CANONICAL_SOLN)
    else:
        return MockCompletion('')
    
def mock_create_triviaqa(**kwargs: Dict[str, str]):
    prompt = kwargs['prompt']
    assert isinstance(prompt, str)
    if prompt.endswith(
            'Question: Who was the man behind The Chipmunks?\nAnswer:'
    ):  # pyright: ignore[reportUnnecessaryComparison]
        return MockCompletion('David Seville')
    else:
        # dummy token to make sure the model is incorrect on any other prompt
        return MockCompletion(' ')


def test_openai_completions_api_eval_wrapper(tmp_path: str,
                                             openai_api_key_env_var: str):
    _ = pytest.importorskip('openai')
    os.environ['CODE_EVAL_DEVICE'] = "LOCAL"

    model_name = 'gpt-3.5-turbo-instruct'
    tokenizer = TiktokenTokenizerWrapper(model_name=model_name,
                                         pad_token='<|endoftext|>')
    model = OpenAICausalLMEvalWrapper(model_cfg={'version': model_name},
                                      tokenizer=tokenizer)
    with patch.object(model, 'client') as mock:

        task_cfg = load_icl_config()
        evaluators, _ = build_icl_evaluators(task_cfg.icl_tasks,
                                                tokenizer,
                                                1024,
                                                2,
                                                destination_dir=str(tmp_path))

        for evaluator in evaluators:
            if evaluator.label == 'jeopardy/0-shot/american_history' or evaluator.label == 'jeopardy/1-shot/american_history':
                mock.completions.create = mock_create_jeopardy
                batch = next(evaluator.dataloader.dataloader.__iter__())
                result = model.eval_forward(batch)
                model.get_metrics()['InContextLearningLMAccuracy'].total = 0.0
                model.get_metrics()['InContextLearningLMAccuracy'].correct = 0.0

                model.update_metric(
                    batch,
                    result,
                    metric=model.get_metrics()
                    ['InContextLearningLMAccuracy'])  # pyright: ignore
                acc = model.get_metrics(
                )['InContextLearningLMAccuracy'].compute(  # pyright: ignore
                )  # pyright: ignore
                assert acc == 0.5
            elif evaluator.label == 'triviaqa/0-shot' or evaluator.label == 'triviaqa/1-shot':
                mock.completions.create = mock_create_triviaqa
                batch = next(evaluator.dataloader.dataloader.__iter__())
                result = model.eval_forward(batch)
                model.update_metric(
                    batch,
                    result,
                    metric=model.get_metrics()
                    ['InContextLearningQAAccuracy'])  # pyright: ignore
                acc = model.get_metrics(
                )['InContextLearningQAAccuracy'].compute(  # pyright: ignore
                )  # pyright: ignore
                assert acc == 0.5
            elif evaluator.label == 'human_eval/0-shot':
                mock.completions.create = mock_create_human_eval
                batch = next(evaluator.dataloader.dataloader.__iter__())
                result = model.eval_forward(batch)
                model.update_metric(
                    batch,
                    result,
                    metric=model.get_metrics()
                    ['InContextLearningCodeEvalAccuracy'])  # pyright: ignore
                acc = model.get_metrics(
                )['InContextLearningCodeEvalAccuracy'].compute(  # pyright: ignore
                )  # pyright: ignore
                assert acc == 0.5


def test_chat_api_eval_wrapper(tmp_path: str, openai_api_key_env_var: str):
    _ = pytest.importorskip('openai')

    model_name = 'gpt-3.5-turbo'
    tokenizer = TiktokenTokenizerWrapper(model_name=model_name,
                                         pad_token='<|endoftext|>')
    chatmodel = OpenAIChatAPIEvalWrapper(model_cfg={'version': model_name},
                                         tokenizer=tokenizer)
    with patch.object(chatmodel, 'client') as mock:
        task_cfg = load_icl_config()
        evaluators, _ = build_icl_evaluators(task_cfg.icl_tasks,
                                             tokenizer,
                                             1024,
                                             2,
                                             destination_dir=str(tmp_path))

        for evaluator in evaluators:

            if evaluator.label == 'jeopardy/0-shot/american_history' or evaluator.label == 'jeopardy/1-shot/american_history':
                mock.chat.completions.create.return_value = MockChatCompletion(
                    'Treason!')
                batch = next(evaluator.dataloader.dataloader.__iter__())
                result = chatmodel.eval_forward(batch)
                chatmodel.get_metrics(
                )['InContextLearningLMAccuracy'].total = 0.0
                chatmodel.get_metrics(
                )['InContextLearningLMAccuracy'].correct = 0.0

                chatmodel.update_metric(
                    batch,
                    result,
                    metric=chatmodel.get_metrics()
                    ['InContextLearningLMAccuracy'])  # pyright: ignore
                acc = chatmodel.get_metrics(
                )['InContextLearningLMAccuracy'].compute(  # pyright: ignore
                )  # pyright: ignore
                assert acc == 0.5
            elif evaluator.label == 'triviaqa/0-shot' or evaluator.label == 'triviaqa/1-shot':
                mock.chat.completions.create.return_value = MockChatCompletion(
                    'David Seville')
                batch = next(evaluator.dataloader.dataloader.__iter__())
                result = chatmodel.eval_forward(batch)
                chatmodel.update_metric(
                    batch,
                    result,
                    metric=chatmodel.get_metrics()
                    ['InContextLearningQAAccuracy'])  # pyright: ignore
                acc = chatmodel.get_metrics(
                )['InContextLearningQAAccuracy'].compute(  # pyright: ignore
                )  # pyright: ignore
                assert acc == 0.5
            elif evaluator.label == 'human_eval/0-shot':
                mock.chat.completions.create.return_value = MockChatCompletion(
                    CANONICAL_SOLN)
                batch = next(evaluator.dataloader.dataloader.__iter__())
                result = chatmodel.eval_forward(batch)
                chatmodel.update_metric(
                    batch,
                    result,
                    metric=chatmodel.get_metrics()
                    ['InContextLearningCodeEvalAccuracy'])  # pyright: ignore
                acc = chatmodel.get_metrics(
                )['InContextLearningCodeEvalAccuracy'].compute(  # pyright: ignore
                )  # pyright: ignore
                assert acc == 0.5
