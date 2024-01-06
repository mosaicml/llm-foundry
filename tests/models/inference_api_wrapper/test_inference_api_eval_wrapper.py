# Copyright 2022-2024 MosaicML LLM Foundry authors
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


@pytest.fixture(scope='module')
def openai_api_key_env_var() -> str:
    os.environ['OPENAI_API_KEY'] = 'dummy'
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
                })
            ])
    })


class MockTopLogProb:

    def __init__(self, expected_token: str) -> None:
        self.top_logprobs = [{expected_token: 0}]


class MockLogprob:

    def __init__(self, expected_token: str) -> None:
        self.logprobs = MockTopLogProb(expected_token)


class MockCompletion:

    def __init__(self, expected_token: str) -> None:
        self.choices = [MockLogprob(expected_token)]


class MockContent:

    def __init__(self, expected_token: str) -> None:
        setattr(self, 'content', expected_token)


class MockMessage:

    def __init__(self, expected_token: str) -> None:
        setattr(self, 'message', MockContent(expected_token))


class MockChatCompletion:

    def __init__(self, expected_token: str) -> None:
        setattr(self, 'choices', [MockMessage(expected_token)])


def mock_create(**kwargs: Dict[str, str]):
    prompt = kwargs['prompt']
    if prompt == 'AMERICAN HISTORY: On May 29, 1765 Patrick Henrys Stamp Act protest was interrupted with this one word\nAnswer:':  # pyright: ignore[reportUnnecessaryComparison]
        return MockCompletion(' Tre')

    elif prompt == 'AMERICAN HISTORY: On May 29, 1765 Patrick Henrys Stamp Act protest was interrupted with this one word\nAnswer: Tre':  # pyright: ignore[reportUnnecessaryComparison]
        return MockCompletion('ason')

    elif prompt == 'AMERICAN HISTORY: On May 29, 1765 Patrick Henrys Stamp Act protest was interrupted with this one word\nAnswer: Treason':  # pyright: ignore[reportUnnecessaryComparison]
        return MockCompletion('!')

    else:
        # dummy token to make sure the model is incorrect on any other prompt
        return MockCompletion(' ')


def test_openai_api_eval_wrapper(tmp_path: str, openai_api_key_env_var: str):
    _ = pytest.importorskip('openai')

    model_name = 'davinci'
    tokenizer = TiktokenTokenizerWrapper(model_name=model_name,
                                         pad_token='<|endoftext|>')
    model = OpenAICausalLMEvalWrapper(model_cfg={'version': model_name},
                                      tokenizer=tokenizer)
    with patch.object(model, 'client') as mock:
        mock.completions.create = mock_create

        task_cfg = load_icl_config()
        evaluators, _ = build_icl_evaluators(task_cfg.icl_tasks,
                                             tokenizer,
                                             1024,
                                             2,
                                             destination_dir=str(tmp_path))

        batch = next(evaluators[0].dataloader.dataloader.__iter__())
        result = model.eval_forward(batch)
        model.update_metric(batch,
                            result,
                            metric=model.get_metrics()
                            ['InContextLearningLMAccuracy'])  # pyright: ignore
        acc = model.get_metrics(
        )['InContextLearningLMAccuracy'].compute(  # pyright: ignore
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
        mock.chat.completions.create.return_value = MockChatCompletion(
            'Treason!')

        task_cfg = load_icl_config()
        evaluators, _ = build_icl_evaluators(task_cfg.icl_tasks,
                                             tokenizer,
                                             1024,
                                             2,
                                             destination_dir=str(tmp_path))

        batch = next(evaluators[0].dataloader.dataloader.__iter__())
        result = chatmodel.eval_forward(batch)
        chatmodel.update_metric(
            batch,
            result,
            metric=chatmodel.get_metrics()
            ['InContextLearningLMAccuracy'])  # pyright: ignore
        acc = chatmodel.get_metrics(
        )['InContextLearningLMAccuracy'].compute(  # pyright: ignore
        )
        assert acc == 0.5
