# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from typing import Dict
from unittest.mock import patch

import pytest
from omegaconf import DictConfig, ListConfig

from llmfoundry.models.inference_api_wrapper import (OpenAICausalLMEvalWrapper,
                                                     OpenAIChatAPIEvalWrapper)
from llmfoundry.tokenizers import TiktokenTokenizerWrapper
from llmfoundry.utils.builders import build_icl_evaluators


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


def mock_create(**kwargs: Dict[str, str]):
    prompt = kwargs['prompt']
    if prompt == 'AMERICAN HISTORY: On May 29, 1765 Patrick Henrys Stamp Act protest was interrupted with this one word\nAnswer:':  # pyright: ignore[reportUnnecessaryComparison]
        return {
            'choices': [{
                'logprobs': {
                    'top_logprobs': [{
                        ' Tre': 0,
                    }],
                },
            }],
        }
    elif prompt == 'AMERICAN HISTORY: On May 29, 1765 Patrick Henrys Stamp Act protest was interrupted with this one word\nAnswer: Tre':  # pyright: ignore[reportUnnecessaryComparison]
        return {
            'choices': [{
                'logprobs': {
                    'top_logprobs': [{
                        'ason': 0,
                    }],
                },
            }],
        }
    elif prompt == 'AMERICAN HISTORY: On May 29, 1765 Patrick Henrys Stamp Act protest was interrupted with this one word\nAnswer: Treason':  # pyright: ignore[reportUnnecessaryComparison]
        return {
            'choices': [{
                'logprobs': {
                    'top_logprobs': [{
                        '!': 0,
                    }],
                },
            }],
        }
    else:
        # dummy token to make sure the model is incorrect on any other prompt
        return {
            'choices': [{
                'logprobs': {
                    'top_logprobs': [{
                        ' ': 0,
                    }],
                },
            }],
        }


def test_openai_api_eval_wrapper(tmp_path: str):
    _ = pytest.importorskip('openai')
    with patch('openai.Completion') as mock:
        mock.create = mock_create
        model_name = 'davinci'
        tokenizer = TiktokenTokenizerWrapper(model_name=model_name,
                                             pad_token='<|endoftext|>')
        model = OpenAICausalLMEvalWrapper(model_cfg={'version': model_name},
                                          tokenizer=tokenizer)
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


def test_chat_api_eval_wrapper(tmp_path: str):
    _ = pytest.importorskip('openai')
    with patch('openai.ChatCompletion') as mock:
        mock.create.return_value = {
            'choices': [{
                'message': {
                    'role': 'assistant',
                    'content': 'Treason!'
                },
            }],
        }
        model_name = 'gpt-3.5-turbo'
        tokenizer = TiktokenTokenizerWrapper(model_name=model_name,
                                             pad_token='<|endoftext|>')
        chatmodel = OpenAIChatAPIEvalWrapper(model_cfg={'version': model_name},
                                             tokenizer=tokenizer)
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
