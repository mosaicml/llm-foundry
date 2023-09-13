# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import os
import random
import shutil
from pathlib import Path

import pytest
from omegaconf import DictConfig, ListConfig

from llmfoundry.models.inference_api_wrapper import (OpenAICausalLMEvalWrapper,
                                                     OpenAIChatAPIEvalWrapper,
                                                     OpenAITokenizerWrapper)
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


@pytest.mark.skipif(
    os.getenv('OPENAI_API_KEY') is None,
    reason='Unit test requires OPENAI_API_KEY environment variable.')
def test_openai_api_eval_wrapper(tmp_path):
    model_name = 'davinci'
    tokenizer = OpenAITokenizerWrapper(model_name)
    model = OpenAICausalLMEvalWrapper(model_cfg={'version': model_name},
                                      tokenizer=tokenizer)
    task_cfg = load_icl_config()
    evaluators, _ = build_icl_evaluators(
        task_cfg.icl_tasks,
        tokenizer,
        1024,
        8,
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
    assert acc > 0.0


@pytest.mark.skipif(
    os.getenv('OPENAI_API_KEY') is None,
    reason='Unit test requires OPENAI_API_KEY environment variable.')
def test_chat_api_eval_wrapper(tmp_path):
    model_name = 'gpt-3.5-turbo'
    tokenizer = OpenAITokenizerWrapper(model_name)
    chatmodel = OpenAIChatAPIEvalWrapper(model_cfg={'version': model_name},
                                         tokenizer=tokenizer)
    task_cfg = load_icl_config()
    evaluators, _ = build_icl_evaluators(
        task_cfg.icl_tasks,
        tokenizer,
        1024,
        8,
        destination_dir=str(tmp_path))

    batch = next(evaluators[0].dataloader.dataloader.__iter__())
    result = chatmodel.eval_forward(batch)

    chatmodel.update_metric(batch,
                            result,
                            metric=chatmodel.get_metrics()
                            ['InContextLearningLMAccuracy'])  # pyright: ignore
    acc = chatmodel.get_metrics(
    )['InContextLearningLMAccuracy'].compute(  # pyright: ignore
    )
    assert acc > 0.0
