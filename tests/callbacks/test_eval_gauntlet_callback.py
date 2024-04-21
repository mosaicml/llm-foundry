# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Dict, List, Optional

import omegaconf as om
import pytest
import torch
from composer.core import State
from composer.loggers import InMemoryLogger, Logger
from transformers import AutoTokenizer

from llmfoundry.eval.metrics.nlp import InContextLearningLMAccuracy
from llmfoundry.utils.builders import build_icl_data_and_gauntlet
from llmfoundry.utils.config_utils import to_dict_recursive


@pytest.fixture(autouse=True)
def set_correct_cwd():
    if not os.getcwd().endswith('llm-foundry/scripts'):
        os.chdir('scripts')

    yield

    if os.getcwd().endswith('llm-foundry/scripts'):
        os.chdir('..')


class MockState(State):

    def __init__(self, logger_keys: List[str], accuracy: float = 0.25) -> None:
        self.eval_metrics = {}
        self.timestamp = 0
        for key in logger_keys:
            dl_name = '/'.join(key.split('/')[1:-1])
            self.eval_metrics[dl_name] = {}
            self.eval_metrics[dl_name][
                'InContextLearningLMAccuracy'] = InContextLearningLMAccuracy()
            self.eval_metrics[dl_name][
                'InContextLearningLMAccuracy'].correct = torch.tensor(accuracy *
                                                                      100)
            self.eval_metrics[dl_name][
                'InContextLearningLMAccuracy'].total = torch.tensor(100)


class MockLogger(Logger):

    def __init__(self, state: MockState):
        self.inmemorylogger = InMemoryLogger()
        self.inmemorylogger.state = state

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        self.inmemorylogger.log_metrics(metrics)


@pytest.mark.parametrize('averages', [{
    'core_average': ['world_knowledge', 'language_understanding']
}, None])
def test_gauntlet_callback(averages: Optional[dict]):
    icl_task_config = om.OmegaConf.create("""
            - label: jeopardy_small
              dataset_uri: eval/local_data/world_knowledge/jeopardy_small.jsonl # ADD YOUR OWN DATASET URI
              num_fewshot: [10]
              icl_task_type: language_modeling
              continuation_delimiter: "\nAnswer: " # this separates questions from answers
              has_categories: true
            - label: lambada_openai_small
              dataset_uri: eval/local_data/language_understanding/lambada_openai_small.jsonl # ADD YOUR OWN DATASET URI
              num_fewshot: [0]
              icl_task_type: language_modeling
            """)
    icl_task_config_list: List[om.DictConfig] = list(
        icl_task_config)  # type: ignore
    assert all(isinstance(c, om.DictConfig) for c in icl_task_config_list)

    eval_gauntlet_config = om.OmegaConf.create("""
                weighting: EQUAL
                subtract_random_baseline: true
                rescale_accuracy: true
                categories:
                - name: world_knowledge
                  benchmarks:
                    - name: jeopardy_small
                      num_fewshot: 10
                      random_baseline: 0
                - name: language_understanding
                  benchmarks:
                    - name: lambada_openai_small
                      num_fewshot: 0
                      random_baseline: 0.0
          """)
    assert isinstance(eval_gauntlet_config, om.DictConfig)

    if averages is not None:
        eval_gauntlet_config.averages = averages
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')

    # test loading functionality
    _, _, eval_gauntlet_callback = build_icl_data_and_gauntlet(
        [to_dict_recursive(c) for c in icl_task_config_list],
        to_dict_recursive(eval_gauntlet_config), tokenizer, 4, 1024, 1)
    assert eval_gauntlet_callback is not None
    state = MockState(eval_gauntlet_callback.logger_keys)
    logger = MockLogger(state)

    # test computing functionality
    result = eval_gauntlet_callback.eval_after_all(state, logger)

    for category in [
            'world_knowledge',
            'language_understanding',
    ]:
        name = f'icl/metrics/eval_gauntlet/{category}'
        assert result[name] == pytest.approx(0.25)

    if averages is None:
        assert result[
            'icl/metrics/eval_gauntlet/default_average'] == pytest.approx(0.25)
    else:
        assert result[
            'icl/metrics/eval_gauntlet/core_average'] == pytest.approx(0.25)
