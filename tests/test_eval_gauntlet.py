# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Dict, List

import omegaconf as om
import pytest
import torch
from composer.core import State
from composer.loggers import InMemoryLogger, Logger
from composer.metrics import InContextLearningLMAccuracy
from transformers import AutoTokenizer

from llmfoundry.utils.builders import build_icl_data_and_gauntlet


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


@pytest.mark.parametrize(
    'gauntlet_from_path',
    [True, False],
)
def test_gauntlet_callback(gauntlet_from_path: bool):
    icl_task_config = om.OmegaConf.create("""
            - label: jeopardy
              dataset_uri: eval/local_data/world_knowledge/jeopardy_all.jsonl # ADD YOUR OWN DATASET URI
              num_fewshot: [10]
              icl_task_type: language_modeling
              continuation_delimiter: "\nAnswer: " # this separates questions from answers
              has_categories: true
            - label: lambada_openai
              dataset_uri: eval/local_data/language_understanding/lambada_openai.jsonl
              num_fewshot: [0]
              icl_task_type: language_modeling
            - label: squad
              dataset_uri: eval/local_data/reading_comprehension/squad.jsonl # ADD YOUR OWN DATASET URI
              num_fewshot: [10]
              icl_task_type: language_modeling
            """)
    assert isinstance(icl_task_config, om.ListConfig) or isinstance(
        icl_task_config, str)

    if gauntlet_from_path:
        eval_gauntlet_config = 'eval/yamls/eval_gauntlet.yaml'
    else:
        eval_gauntlet_config = om.OmegaConf.create("""
                weighting: EQUAL
                subtract_random_baseline: true
                rescale_accuracy: true
                categories:
                - name: world_knowledge
                  benchmarks:
                    - name: jeopardy
                      num_fewshot: 10
                      random_baseline: 0
                    - name: bigbench_qa_wikidata
                      num_fewshot: 10
                      random_baseline: 0
                    - name: arc_easy
                      num_fewshot: 10
                      random_baseline: 0.25
                    - name: arc_challenge
                      num_fewshot: 10
                      random_baseline: 0.25
                    - name: mmlu
                      num_fewshot: 10
                      random_baseline: 0.25
                    - name: bigbench_misconceptions
                      num_fewshot: 10
                      random_baseline: 0.5
                - name: commonsense_reasoning
                  benchmarks:
                    - name: copa
                      num_fewshot: 0
                      random_baseline: 0.5
                    - name: piqa
                      num_fewshot: 10
                      random_baseline: 0.5
                    - name: openbook_qa
                      num_fewshot: 0
                      random_baseline: 0.25
                    - name: bigbench_novel_concepts
                      num_fewshot: 10
                      random_baseline: 0.25
                    - name: bigbench_strange_stories
                      num_fewshot: 10
                      random_baseline: 0.5
                    - name: bigbench_strategy_qa
                      num_fewshot: 10
                      random_baseline: 0.5
                - name: language_understanding
                  benchmarks:
                    - name: lambada_openai
                      num_fewshot: 0
                      random_baseline: 0.0
                    - name: hellaswag
                      num_fewshot: 10
                      random_baseline: 0.25
                    - name: winograd
                      num_fewshot: 0
                      random_baseline: 0.5
                    - name: winogrande
                      num_fewshot: 0
                      random_baseline: 0.5
                    - name: bigbench_conlang_translation
                      num_fewshot: 0
                      random_baseline: 0.0
                    - name: bigbench_language_identification
                      num_fewshot: 10
                      random_baseline: 0.25
                    - name: bigbench_conceptual_combinations
                      num_fewshot: 10
                      random_baseline: 0.25
                - name: symbolic_problem_solving
                  benchmarks:
                    - name: bigbench_elementary_math_qa
                      num_fewshot: 10
                      random_baseline: 0.25
                    - name: bigbench_dyck_languages
                      num_fewshot: 10
                      random_baseline: 0
                    - name: bigbench_cs_algorithms
                      num_fewshot: 10
                      random_baseline: 0
                    - name: bigbench_logical_deduction
                      num_fewshot: 10
                      random_baseline: 0.25
                    - name: bigbench_operators
                      num_fewshot: 10
                      random_baseline: 0.0
                    - name: bigbench_repeat_copy_logic
                      num_fewshot: 10
                      random_baseline: 0.0
                    - name: simple_arithmetic_withspaces
                      num_fewshot: 10
                      random_baseline: 0.0
                    - name: simple_arithmetic_nospaces
                      num_fewshot: 10
                      random_baseline: 0.0
                    - name: math_qa
                      num_fewshot: 10
                      random_baseline: 0.25
                    - name: logi_qa
                      num_fewshot: 10
                      random_baseline: 0.25
                - name: reading_comprehension
                  benchmarks:
                    - name: pubmed_qa_labeled
                      num_fewshot: 10
                      random_baseline: 0.0
                    - name: squad
                      num_fewshot: 10
                      random_baseline: 0
                    - name: bigbench_understanding_fables
                      num_fewshot: 10
                      random_baseline: 0.25
                    - name: boolq
                      num_fewshot: 10
                      random_baseline: 0.5
          """)
    assert isinstance(eval_gauntlet_config, om.DictConfig) or isinstance(
        eval_gauntlet_config, str)
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')

    # test loading functionality
    _, _, eval_gauntlet_callback = build_icl_data_and_gauntlet(
        icl_task_config, eval_gauntlet_config, tokenizer, 4, 1024, 1)
    assert eval_gauntlet_callback is not None
    state = MockState(eval_gauntlet_callback.logger_keys)
    logger = MockLogger(state)

    # test computing functionality
    result = eval_gauntlet_callback.eval_after_all(state, logger)

    for category in [
            'world_knowledge',
            'language_understanding',
            'reading_comprehension',
    ]:
        name = f'icl/metrics/eval_gauntlet/{category}'
        assert result[name] == pytest.approx(0.25)

    assert result['icl/metrics/eval_gauntlet/average'] == pytest.approx(0.25)
