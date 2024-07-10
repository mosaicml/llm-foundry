# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from llmfoundry.eval.datasets.in_context_learning_evaluation import (
    InContextLearningDataset,
    InContextLearningGenerationTaskWithAnswersDataset,
    InContextLearningLMTaskDataset,
    InContextLearningMultipleChoiceTaskDataset,
    InContextLearningSchemaTaskDataset,
    get_icl_task_dataloader,
)
from llmfoundry.eval.metrics.nlp import (
    InContextLearningGenerationExactMatchAccuracy,
    InContextLearningLMAccuracy,
    InContextLearningLMExpectedCalibrationError,
    InContextLearningMCExpectedCalibrationError,
    InContextLearningMetric,
    InContextLearningMultipleChoiceAccuracy,
)

from llmfoundry.eval.eval import (
    evaluate,
    eval_from_yaml,
)

__all__ = [
    'evaluate',
    'eval_from_yaml',
    'InContextLearningDataset',
    'InContextLearningLMTaskDataset',
    'InContextLearningMultipleChoiceTaskDataset',
    'InContextLearningSchemaTaskDataset',
    'InContextLearningGenerationTaskWithAnswersDataset',
    'get_icl_task_dataloader',
    'InContextLearningMetric',
    'InContextLearningLMAccuracy',
    'InContextLearningMultipleChoiceAccuracy',
    'InContextLearningGenerationExactMatchAccuracy',
    'InContextLearningLMExpectedCalibrationError',
    'InContextLearningMCExpectedCalibrationError',
]
