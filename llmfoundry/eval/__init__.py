# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Natively supported datasets."""

from llmfoundry.eval.datasets import (
    InContextLearningCodeEvalDataset, InContextLearningDataset,
    InContextLearningGenerationTaskWithAnswersDataset,
    InContextLearningLMTaskDataset, InContextLearningMultipleChoiceTaskDataset,
    InContextLearningSchemaTaskDataset, get_continuation_span,
    get_fewshot_sample_idxs, get_icl_task_dataloader, make_padded_input,
    strip_data, tokenizer_needs_prefix_space, trim_context)
from llmfoundry.eval.metrics import (
    InContextLearningCodeEvalAccuracy,
    InContextLearningGenerationExactMatchAccuracy, InContextLearningLMAccuracy,
    InContextLearningLMExpectedCalibrationError,
    InContextLearningMCExpectedCalibrationError, InContextLearningMetric,
    InContextLearningMultipleChoiceAccuracy)

__all__ = [
    'InContextLearningLMAccuracy',
    'InContextLearningMultipleChoiceAccuracy',
    'InContextLearningGenerationExactMatchAccuracy',
    'InContextLearningMCExpectedCalibrationError',
    'InContextLearningLMExpectedCalibrationError',
    'InContextLearningMetric',
    'InContextLearningCodeEvalAccuracy',
    'InContextLearningDataset',
    'InContextLearningGenerationTaskWithAnswersDataset',
    'InContextLearningLMTaskDataset',
    'InContextLearningCodeEvalDataset',
    'InContextLearningMultipleChoiceTaskDataset',
    'InContextLearningSchemaTaskDataset',
    'get_icl_task_dataloader',
    'strip_data',
    'tokenizer_needs_prefix_space',
    'trim_context',
    'get_continuation_span',
    'get_fewshot_sample_idxs',
    'make_padded_input',
]
