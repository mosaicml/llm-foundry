# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Natively supported in-context learning evaluation datasets."""

from llmfoundry.eval.datasets.in_context_learning_evaluation import (
    InContextLearningDataset,
    InContextLearningGenerationTaskWithAnswersDataset,
    InContextLearningLMTaskDataset,
    InContextLearningMultipleChoiceTaskDataset,
    InContextLearningSchemaTaskDataset,
    get_icl_task_dataloader,
)
from llmfoundry.eval.datasets.utils import (
    MultiTokenEOSCriteria,
    convert_tokens_to_tensors,
    get_continuation_span,
    get_fewshot_sample_idxs,
    make_padded_input,
    stop_sequences_criteria,
    strip_data,
    tokenizer_needs_prefix_space,
    trim_context,
)

__all__ = [
    'InContextLearningDataset',
    'InContextLearningGenerationTaskWithAnswersDataset',
    'InContextLearningLMTaskDataset',
    'InContextLearningMultipleChoiceTaskDataset',
    'InContextLearningSchemaTaskDataset',
    'get_icl_task_dataloader',
    'MultiTokenEOSCriteria',
    'strip_data',
    'tokenizer_needs_prefix_space',
    'trim_context',
    'get_continuation_span',
    'make_padded_input',
    'convert_tokens_to_tensors',
    'get_fewshot_sample_idxs',
    'stop_sequences_criteria',
]
