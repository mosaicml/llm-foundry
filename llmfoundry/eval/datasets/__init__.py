# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Natively supported datasets."""

from llmfoundry.eval.datasets.in_context_learning_evaluation import (
    InContextLearningCodeEvalDataset, InContextLearningDataset,
    InContextLearningLMTaskDataset, InContextLearningMultipleChoiceTaskDataset,
    InContextLearningQATaskDataset, InContextLearningSchemaTaskDataset,
    get_icl_task_dataloader)

__all__ = [
    'InContextLearningDataset',
    'InContextLearningQATaskDataset',
    'InContextLearningLMTaskDataset',
    'InContextLearningCodeEvalDataset',
    'InContextLearningMultipleChoiceTaskDataset',
    'InContextLearningSchemaTaskDataset',
]
