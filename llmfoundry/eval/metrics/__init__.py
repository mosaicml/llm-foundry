# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""A collection of common torchmetrics."""

from llmfoundry.eval.metrics.nlp import (InContextLearningCodeEvalAccuracy, InContextLearningLMAccuracy,
                                  InContextLearningLMExpectedCalibrationError,
                                  InContextLearningMCExpectedCalibrationError, InContextLearningMetric,
                                  InContextLearningMultipleChoiceAccuracy, InContextLearningQAAccuracy)

__all__ = [
    'InContextLearningLMAccuracy',
    'InContextLearningMultipleChoiceAccuracy',
    'InContextLearningQAAccuracy',
    'InContextLearningMCExpectedCalibrationError',
    'InContextLearningLMExpectedCalibrationError',
    'InContextLearningMetric',
    'InContextLearningCodeEvalAccuracy',
]

