# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from composer.metrics import (InContextLearningCodeEvalAccuracy,
                              InContextLearningLMAccuracy,
                              InContextLearningLMExpectedCalibrationError,
                              InContextLearningMCExpectedCalibrationError,
                              InContextLearningMultipleChoiceAccuracy,
                              InContextLearningQAAccuracy, MaskedAccuracy)
from composer.metrics.nlp import LanguageCrossEntropy, LanguagePerplexity

from llmfoundry.metrics.token_acc import TokenAccuracy
from llmfoundry.registry import metrics

metrics.register('token_accuracy', func=TokenAccuracy)
metrics.register('lm_accuracy', func=InContextLearningLMAccuracy)
metrics.register('lm_expected_calibration_error',
                 func=InContextLearningLMExpectedCalibrationError)
metrics.register('mc_expected_calibration_error',
                 func=InContextLearningMCExpectedCalibrationError)
metrics.register('mc_accuracy', func=InContextLearningMultipleChoiceAccuracy)
metrics.register('qa_accuracy', func=InContextLearningQAAccuracy)
metrics.register('code_eval_accuracy', func=InContextLearningCodeEvalAccuracy)
metrics.register('language_cross_entropy', func=LanguageCrossEntropy)
metrics.register('language_perplexity', func=LanguagePerplexity)
metrics.register('masked_accuracy', func=MaskedAccuracy)

DEFAULT_CAUSAL_LM_TRAIN_METRICS = [
    'language_cross_entropy',
    'language_perplexity',
    'token_accuracy',
]

DEFAULT_CAUSAL_LM_EVAL_METRICS = [
    'language_cross_entropy',
    'language_perplexity',
    'token_accuracy',
    'lm_accuracy',
    'lm_expected_calibration_error',
    'mc_expected_calibration_error',
    'mc_accuracy',
    'qa_accuracy',
    'code_eval_accuracy',
]

DEFAULT_PREFIX_LM_METRICS = [
    'language_cross_entropy',
    'masked_accuracy',
]

DEFAULT_ENC_DEC_METRICS = [
    'language_cross_entropy',
    'masked_accuracy',
]

__all__ = [
    'TokenAccuracy',
    'InContextLearningLMAccuracy',
    'InContextLearningLMExpectedCalibrationError',
    'InContextLearningMCExpectedCalibrationError',
    'InContextLearningMultipleChoiceAccuracy',
    'InContextLearningQAAccuracy',
    'InContextLearningCodeEvalAccuracy',
    'LanguageCrossEntropy',
    'LanguagePerplexity',
    'MaskedAccuracy',
    'DEFAULT_CAUSAL_LM_TRAIN_METRICS',
    'DEFAULT_CAUSAL_LM_EVAL_METRICS',
    'DEFAULT_PREFIX_LM_METRICS',
    'DEFAULT_ENC_DEC_METRICS',
]
