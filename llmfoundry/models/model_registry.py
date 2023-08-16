# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from llmfoundry.models.hf import (ComposerHFBertForMaskedLM,
                                  ComposerHFBertForSequenceClassification,
                                  ComposerHFCausalLM, ComposerHFPrefixLM,
                                  ComposerHFT5)
from llmfoundry.models.mosaicbert import (
    ComposerMosaicBertForMaskedLM, ComposerMosaicBertForSequenceClassification)
from llmfoundry.models.mpt import ComposerMPTCausalLM

COMPOSER_MODEL_REGISTRY = {
    'mpt_causal_lm':
        ComposerMPTCausalLM,
    'hf_causal_lm':
        ComposerHFCausalLM,
    'hf_prefix_lm':
        ComposerHFPrefixLM,
    'hf_t5':
        ComposerHFT5,
    'mosaicbert_masked_lm':
        ComposerMosaicBertForMaskedLM,
    'mosaicbert_sequence_classification':
        ComposerMosaicBertForSequenceClassification,
    'hf_bert_masked_lm':
        ComposerHFBertForMaskedLM,
    'hf_bert_sequence_classification':
        ComposerHFBertForSequenceClassification,
}
