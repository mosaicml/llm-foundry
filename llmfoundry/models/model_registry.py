# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from llmfoundry.models.hf import (ComposerHFCausalLM, ComposerHFPrefixLM,
                                  ComposerHFT5)
from llmfoundry.models.inference_api_wrapper import (FMAPICasualLMEvalWrapper,
                                                     FMAPIChatAPIEvalWrapper,
                                                     OpenAICausalLMEvalWrapper,
                                                     OpenAIChatAPIEvalWrapper)
from llmfoundry.models.mpt import ComposerMPTCausalLM

COMPOSER_MODEL_REGISTRY = {
    'mpt_causal_lm': ComposerMPTCausalLM,
    'hf_causal_lm': ComposerHFCausalLM,
    'hf_prefix_lm': ComposerHFPrefixLM,
    'hf_t5': ComposerHFT5,
    'openai_causal_lm': OpenAICausalLMEvalWrapper,
    'fmapi_causal_lm': FMAPICasualLMEvalWrapper,
    'openai_chat': OpenAIChatAPIEvalWrapper,
    'fmapi_chat': FMAPIChatAPIEvalWrapper,
}
