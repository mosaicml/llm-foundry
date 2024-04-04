# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from llmfoundry.models.hf import ComposerHFCausalLM, ComposerHFT5
from llmfoundry.models.inference_api_wrapper import (FMAPICasualLMEvalWrapper,
                                                     FMAPIChatAPIEvalWrapper,
                                                     OpenAICausalLMEvalWrapper,
                                                     OpenAIChatAPIEvalWrapper,
                                                     VLLMCausalLMEvalWrapper,
                                                     GeminiChatAPIEvalrapper)
from llmfoundry.models.mpt import (ComposerMPTCausalLM, MPTConfig,
                                   MPTForCausalLM, MPTModel, MPTPreTrainedModel)
from llmfoundry.registry import models

models.register('mpt_causal_lm', func=ComposerMPTCausalLM)
models.register('hf_causal_lm', func=ComposerHFCausalLM)
models.register('hf_t5', func=ComposerHFT5)
models.register('openai_causal_lm', func=OpenAICausalLMEvalWrapper)
models.register('fmapi_causal_lm', func=FMAPICasualLMEvalWrapper)
models.register('openai_chat', func=OpenAIChatAPIEvalWrapper)
models.register('fmapi_chat', func=FMAPIChatAPIEvalWrapper)
models.register('gemini_chat', func=GeminiChatAPIEvalrapper)
models.register('vllm_causal_lm', func=VLLMCausalLMEvalWrapper)

__all__ = [
    'ComposerHFCausalLM',
    'ComposerHFT5',
    'MPTConfig',
    'MPTPreTrainedModel',
    'MPTModel',
    'MPTForCausalLM',
    'ComposerMPTCausalLM',
    'OpenAICausalLMEvalWrapper',
    'FMAPICasualLMEvalWrapper',
    'OpenAIChatAPIEvalWrapper',
    'FMAPIChatAPIEvalWrapper',
]
