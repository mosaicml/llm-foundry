# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

from examples.llm.src.models.hf import (ComposerHFCausalLM, ComposerHFPrefixLM,
                                        ComposerHFT5)
from examples.llm.src.models.mosaic_gpt import ComposerMosaicGPT

COMPOSER_MODEL_REGISTRY = {
    'mosaic_gpt': ComposerMosaicGPT,
    'hf_causal_lm': ComposerHFCausalLM,
    'hf_prefix_lm': ComposerHFPrefixLM,
    'hf_t5': ComposerHFT5,
}
