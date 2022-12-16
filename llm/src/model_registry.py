# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

from src.hf_causal_lm import ComposerHFCausalLM
from src.mosaic_gpt import ComposerMosaicGPT

COMPOSER_MODEL_REGISTRY = {
    'mosaic_gpt': ComposerMosaicGPT,
    'hf_causal_lm': ComposerHFCausalLM,
}
