# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

try:
    import torch

    from examples.llm.src.model_registry import COMPOSER_MODEL_REGISTRY
    from examples.llm.src.models.hf import (ComposerHFCausalLM,
                                            ComposerHFPrefixLM, ComposerHFT5)
    from examples.llm.src.models.layers.attention import (
        MultiheadAttention, alibi_bias, attn_bias, attn_bias_shape,
        flash_attn_fn, scaled_multihead_dot_product_attention,
        triton_flash_attn_fn)
    from examples.llm.src.models.layers.gpt_blocks import GPTMLP, GPTBlock
    from examples.llm.src.models.mosaic_gpt import (ComposerMosaicGPT,
                                                    MosaicGPT, MosaicGPTConfig)
except ImportError as e:
    try:
        is_cuda_available = torch.cuda.is_available()  # type: ignore
    except:
        is_cuda_available = False

    extras = '.[llm]' if is_cuda_available else '.[llm-cpu]'
    raise ImportError(
        f'Please make sure to pip install {extras} to get the requirements for the LLM example.'
    ) from e

__all__ = [
    'ComposerHFCausalLM',
    'ComposerHFPrefixLM',
    'ComposerHFT5',
    'COMPOSER_MODEL_REGISTRY',
    'scaled_multihead_dot_product_attention',
    'flash_attn_fn',
    'triton_flash_attn_fn',
    'MultiheadAttention',
    'attn_bias_shape',
    'attn_bias',
    'alibi_bias',
    'GPTMLP',
    'GPTBlock',
    'MosaicGPT',
    'MosaicGPTConfig',
    'ComposerMosaicGPT',
]
