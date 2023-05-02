# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

try:
    import torch

    from llmfoundry.data import (MixtureOfDenoisersCollator,
                                 Seq2SeqFinetuningCollator,
                                 build_finetuning_dataloader,
                                 build_text_denoising_dataloader)
    from llmfoundry.model_registry import COMPOSER_MODEL_REGISTRY
    from llmfoundry.models.hf import (ComposerHFCausalLM, ComposerHFPrefixLM,
                                      ComposerHFT5)
    from llmfoundry.models.layers.attention import (
        MultiheadAttention, attn_bias_shape, build_alibi_bias, build_attn_bias,
        flash_attn_fn, scaled_multihead_dot_product_attention,
        triton_flash_attn_fn)
    from llmfoundry.models.layers.gpt_blocks import GPTMLP, GPTBlock
    from llmfoundry.models.mosaic_gpt import (ComposerMosaicGPT, MosaicGPT,
                                              MosaicGPTConfig)

except ImportError as e:
    try:
        is_cuda_available = torch.cuda.is_available()  # type: ignore
    except:
        is_cuda_available = False

    extras = '.[gpu]' if is_cuda_available else '.'
    raise ImportError(
        f'Please make sure to pip install {extras} to get the requirements for the LLM example.'
    ) from e

__all__ = [
    'build_text_denoising_dataloader',
    'build_finetuning_dataloader',
    'flash_attn_fn',
    'triton_flash_attn_fn',
    'MixtureOfDenoisersCollator',
    'Seq2SeqFinetuningCollator',
    'ComposerHFCausalLM',
    'ComposerHFPrefixLM',
    'ComposerHFT5',
    'COMPOSER_MODEL_REGISTRY',
    'scaled_multihead_dot_product_attention',
    'MultiheadAttention',
    'attn_bias_shape',
    'attn_bias',
    'alibi_bias',
    'GPTMLP',
    'GPTBlock',
    'MosaicGPTConfig',
    'MosaicGPT',
    'ComposerMosaicGPT',
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

__version__ = '0.0.4'
