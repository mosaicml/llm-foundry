# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import torch

try:
    from llmfoundry import optim, utils
    from llmfoundry.data import (ConcatTokensDataset,
                                 MixtureOfDenoisersCollator, NoConcatDataset,
                                 Seq2SeqFinetuningCollator,
                                 build_finetuning_dataloader,
                                 build_text_denoising_dataloader)
    from llmfoundry.models.hf import (ComposerHFCausalLM, ComposerHFPrefixLM,
                                      ComposerHFT5)
    from llmfoundry.models.layers.attention import (
        MultiheadAttention, attn_bias_shape, build_alibi_bias, build_attn_bias,
        flash_attn_fn, scaled_multihead_dot_product_attention,
        triton_flash_attn_fn)
    from llmfoundry.models.layers.blocks import MPTMLP, MPTBlock
    from llmfoundry.models.model_registry import COMPOSER_MODEL_REGISTRY
    from llmfoundry.models.mpt import (ComposerMPTCausalLM, MPTConfig,
                                       MPTForCausalLM, MPTModel,
                                       MPTPreTrainedModel)

except ImportError as e:
    try:
        is_cuda_available = torch.cuda.is_available()
    except:
        is_cuda_available = False

    extras = '.[gpu]' if is_cuda_available else '.'
    raise ImportError(
        f'Please make sure to pip install {extras} to get the requirements for the LLM example.'
    ) from e

__all__ = [
    'build_text_denoising_dataloader',
    'build_finetuning_dataloader',
    'MixtureOfDenoisersCollator',
    'Seq2SeqFinetuningCollator',
    'MPTMLP',
    'MPTBlock',
    'MPTConfig',
    'MPTPreTrainedModel',
    'MPTModel',
    'MPTForCausalLM',
    'ComposerMPTCausalLM',
    'ComposerHFCausalLM',
    'ComposerHFPrefixLM',
    'ComposerHFT5',
    'COMPOSER_MODEL_REGISTRY',
    'scaled_multihead_dot_product_attention',
    'flash_attn_fn',
    'triton_flash_attn_fn',
    'MultiheadAttention',
    'NoConcatDataset',
    'ConcatTokensDataset',
    'attn_bias_shape',
    'build_attn_bias',
    'build_alibi_bias',
    'optim',
    'utils',
]

__version__ = '0.2.0'
