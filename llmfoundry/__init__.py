# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import warnings

# bitsandbytes is a very noisy library. A lot of it is print statements that we can't easily suppress,
# but we can at least suppress a bunch of spurious warnings.
warnings.filterwarnings('ignore', category=UserWarning, module='bitsandbytes')

import logging

from llmfoundry.utils.logging_utils import SpecificWarningFilter

# Filter out Hugging Face warning for not using a pinned revision of the model
hf_dynamic_modules_logger = logging.getLogger(
    'transformers.dynamic_module_utils')
new_files_warning_filter = SpecificWarningFilter(
    'A new version of the following files was downloaded from')

hf_dynamic_modules_logger.addFilter(new_files_warning_filter)

# Before importing any transformers models, we need to disable transformers flash attention if
# we are in an environment with flash attention version <2. Transformers hard errors on a not properly
# gated import otherwise.
import transformers

from llmfoundry import optim, utils
from llmfoundry.data import (ConcatTokensDataset, MixtureOfDenoisersCollator,
                             NoConcatDataset, Seq2SeqFinetuningCollator,
                             build_finetuning_dataloader,
                             build_text_denoising_dataloader)
from llmfoundry.models.hf import (ComposerHFCausalLM, ComposerHFPrefixLM,
                                  ComposerHFT5)
from llmfoundry.models.layers.attention import (
    MultiheadAttention, attn_bias_shape, build_alibi_bias, build_attn_bias,
    flash_attn_fn, is_flash_v1_installed,
    scaled_multihead_dot_product_attention, triton_flash_attn_fn)
from llmfoundry.models.layers.blocks import MPTBlock
from llmfoundry.models.layers.ffn import FFN_CLASS_REGISTRY, MPTMLP, build_ffn
from llmfoundry.models.model_registry import COMPOSER_MODEL_REGISTRY
from llmfoundry.models.mpt import (ComposerMPTCausalLM, MPTConfig,
                                   MPTForCausalLM, MPTModel, MPTPreTrainedModel)
from llmfoundry.tokenizers import TiktokenTokenizerWrapper
if is_flash_v1_installed():
    transformers.utils.is_flash_attn_available = lambda: False

__all__ = [
    'build_text_denoising_dataloader',
    'build_finetuning_dataloader',
    'MixtureOfDenoisersCollator',
    'Seq2SeqFinetuningCollator',
    'MPTBlock',
    'FFN_CLASS_REGISTRY',
    'MPTMLP',
    'build_ffn',
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
    'TiktokenTokenizerWrapper',
]

__version__ = '0.5.0'
