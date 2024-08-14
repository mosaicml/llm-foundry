# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from llmfoundry.models.layers.attention import (
    ATTN_CLASS_REGISTRY, GroupedQueryAttention, MultiheadAttention,
    MultiQueryAttention, attn_bias_shape, build_alibi_bias, build_attn_bias,
    flash_attn_fn, scaled_multihead_dot_product_attention, triton_flash_attn_fn)
from llmfoundry.models.layers.blocks import MPTBlock
from llmfoundry.models.layers.custom_embedding import SharedEmbedding
from llmfoundry.models.layers.dmoe import DroplessMLP, LearnedRouter, dMoE
from llmfoundry.models.layers.fc import *
from llmfoundry.models.layers.ffn import MPTGLU, MPTMLP
from llmfoundry.models.layers.layer_builders import (
    build_attention_layer,
    build_fc,
    build_ffn,
    build_norm,
)
from llmfoundry.models.layers.norm import (
    LPLayerNorm,
    LPRMSNorm,
    RMSNorm,
    TritonRMSNorm,
    rms_norm,
)

__all__ = [
    'scaled_multihead_dot_product_attention',
    'flash_attn_fn',
    'MultiheadAttention',
    'MultiQueryAttention',
    'GroupedQueryAttention',
    'attn_bias_shape',
    'build_attn_bias',
    'build_alibi_bias',
    'check_alibi_support',
    'MPTBlock',
    'FusedNormAttentionNorm',
    'SharedEmbedding',
    'dMoE',
    'LearnedRouter',
    'DroplessMLP',
    'MPTMLP',
    'MPTGLU',
    'build_attention_layer',
    'build_ffn',
    'build_fc',
    'build_norm',
    'LPLayerNorm',
    'LPRMSNorm',
    'RMSNorm',
    'TritonRMSNorm',
    'rms_norm',
]
