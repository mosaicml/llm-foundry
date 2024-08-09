# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Defaults for MPT model component configs."""

from typing import Dict

ffn_config_defaults: Dict = {
    'ffn_type': 'mptmlp',
}

attn_config_defaults: Dict = {
    'attn_type': 'multihead_attention',
    'attn_pdrop': 0.0,
    'attn_impl': 'flash',
    'qk_ln': False,
    'qk_gn': False,
    'fused_qkv': True,
    'clip_qkv': None,
    'softmax_scale': None,
    'attn_uses_sequence_id': False,
    'sliding_window_size': -1,
    'attn_logit_softcapping': None,
    'alibi': False,
    'alibi_bias_max': 8,
    'rope': False,
    'rope_theta': 10000,
    'rope_impl': 'dail',
    'rope_dail_config': {
        'type': 'original',
        'pos_idx_in_fp32': True,
        'xpos_scale_base': 512,
    },
    'rope_hf_config': {
        'type': 'no_scaling',
        'factor': 1.0,
    },
}

init_config_defaults: Dict = {
    'name': 'kaiming_normal_',
    'fan_mode': 'fan_in',
    'init_nonlinearity': 'relu',
    'init_div_is_residual': True,
    'emb_init_std': None,
    'emb_init_uniform_lim': None,
    'init_std': None,
    'init_gain': 0.0,
}

fc_type_defaults: Dict = {
    'name': 'torch',
}
