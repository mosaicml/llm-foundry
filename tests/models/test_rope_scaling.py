# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

from llmfoundry.models.mpt.modeling_mpt import gen_rotary_embedding

rope_config = {
    'rope_theta': 500000.0,
    'rope_impl': 'hf',
    'rope_hf_config': {
        'factor': 8.0,
        'low_freq_factor': 1.0,
        'high_freq_factor': 4.0,
        'original_max_position_embeddings': 8192,
        'type': 'llama3',
    },
}

rope_dail_config = {}


def test_rope_scaling():
    d_model = 128
    n_heads = 32
    max_seq_len = 65536

    embedding = gen_rotary_embedding(
        d_model=d_model,
        n_heads=n_heads,
        rope_dail_config=rope_dail_config,
        max_seq_len=max_seq_len,
        **rope_config,
    )

    assert isinstance(embedding, LlamaRotaryEmbedding)
