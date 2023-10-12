# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Tuple
from unittest.mock import patch

import pytest
import torch
from composer.core.precision import get_precision_context
from composer.utils import dist, get_device, reproducibility
from omegaconf import DictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from llmfoundry import COMPOSER_MODEL_REGISTRY
from llmfoundry.models.mpt.modeling_mpt import MPTForCausalLM
from llmfoundry.utils import build_tokenizer

EOS_TOKEN_ID = 0


class MockMPTForCausalLM(MPTForCausalLM):
    """Class that overrides the forward of MPTForCausalLM."""

    def forward(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[List[Tuple[torch.FloatTensor]]] = None,
        attention_mask: Optional[torch.ByteTensor] = None,
        prefix_mask: Optional[torch.ByteTensor] = None,
        sequence_id: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ):
        result = super().forward(input_ids, past_key_values, attention_mask,
                                 prefix_mask, sequence_id, labels, return_dict,
                                 output_attentions, output_hidden_states,
                                 use_cache, inputs_embeds)
        # Modify the logits to select the next token.
        if dist.get_global_rank() == 0:
            # Rank 0 hits EOS immediately.
            result.logits[:, :, EOS_TOKEN_ID] = torch.inf
        else:
            # Other ranks do not hit EOS.
            result.logits[:, :, EOS_TOKEN_ID] = -torch.inf
        return result


@pytest.mark.world_size(2)
@pytest.mark.gpu
@pytest.mark.parametrize('attn_impl', ['triton', 'torch'])
@pytest.mark.parametrize('use_alibi', [True, False])
@patch('llmfoundry.models.mpt.modeling_mpt.MPTForCausalLM',
       new=MockMPTForCausalLM)
def test_mpt_generate_multi_gpu(attn_impl: str, use_alibi: bool):
    """Tests mpt generation with mutiple gpus.

    and generations of different lengths.
    """
    composer_device = get_device('gpu')
    dist.initialize_dist(composer_device)
    reproducibility.seed_all(42)

    model_config = DictConfig({
        'name': 'mpt_causal_lm',
        'd_model': 128,
        'n_heads': 4,
        'n_layers': 2,
        'expansion_ratio': 2,
        'no_bias': False,
        'use_cache': True,
        'attn_config': {
            'attn_impl': attn_impl,
            'attn_uses_sequence_id': False,
            'alibi': use_alibi
        },
    })

    # build tokenizer
    tokenizer = build_tokenizer('EleutherAI/gpt-neox-20b', {})

    # build model
    model = COMPOSER_MODEL_REGISTRY[model_config.name](model_config, tokenizer)
    model = composer_device.module_to_device(model)
    model.eval()

    model.model = FSDP(model.model)

    with get_precision_context('amp_bf16'):
        _ = model.generate(composer_device.tensor_to_device(
            tokenizer('hello', return_tensors='pt')['input_ids']),
                           max_new_tokens=3,
                           eos_token_id=EOS_TOKEN_ID,
                           use_cache=True,
                           synced_gpus=True)
