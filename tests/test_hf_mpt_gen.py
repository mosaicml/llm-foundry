# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional, Tuple

import pytest
from composer.core.precision import get_precision_context
from composer.utils import get_device, reproducibility
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from composer.utils import dist

from llmfoundry import COMPOSER_MODEL_REGISTRY
from llmfoundry.utils import build_tokenizer
import torch

from transformers import AutoModelForCausalLM
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from unittest.mock import patch
from llmfoundry.models.mpt.modeling_mpt import MPTForCausalLM, MPTConfig

@pytest.mark.gpu
@pytest.mark.parametrize('device', ['cpu', 'gpu'])
@pytest.mark.parametrize('attn_impl', ['triton', 'torch'])
def test_init_hfhub_mpt(device: str, attn_impl: str):
    if device == 'cpu' and attn_impl == 'triton':
        pytest.skip(f'{attn_impl=} not implemented for {device=}.')
    composer_device = get_device(device)

    with open('scripts/train/yamls/pretrain/testing.yaml') as f:
        test_cfg = om.load(f)

    assert isinstance(test_cfg, DictConfig)
    reproducibility.seed_all(test_cfg.get('seed', 42))

    attn_uses_sequence_id = True if test_cfg.get('eos_token_id',
                                                 None) is not None else False
    test_cfg.model = DictConfig({
        'name': 'hf_causal_lm',
        'pretrained_model_name_or_path': 'mosaicml/mpt-7b',
        'pretrained': False,
        'config_overrides': {
            'd_model': 128,
            'n_heads': 4,
            'n_layers': 2,
            'expansion_ratio': 2,
            'attn_config': {
                'attn_impl': attn_impl,
                'attn_uses_sequence_id': attn_uses_sequence_id,
            },
        },
    })

    # build tokenizer
    tokenizer_cfg: Dict[str,
                        Any] = om.to_container(test_cfg.tokenizer,
                                               resolve=True)  # type: ignore
    tokenizer_name = tokenizer_cfg['name']
    tokenizer_kwargs = tokenizer_cfg.get('kwargs', {})
    tokenizer = build_tokenizer(tokenizer_name, tokenizer_kwargs)

    # build model
    model = COMPOSER_MODEL_REGISTRY[test_cfg.model.name](test_cfg.model,
                                                         tokenizer)
    test_cfg.n_params = sum(p.numel() for p in model.parameters())

    model.eval()
    model = composer_device.module_to_device(model)

    with get_precision_context('amp_bf16' if composer_device.name ==
                               'gpu' else 'fp32'):
        _ = model.generate(
            composer_device.tensor_to_device(
                tokenizer('hello', return_tensors='pt')['input_ids']),
            max_new_tokens=10,
        )


def test_init_hfhub_mpt_cpu():
    test_init_hfhub_mpt(device='cpu', attn_impl='torch')

EOS_TOKEN_ID = 0

class MockMPTForCausalLM(MPTForCausalLM):
    """Class that overrides the forward of MPTForCausalLM.
    """
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
        result = super().forward(input_ids, past_key_values, attention_mask, prefix_mask, sequence_id, labels, return_dict, output_attentions, output_hidden_states, use_cache, inputs_embeds)
        # Modify the logits to select the next token.
        if dist.get_global_rank() == 0:
            # Rank 0 hits EOS immediately.
            result.logits[:, :, EOS_TOKEN_ID] = torch.inf
        else:
            # Other ranks do not hit EOS.
            result.logits[:, :, EOS_TOKEN_ID] = -torch.inf
        return result

def mock_from_config(config: MPTConfig, **_):
    config_dict = config.to_dict()
    config = MPTConfig.from_dict(config_dict)
    return MockMPTForCausalLM._from_config(config)

@pytest.mark.world_size(2)
@pytest.mark.gpu
@patch.object(AutoModelForCausalLM, 'from_config', new=mock_from_config)
def test_mpt_generate_multi_gpu():
    """Tests mpt generation with mutiple gpus and
    generations of different lengths. 
    """
    composer_device = get_device('gpu')
    dist.initialize_dist(composer_device)
    with open('scripts/train/yamls/pretrain/testing.yaml') as f:
        test_cfg = om.load(f)

    assert isinstance(test_cfg, DictConfig)
    reproducibility.seed_all(test_cfg.get('seed', 42))

    test_cfg.model = DictConfig({
        'name': 'hf_causal_lm',
        'pretrained_model_name_or_path': 'mosaicml/mpt-7b',
        'pretrained': False,
        'config_overrides': {
            'd_model': 128,
            'n_heads': 4,
            'n_layers': 2,
            'expansion_ratio': 2,
            'no_bias': False,
            'use_cache': False
        }
    })

    # build tokenizer
    tokenizer_name = test_cfg.tokenizer.name
    tokenizer = build_tokenizer(tokenizer_name, {'max_seq_len': 15}) 
  
    # build model
    model = COMPOSER_MODEL_REGISTRY[test_cfg.model.name](test_cfg.model,
                                                        tokenizer)
    model = composer_device.module_to_device(model)

    model.model = FSDP(model.model)

    _ = model.generate(
        composer_device.tensor_to_device(
            tokenizer('hello', return_tensors='pt')['input_ids']),
        max_new_tokens=10,
        eos_token_id=EOS_TOKEN_ID,
        use_cache=True,
        synced_gpus=True
    )
