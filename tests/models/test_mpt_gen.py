# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, List, Optional, Tuple
from unittest.mock import Mock, patch

import pytest
import torch
from composer import Trainer
from composer.callbacks import Generate as ComposerGenerate
from composer.core.precision import get_precision_context
from composer.utils import dist, get_device
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from llmfoundry.models.mpt.modeling_mpt import (ComposerMPTCausalLM,
                                                MPTForCausalLM)

EOS_TOKEN_ID = 0


class MockMPTForCausalLM(MPTForCausalLM):
    """Class that overrides the forward of MPTForCausalLM."""

    def forward(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[List[Tuple[torch.FloatTensor]]] = None,
        attention_mask: Optional[torch.ByteTensor] = None,
        sequence_id: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ):
        result = super().forward(input_ids, past_key_values, attention_mask,
                                 sequence_id, labels, return_dict,
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
@pytest.mark.parametrize('attn_impl', ['flash', 'torch'])
@pytest.mark.parametrize('use_alibi', [True, False])
@pytest.mark.parametrize('tie_word_embeddings', [True, False])
@patch('llmfoundry.models.mpt.modeling_mpt.MPTForCausalLM',
       new=MockMPTForCausalLM)
def test_mpt_generate_multi_gpu(attn_impl: str, use_alibi: bool,
                                tie_word_embeddings: bool,
                                build_tiny_mpt: Callable[...,
                                                         ComposerMPTCausalLM],
                                mpt_tokenizer: PreTrainedTokenizerBase):
    """Tests mpt generation with mutiple gpus.

    and generations of different lengths.
    """
    device = get_device('gpu')

    model = build_tiny_mpt(
        tie_word_embeddings=tie_word_embeddings,
        attn_config={
            'attn_impl': attn_impl,
            'attn_uses_sequence_id': False,
            'alibi': use_alibi
        },
    )
    model = device.module_to_device(model)

    model.eval()

    model.model = FSDP(model.model)

    with get_precision_context('amp_bf16'):
        _ = model.generate(device.tensor_to_device(
            mpt_tokenizer('hello', return_tensors='pt')['input_ids']),
                           max_new_tokens=3,
                           eos_token_id=EOS_TOKEN_ID,
                           use_cache=True,
                           synced_gpus=True)


@pytest.mark.gpu
@pytest.mark.parametrize('attn_impl', ['flash', 'torch'])
@pytest.mark.parametrize('use_alibi', [True, False])
def test_mpt_generate_callback(attn_impl: str, use_alibi: bool,
                               build_tiny_mpt: Callable[...,
                                                        ComposerMPTCausalLM],
                               tiny_ft_dataloader: DataLoader):
    device = get_device('gpu')

    # build mpt model
    model = build_tiny_mpt(
        tie_word_embeddings=True,
        attn_config={
            'attn_impl': attn_impl,
            'attn_uses_sequence_id': False,
            'alibi': use_alibi
        },
    )
    model = device.module_to_device(model)

    # generate callback
    prompts = [
        'The best banana bread recipe is',
        '2+2=',
        'how much wood could a woodchuck chuck',
    ]
    gen_interval = 1
    generate = ComposerGenerate(
        prompts,
        interval=f'{gen_interval}ba',
        max_new_tokens=5,
        batch_size=len(prompts),
        use_cache=True,
    )
    generate.generate = Mock(wraps=generate.generate, autospec=True)

    # build trainer
    trainer = Trainer(
        model=model,
        train_dataloader=tiny_ft_dataloader,
        device=device,
        max_duration=f'{gen_interval}ba',
        callbacks=[generate],
    )
    trainer.logger.log_table = Mock()
    trainer.fit()

    generate.generate.assert_called_once()
    trainer.logger.log_table.assert_called_once()


@pytest.mark.gpu
@pytest.mark.parametrize('attn_impl', ['flash', 'torch'])
@pytest.mark.parametrize('use_alibi', [True, False])
def test_mpt_generate_callback_not_tied(
        use_alibi: bool, attn_impl: str,
        build_tiny_mpt: Callable[..., ComposerMPTCausalLM],
        tiny_ft_dataloader: DataLoader):
    device = get_device('gpu')

    # build mpt model
    model = build_tiny_mpt(
        tie_word_embeddings=False,
        attn_config={
            'attn_impl': attn_impl,
            'attn_uses_sequence_id': False,
            'alibi': use_alibi,
        },
    )
    model = device.module_to_device(model)

    # generate callback
    prompts = [
        'The best banana bread recipe is',
        '2+2=',
        'how much wood could a woodchuck chuck',
    ]
    gen_interval = 1
    generate = ComposerGenerate(
        prompts,
        interval=f'{gen_interval}ba',
        max_new_tokens=5,
        batch_size=len(prompts),
        use_cache=True,
    )
    generate.generate = Mock(wraps=generate.generate, autospec=True)

    # build trainer
    trainer = Trainer(
        model=model,
        train_dataloader=tiny_ft_dataloader,
        device=device,
        max_duration=f'{gen_interval}ba',
        callbacks=[generate],
    )
    trainer.logger.log_table = Mock()
    trainer.fit()

    generate.generate.assert_called_once()
    trainer.logger.log_table.assert_called_once()
