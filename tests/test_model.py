# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import copy
import gc
import os
import warnings
from typing import cast
from unittest import mock

import pytest
import torch
import torch.nn as nn
from accelerate import init_empty_weights
from composer.core.precision import Precision, get_precision_context
from composer.optim import DecoupledAdamW
from composer.trainer.dist_strategy import prepare_fsdp_module
from composer.utils import dist, get_device, reproducibility
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedTokenizer, PreTrainedTokenizerFast,
                          pipeline)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.bloom.modeling_bloom import build_alibi_tensor

from llmfoundry import (COMPOSER_MODEL_REGISTRY, ComposerHFCausalLM,
                        ComposerHFPrefixLM)
from llmfoundry.models.hf.model_wrapper import HuggingFaceModelWithZLoss
from llmfoundry.models.layers import NORM_CLASS_REGISTRY, build_alibi_bias
from llmfoundry.models.layers.blocks import MPTBlock
from llmfoundry.models.mpt import MPTConfig, MPTForCausalLM
from llmfoundry.utils import build_tokenizer


def get_config(
        conf_path='scripts/train/yamls/pretrain/testing.yaml') -> DictConfig:
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    print(conf_path)
    with open(conf_path) as f:
        test_cfg = om.load(f)
    return cast(DictConfig, test_cfg)


def get_objs(conf_path='scripts/train/yamls/pretrain/testing.yaml'):
    warnings.filterwarnings(
        action='ignore',
        message='Torchmetrics v0.9 introduced a new argument class property')
    test_cfg = get_config(conf_path=conf_path)

    reproducibility.seed_all(test_cfg.seed)

    # Read FSDP Config as a dict
    fsdp_config = test_cfg.get('fsdp_config', None)
    fsdp_config = om.to_container(fsdp_config,
                                  resolve=True) if fsdp_config else None

    # Build Model
    # For fast initialization, use `meta` device
    print('Initializing model...')
    device = 'cpu'
    test_cfg.precision = 'fp32'
    test_cfg.model.attn_config = {
        'attn_impl': 'torch',
    }
    # device = 'cuda'
    # test_cfg.precision = 'amp'
    test_cfg.model.init_device = device
    test_cfg.device = device

    test_cfg.global_train_batch_size = 2
    test_cfg.device_eval_batch_size = 2
    test_cfg.device_train_microbatch_size = 2

    tokenizer = build_tokenizer(test_cfg.tokenizer)

    model = COMPOSER_MODEL_REGISTRY[test_cfg.model.name](test_cfg.model,
                                                         tokenizer)
    # Optimizer
    assert test_cfg.optimizer.name == 'decoupled_adamw'
    optimizer = DecoupledAdamW(model.parameters(),
                               lr=test_cfg.optimizer.lr,
                               betas=test_cfg.optimizer.betas,
                               eps=test_cfg.optimizer.eps,
                               weight_decay=test_cfg.optimizer.weight_decay)

    return test_cfg, model, optimizer


def gen_random_batch(batch_size, test_cfg):
    # generate input batch of random data, suitable for a Causal or Prefix LM
    batch = {}
    batch['input_ids'] = torch.randint(
        low=0,
        high=test_cfg.model.vocab_size,
        size=(batch_size, test_cfg.max_seq_len)).to(test_cfg.device)
    batch['labels'] = torch.randint(low=0,
                                    high=test_cfg.model.vocab_size,
                                    size=(batch_size, test_cfg.max_seq_len)).to(
                                        test_cfg.device)
    batch['attention_mask'] = torch.ones(size=(batch_size,
                                               test_cfg.max_seq_len),
                                         dtype=torch.int64).to(test_cfg.device)
    batch['bidirectional_mask'] = batch['attention_mask'].clone()
    batch['bidirectional_mask'][:, (test_cfg.max_seq_len // 2):] = 0
    return batch


def gen_random_enc_dec_batch(batch_size, vocab_size, max_seq_len, device):
    # generate input batch of random data, suitable for a T5
    batch = {}
    batch['input_ids'] = torch.randint(low=0,
                                       high=vocab_size,
                                       size=(batch_size,
                                             max_seq_len)).to(device)
    batch['labels'] = torch.randint(low=0,
                                    high=vocab_size,
                                    size=(batch_size, max_seq_len)).to(device)
    batch['decoder_input_ids'] = torch.zeros_like(batch['labels'])
    batch['decoder_input_ids'][:, 1:] = batch['labels'][:, :-1]
    batch['attention_mask'] = torch.ones(size=(batch_size, max_seq_len),
                                         dtype=torch.int64).to(device)
    batch['decoder_attention_mask'] = batch['attention_mask'].clone()
    return batch


def test_full_forward_and_backward(batch_size=2):
    test_cfg, model, optimizer = get_objs(
        conf_path='scripts/train/yamls/pretrain/testing.yaml')

    batch = gen_random_batch(batch_size, test_cfg)

    assert batch['input_ids'].shape == torch.Size(
        [batch_size, test_cfg.max_seq_len])
    model.train()
    original_params = next(model.parameters()).clone().data
    outputs = model(batch)
    loss = model.loss(outputs, batch)
    loss.backward()
    optimizer.step()
    updated_params = next(model.parameters()).clone().data
    assert not torch.equal(original_params, updated_params)


def test_attention_mechanism(batch_size=2):
    test_cfg, model, _ = get_objs(
        conf_path='scripts/train/yamls/pretrain/testing.yaml')

    batch = gen_random_batch(batch_size, test_cfg)

    model.eval()
    # run a partial forward where we explicitly inspect the attention_mask from the causal_attn block
    input_ids, attention_mask = batch['input_ids'], batch[
        'attention_mask'].bool()

    _, S = input_ids.size()
    assert (
        S <= test_cfg.max_seq_len
    ), f'Cannot forward input with seq_len={S}, this model only supports seq_len<={test_cfg.max_seq_len}'
    pos = torch.arange(0, S, dtype=torch.long,
                       device=input_ids.device).unsqueeze(0)

    tok_emb = model.model.transformer.wte(input_ids)
    pos_emb = model.model.transformer.wpe(pos)
    x = model.model.transformer.emb_drop(tok_emb + pos_emb)

    # basically the attention mask should be a tensor shape (bsz, seqlen, seqlen)
    # wih -inf along the upper triangle as well as wherever there are any pad tokens
    # and with 0 everywhere else
    expected_zerod_weights = nn.Transformer.generate_square_subsequent_mask(test_cfg.max_seq_len)\
        .reshape(1, test_cfg.max_seq_len, test_cfg.max_seq_len)
    expected_zerod_weights = torch.isneginf(  # type: ignore
        torch.cat(batch_size * [expected_zerod_weights]))
    torch_key_padding = torch.cat(  # type: ignore
        test_cfg.max_seq_len *
        [(~attention_mask).reshape(batch_size, 1, test_cfg.max_seq_len)],
        axis=1)
    expected_zerod_weights |= torch_key_padding

    attn_bias, attention_mask = model.model.transformer._attn_bias(
        device=x.device, dtype=x.dtype, attention_mask=attention_mask)

    for block in model.model.transformer.blocks:
        a = block.norm_1(x)
        b, attention_weights, _ = block.attn(
            a,
            past_key_value=None,
            attn_bias=attn_bias,
            attention_mask=attention_mask,
            is_causal=model.model.transformer.is_causal,
            needs_weights=True)

        zerod_weights = (attention_weights == 0)
        assert torch.equal(expected_zerod_weights.expand(*zerod_weights.shape),
                           zerod_weights)
        x = x + block.resid_attn_dropout(b)
        m = block.norm_2(x)
        n = block.ffn(m)
        x = x + block.resid_ffn_dropout(n)


@pytest.mark.parametrize('prefixlm', [False, True])
def test_full_forward_and_backward_gpt2_small(prefixlm, batch_size=2):
    warnings.filterwarnings(
        action='ignore',
        message='Torchmetrics v0.9 introduced a new argument class property')
    conf_path = 'scripts/train/yamls/pretrain/gpt2-small.yaml'
    with open(conf_path) as f:
        neo_cfg = om.load(f)

    device = 'cpu'
    neo_cfg.device = device
    neo_cfg.max_seq_len = 256

    if prefixlm:
        neo_cfg.model.name = 'hf_prefix_lm'
    else:
        neo_cfg.model.name = 'hf_causal_lm'

    tokenizer = build_tokenizer(neo_cfg.tokenizer)

    model = COMPOSER_MODEL_REGISTRY[neo_cfg.model.name](neo_cfg.model,
                                                        tokenizer).to(device)

    assert isinstance(model.tokenizer,
                      (PreTrainedTokenizer, PreTrainedTokenizerFast))

    assert neo_cfg.optimizer.name == 'decoupled_adamw'
    optimizer = DecoupledAdamW(model.parameters(),
                               lr=neo_cfg.optimizer.lr,
                               betas=neo_cfg.optimizer.betas,
                               eps=neo_cfg.optimizer.eps,
                               weight_decay=neo_cfg.optimizer.weight_decay)

    # set vocab size using model num_embeddings
    neo_cfg.model.vocab_size = model.model.transformer.wte.num_embeddings
    batch = gen_random_batch(batch_size, neo_cfg)

    assert batch['input_ids'].shape == torch.Size(
        [batch_size, neo_cfg.max_seq_len])
    model.train()
    original_params = next(model.parameters()).clone().data
    outputs = model(batch)
    loss = model.loss(outputs, batch)
    loss.backward()
    optimizer.step()
    updated_params = next(model.parameters()).clone().data
    assert not torch.equal(original_params, updated_params)


def test_full_forward_and_backward_t5_small(batch_size=2):
    warnings.filterwarnings(
        action='ignore',
        message='Torchmetrics v0.9 introduced a new argument class property')
    conf_path = 'scripts/train/yamls/finetune/t5-small_dolly_sft.yaml'
    with open(conf_path) as f:
        t5_cfg = om.load(f)

    device = 'cpu'
    t5_cfg.device = device
    t5_cfg.max_seq_len = 16

    tokenizer = build_tokenizer(t5_cfg.tokenizer)

    model = COMPOSER_MODEL_REGISTRY[t5_cfg.model.name](t5_cfg.model,
                                                       tokenizer).to(device)

    assert isinstance(model.tokenizer,
                      (PreTrainedTokenizer, PreTrainedTokenizerFast))

    optimizer = DecoupledAdamW(model.parameters(),
                               lr=t5_cfg.optimizer.lr,
                               betas=t5_cfg.optimizer.betas,
                               eps=t5_cfg.optimizer.eps,
                               weight_decay=t5_cfg.optimizer.weight_decay)

    # set vocab size using model num_embeddings
    batch = gen_random_enc_dec_batch(batch_size, model.model.config.vocab_size,
                                     t5_cfg.max_seq_len, device)

    assert batch['input_ids'].shape == torch.Size(
        [batch_size, t5_cfg.max_seq_len])
    model.train()
    original_params = next(model.parameters()).clone().data
    outputs = model(batch)
    loss = model.loss(outputs, batch)
    loss.backward()
    optimizer.step()
    updated_params = next(model.parameters()).clone().data
    assert not torch.equal(original_params, updated_params)


@pytest.mark.parametrize(
    'attn_impl,precision',
    [('torch', torch.float16), ('torch', torch.bfloat16),
     pytest.param('flash', torch.float16, marks=pytest.mark.gpu),
     pytest.param('flash', torch.bfloat16, marks=pytest.mark.gpu)])
def test_determinism(attn_impl: str, precision):
    if not torch.cuda.is_available():
        pytest.skip(
            'This test requires CUDA to be available in order to run with bfloat16 precision.'
        )
    reproducibility.seed_all(1111)

    conf_path = 'scripts/train/yamls/pretrain/testing.yaml'
    with open(conf_path) as f:
        test_cfg = om.load(f)

    test_cfg.model.attn_config = {
        'attn_impl': attn_impl,
    }
    test_cfg.model.init_device = 'cuda:0'
    test_cfg.device = 'cuda:0'

    tokenizer = build_tokenizer(test_cfg.tokenizer)

    model_1 = COMPOSER_MODEL_REGISTRY[test_cfg.model.name](test_cfg.model,
                                                           tokenizer)
    model_2 = copy.deepcopy(model_1)

    optimizer_1 = DecoupledAdamW(model_1.parameters(),
                                 lr=test_cfg.optimizer.lr,
                                 betas=test_cfg.optimizer.betas,
                                 eps=test_cfg.optimizer.eps,
                                 weight_decay=test_cfg.optimizer.weight_decay)
    optimizer_2 = DecoupledAdamW(model_2.parameters(),
                                 lr=test_cfg.optimizer.lr,
                                 betas=test_cfg.optimizer.betas,
                                 eps=test_cfg.optimizer.eps,
                                 weight_decay=test_cfg.optimizer.weight_decay)

    for i in range(5):
        with torch.cuda.amp.autocast(True, precision):
            batch = gen_random_batch(2, test_cfg)
            output_1 = model_1(batch)
            output_2 = model_2(batch)
            assert output_1.logits.allclose(output_2.logits, rtol=0.0,
                                            atol=0.0), f'differed at step {i}'

            loss_1 = model_1.loss(output_1, batch)
            loss_2 = model_2.loss(output_2, batch)
            assert loss_1 == loss_2
            loss_1.backward()
            loss_2.backward()
            optimizer_1.step()
            optimizer_2.step()


@pytest.mark.gpu
def test_loss_fn():
    """Tests the Fused CrossEntropy vs torch.nn.CrossEntropy loss function.

    We provide non-zero tolerances to account for small numerics differences
    between the two loss implementations.
    """
    try:
        from flash_attn.losses.cross_entropy import CrossEntropyLoss as FusedCrossEntropyLoss  # type: ignore # isort: skip
    except:
        pytest.skip('Fused cross entropy was not installed')

    # run numerical test in pure fp32
    torch.backends.cuda.matmul.allow_tf32 = False  # type: ignore (third-party)
    torch.backends.cudnn.allow_tf32 = False  # type: ignore (third-party)

    conf_path = 'scripts/train/yamls/pretrain/testing.yaml'
    with open(conf_path) as f:
        test_cfg = om.load(f)

    assert isinstance(test_cfg, DictConfig)

    test_cfg.device = 'cuda:0'
    test_cfg.model.init_device = 'cuda:0'
    test_cfg.model.init_config = {
        'name': 'baseline_',
        'init_std': 0.02,
    }

    reproducibility.seed_all(test_cfg.get('global_seed', 42))

    tokenizer = build_tokenizer(test_cfg.tokenizer)

    model_1 = COMPOSER_MODEL_REGISTRY[test_cfg.model.name](test_cfg.model,
                                                           tokenizer)
    model_2 = copy.deepcopy(model_1)
    assert isinstance(model_1.loss_fn, torch.nn.CrossEntropyLoss)
    model_2.loss_fn = FusedCrossEntropyLoss(ignore_index=-100)

    optimizer_1 = DecoupledAdamW(model_1.parameters(),
                                 lr=test_cfg.optimizer.lr,
                                 betas=test_cfg.optimizer.betas,
                                 eps=test_cfg.optimizer.eps,
                                 weight_decay=test_cfg.optimizer.weight_decay)
    optimizer_2 = DecoupledAdamW(model_2.parameters(),
                                 lr=test_cfg.optimizer.lr,
                                 betas=test_cfg.optimizer.betas,
                                 eps=test_cfg.optimizer.eps,
                                 weight_decay=test_cfg.optimizer.weight_decay)

    for i in range(15):
        batch = gen_random_batch(2, test_cfg)
        output_1 = model_1(batch)
        output_2 = model_2(batch)
        assert output_1.logits.allclose(output_2.logits, rtol=1e-4,
                                        atol=1e-4), f'differed at step {i}'

        loss_1 = model_1.loss(output_1, batch)
        loss_2 = model_2.loss(output_2, batch)
        assert loss_1.allclose(loss_2, rtol=1e-3,
                               atol=1e-3), f'differed at step {i}'
        loss_1.backward()
        loss_2.backward()
        optimizer_1.step()
        optimizer_2.step()

        for p1, p2 in zip(model_1.parameters(), model_2.parameters()):
            assert p1.data.shape == p2.data.shape
            assert p1.data.allclose(p2.data, rtol=1e-5,
                                    atol=1e-4), f'differed at step {i}'


@pytest.mark.parametrize('prefixlm', [False, True])
def test_opt_wrapping(prefixlm):
    conf = {
        'model': {
            'name': 'hf_prefix_lm' if prefixlm else 'hf_causal_lm',
            'pretrained_model_name_or_path': 'facebook/opt-125m',
            'pretrained': 'false'
        },
        'tokenizer': {
            'name': 'facebook/opt-125m'
        }
    }
    config = DictConfig(conf)

    tokenizer = build_tokenizer(config.tokenizer)

    if prefixlm:
        model = ComposerHFPrefixLM(config.model, tokenizer)
    else:
        model = ComposerHFCausalLM(config.model, tokenizer)

    # check that all the modules we except are blocked from FSDP wrapping
    assert not model.model.model._fsdp_wrap
    assert not model.model.model.decoder._fsdp_wrap
    assert not model.model.model.decoder.embed_tokens._fsdp_wrap
    assert not model.model.lm_head._fsdp_wrap


@pytest.mark.parametrize('norm_type', NORM_CLASS_REGISTRY.keys())
@pytest.mark.parametrize('no_bias', [False, True])
def test_mpt_creation(norm_type, no_bias):
    # Test that the config constructs the model as expected.
    hf_config = MPTConfig(
        init_device='cpu',
        d_model=128,
        n_heads=4,
        n_layers=2,
        expansion_ratio=2,
        max_seq_len=2048,
        emb_pdrop=0.1,
        resid_pdrop=0.2,
        attn_config={
            'attn_impl': 'torch',
        },
        norm_type=norm_type,
        no_bias=no_bias,
    )
    mpt = MPTForCausalLM(hf_config)

    assert mpt.config.d_model == 128
    assert mpt.config.n_heads == 4
    assert mpt.config.n_layers == 2
    assert mpt.config.expansion_ratio == 2
    assert mpt.config.max_seq_len == 2048

    assert mpt.transformer.wte.weight.shape == torch.Size(
        [hf_config.vocab_size, hf_config.d_model])
    assert mpt.transformer.wpe.weight.shape == torch.Size(
        [hf_config.max_seq_len, hf_config.d_model])
    assert mpt.transformer.emb_drop.p == 0.1
    assert len(mpt.transformer.blocks) == 2

    d_model = hf_config.d_model
    for block in mpt.transformer.blocks:
        assert isinstance(block, MPTBlock)
        assert block.norm_1.weight.shape == torch.Size([d_model])
        assert block.norm_2.weight.shape == torch.Size([d_model])
        assert block.ffn.up_proj.weight.shape == torch.Size(
            [hf_config.d_model * hf_config.expansion_ratio, hf_config.d_model])
        assert block.ffn.down_proj.weight.shape == torch.Size(
            [hf_config.d_model, hf_config.d_model * hf_config.expansion_ratio])
        assert block.resid_attn_dropout.p == 0.2
        assert block.resid_ffn_dropout.p == 0.2


@pytest.mark.parametrize('attention_impl,device', [('torch', 'cpu'),
                                                   ('flash', 'gpu'),
                                                   ('triton', 'gpu'),
                                                   ('torch', 'gpu')])
@pytest.mark.parametrize('alibi', [True, False])
def test_forward_with_padding(attention_impl, device, alibi):
    # Test that different placement of padding does not affect the output.
    if not torch.cuda.is_available() and device == 'gpu':
        pytest.skip(
            f'This test requires CUDA to be available in order to run with {attention_impl} attention.'
        )
    if alibi and attention_impl == 'flash':
        pytest.skip(f'alibi only implemented with torch and triton attention.')

    reproducibility.seed_all(1234)
    device = get_device(device)

    hf_config = MPTConfig(
        init_device='cpu',
        d_model=128,
        n_heads=1,
        n_layers=2,
        expansion_ratio=2,
        max_seq_len=2048,
        emb_pdrop=0.1,
        resid_pdrop=0.2,
        attn_config={
            'attn_impl': attention_impl,
            'alibi': alibi,
        },
        init_config={
            'name': 'baseline_',
            'init_std': 0.02,
        },
    )
    mpt = MPTForCausalLM(hf_config)
    mpt.eval()
    mpt = device.module_to_device(mpt)

    with get_precision_context('amp_bf16' if device.name == 'gpu' else 'fp32'):
        # padding on the right side of the input
        right_padding_input_ids = torch.tensor(
            [[11274, 16390, 11, 50256, 50256, 50256],
             [11274, 16390, 11, 50256, 50256, 50256]])
        right_padding_input_ids = device.tensor_to_device(
            right_padding_input_ids)
        right_padding_attention_mask = torch.tensor([[1, 1, 1, 0, 0, 0],
                                                     [1, 1, 1, 0, 0,
                                                      0]]).bool()
        right_padding_attention_mask = device.tensor_to_device(
            right_padding_attention_mask)

        # padding in the middle of the input
        middle_padding_input_ids = torch.tensor(
            [[11274, 16390, 50256, 50256, 50256, 11],
             [11274, 16390, 50256, 50256, 50256, 11]])
        middle_padding_input_ids = device.tensor_to_device(
            middle_padding_input_ids)
        middle_padding_attention_mask = torch.tensor([[1, 1, 0, 0, 0, 1],
                                                      [1, 1, 0, 0, 0,
                                                       1]]).bool()
        middle_padding_attention_mask = device.tensor_to_device(
            middle_padding_attention_mask)

        # padding on the left side of the input
        left_padding_input_ids = torch.tensor(
            [[50256, 50256, 50256, 11274, 16390, 11],
             [50256, 50256, 50256, 11274, 16390, 11]])
        left_padding_input_ids = device.tensor_to_device(left_padding_input_ids)
        left_padding_attention_mask = torch.tensor([[0, 0, 0, 1, 1, 1],
                                                    [0, 0, 0, 1, 1, 1]]).bool()
        left_padding_attention_mask = device.tensor_to_device(
            left_padding_attention_mask)

        # a single batch with padding in different places
        batched_input_ids = torch.tensor([
            [11274, 16390, 11, 50256, 50256, 50256],  # right padding
            [11274, 16390, 50256, 50256, 50256, 11]
        ])  # middle padding
        batched_input_ids = device.tensor_to_device(batched_input_ids)
        batched_attention_mask = torch.tensor([[1, 1, 1, 0, 0, 0],
                                               [1, 1, 0, 0, 0, 1]]).bool()
        batched_attention_mask = device.tensor_to_device(batched_attention_mask)

        right_padding_output = mpt(
            right_padding_input_ids,
            attention_mask=right_padding_attention_mask).logits
        middle_padding_output = mpt(
            middle_padding_input_ids,
            attention_mask=middle_padding_attention_mask).logits
        left_padding_output = mpt(
            left_padding_input_ids,
            attention_mask=left_padding_attention_mask).logits
        batched_output = mpt(batched_input_ids,
                             attention_mask=batched_attention_mask).logits

        # check that right padding and left padding produce the same output
        assert torch.allclose(right_padding_output[0, :3],
                              left_padding_output[0, 3:],
                              atol=1e-6 if attention_impl == 'torch' else 1e-8)
        if not alibi:
            # check that right padding and middle padding produce the same output
            # Note: alibi not implemented for middle padding.
            assert torch.allclose(
                right_padding_output[0, :3],
                middle_padding_output[0, [0, 1, 5]],
                atol=1e-6 if attention_impl == 'torch' else 1e-8)
        # check that right padding and right padding in a batch produce the same output
        assert torch.allclose(right_padding_output[0, :3],
                              batched_output[0, :3],
                              atol=1e-6 if attention_impl == 'torch' else 1e-8)
        if not alibi:
            # check that middle padding and middle padding in a batch produce the same output
            # Note: alibi not implemented for middle padding.
            assert torch.allclose(
                middle_padding_output[0],
                batched_output[1, :],
                atol=1e-6 if attention_impl == 'torch' else 1e-8)


@pytest.mark.parametrize('attention_impl', ['torch', 'triton'])
def test_advanced_mask_building(attention_impl):
    # Test that the correct attention mask is created when both
    # prefix_mask and sequence_id are used
    hf_config = MPTConfig(
        init_device='cpu',
        d_model=16,
        n_heads=1,
        n_layers=1,
        expansion_ratio=1,
        max_seq_len=256,
        emb_pdrop=0.0,
        resid_pdrop=0.0,
        attn_config={
            'attn_impl': attention_impl,
            'prefix_lm': True,
            'attn_uses_sequence_id': True,
            'alibi': False,
        },
    )
    mpt = MPTForCausalLM(hf_config)
    mpt.eval()

    prefix_mask = torch.ByteTensor([[1, 1, 0, 0, 1, 1, 1, 0]])
    sequence_id = torch.LongTensor([[0, 0, 0, 0, 1, 1, 1, 1]])

    attn_bias, _ = mpt.transformer._attn_bias(device=mpt.device,
                                              dtype=torch.float32,
                                              attention_mask=None,
                                              prefix_mask=prefix_mask,
                                              sequence_id=sequence_id)

    assert isinstance(attn_bias, torch.Tensor)
    assert attn_bias.shape == torch.Size([1, 1, 8, 8])

    # We'll construct the expected value of attn_bias and then compare.
    can_attend = torch.tensor([
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 1, 1, 1],
    ])
    can_attend = can_attend.bool().view(1, 1, 8, 8)
    expected_attn_bias = torch.zeros_like(attn_bias)
    expected_attn_bias = expected_attn_bias.masked_fill(
        torch.logical_not(can_attend),
        torch.finfo(attn_bias.dtype).min)

    assert torch.equal(attn_bias, expected_attn_bias)


@pytest.mark.parametrize('attention_impl,device', [('torch', 'cpu'),
                                                   ('flash', 'gpu'),
                                                   ('triton', 'gpu'),
                                                   ('torch', 'gpu')])
@pytest.mark.parametrize('alibi', [True, False])
def test_generate(attention_impl, device, alibi):
    # Test that generate works, and produces the same output with or without
    # padding in the input.
    if not torch.cuda.is_available() and device == 'gpu':
        pytest.skip(
            f'This test requires CUDA to be available in order to run with {attention_impl} attention.'
        )
    if alibi and attention_impl == 'flash':
        pytest.skip(f'alibi only implemented with torch and triton attention.')

    reproducibility.seed_all(1234)
    device = get_device(device)

    hf_config = MPTConfig(
        init_device='cpu',
        d_model=128,
        n_heads=4,
        n_layers=2,
        expansion_ratio=2,
        max_seq_len=2048,
        emb_pdrop=0.1,
        resid_pdrop=0.2,
        attn_config={
            'attn_impl': attention_impl,
            'alibi': alibi,
        },
    )
    mpt = MPTForCausalLM(hf_config)
    mpt.eval()
    mpt = device.module_to_device(mpt)

    # padding on the left of the input
    left_padding_input_ids = torch.tensor(
        [[50256, 50256, 50256, 11274, 16390, 11],
         [50256, 50256, 50256, 11274, 16390, 11]])
    left_padding_input_ids = device.tensor_to_device(left_padding_input_ids)
    left_padding_attention_mask = torch.tensor([[0, 0, 0, 1, 1, 1],
                                                [0, 0, 0, 1, 1, 1]])
    left_padding_attention_mask = device.tensor_to_device(
        left_padding_attention_mask)

    # no padding in the input
    no_padding_input_ids = torch.tensor([[11274, 16390, 11], [11274, 16390,
                                                              11]])
    no_padding_input_ids = device.tensor_to_device(no_padding_input_ids)
    no_padding_attention_mask = torch.tensor([[1, 1, 1], [1, 1, 1]])
    no_padding_attention_mask = device.tensor_to_device(
        no_padding_attention_mask)

    # a single batch with different amounts of left padding in the input
    batched_input_ids = torch.tensor([[50256, 50256, 50256, 11274, 16390, 11],
                                      [50256, 50256, 16, 11274, 16390, 11]])
    batched_input_ids = device.tensor_to_device(batched_input_ids)
    batched_attention_mask = torch.tensor([[0, 0, 0, 1, 1, 1],
                                           [0, 0, 1, 1, 1, 1]]).bool()
    batched_attention_mask = device.tensor_to_device(batched_attention_mask)

    with get_precision_context('amp_bf16' if device.name == 'gpu' else 'fp32'):
        # check that a batch with different amounts of padding doesn't crash
        # and produces the right output shape
        batched_generation = mpt.generate(input_ids=batched_input_ids,
                                          attention_mask=batched_attention_mask,
                                          max_new_tokens=5,
                                          use_cache=False)
        assert batched_generation.shape == (2, 6 + 5)

        reproducibility.seed_all(1234)
        generation_with_left_padding = mpt.generate(
            input_ids=left_padding_input_ids,
            attention_mask=left_padding_attention_mask,
            max_new_tokens=5,
            use_cache=False)
        assert generation_with_left_padding.shape == (2, 6 + 5)
        reproducibility.seed_all(1234)
        generation_with_no_padding = mpt.generate(
            input_ids=no_padding_input_ids,
            attention_mask=no_padding_attention_mask,
            max_new_tokens=5,
            use_cache=False)
        assert generation_with_no_padding.shape == (2, 3 + 5)

        # check that left padding and no padding produce the same output
        assert generation_with_no_padding[:, 3:].equal(
            generation_with_left_padding[:, 6:])


@pytest.mark.gpu
@pytest.mark.parametrize('world_size', [1, 2])
@pytest.mark.parametrize('use_cache', [False, True])
def test_generate_with_device_map(tmp_path, world_size, use_cache):
    if not torch.cuda.is_available():
        pytest.skip(f'This test requires CUDA to be available.')
    if not torch.cuda.device_count() >= world_size:
        pytest.skip(f'This test requires {world_size} GPUs.')

    save_path = tmp_path / 'test-device-map'
    hf_config = MPTConfig(
        init_device='cpu',
        d_model=128,
        n_heads=4,
        n_layers=2,
        expansion_ratio=2,
        max_seq_len=2048,
        emb_pdrop=0.1,
        resid_pdrop=0.2,
        attn_config={
            'attn_impl': 'torch',
        },
        use_cache=use_cache,
    )
    mpt = MPTForCausalLM(hf_config)
    mpt.save_pretrained(save_path)

    AutoConfig.register('mpt', MPTConfig)
    AutoModelForCausalLM.register(MPTConfig, MPTForCausalLM)
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')

    device_map = {
        'transformer.wte': 0,
        'transformer.wpe': 0,
        'transformer.embd_drop': 0,
        'transformer.blocks.0': 0,
        'transformer.blocks.1': 1 if world_size == 2 else 0,
        'transformer.norm_f': 1 if world_size == 2 else 0,
    }

    pipe = pipeline(
        'text-generation',
        model=save_path,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=device_map,
    )
    with torch.autocast('cuda', dtype=torch.bfloat16):
        out = pipe(
            'The quick fox jumped over',
            max_length=10,
            do_sample=True,
        )


def check_hf_model_equivalence(model1, model2):
    # Checks that two huggingface models are equivalent (config and
    # parameters)
    expected_model_config_dict = model1.config.to_dict()
    new_model_config_dict = model2.config.to_dict()

    # this key just says the folder it was loaded from, which is a tmp dir during pytest
    del expected_model_config_dict['_name_or_path']
    del new_model_config_dict['_name_or_path']

    assert expected_model_config_dict == new_model_config_dict
    assert sum(p.numel() for p in model1.parameters()) == sum(
        p.numel() for p in model2.parameters())
    assert all(
        type(module1) == type(module2)
        for module1, module2 in zip(model1.modules(), model2.modules()))

    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        torch.testing.assert_close(p1, p2)


def test_save_from_pretrained(tmp_path):
    # Test that MPT can be used with the HuggingFace
    # save_pretrained/from_pretrained api.
    hf_config = MPTConfig(
        init_device='cpu',
        d_model=128,
        n_heads=4,
        n_layers=2,
        expansion_ratio=2,
        max_seq_len=2048,
        emb_pdrop=0.1,
        resid_pdrop=0.2,
        attn_config={
            'attn_impl': 'torch',
        },
    )
    mpt = MPTForCausalLM(hf_config)

    mpt.save_pretrained(tmp_path / 'test-save-pretrained')
    mpt2 = MPTForCausalLM.from_pretrained(tmp_path / 'test-save-pretrained')

    check_hf_model_equivalence(mpt, mpt2)


@pytest.mark.parametrize('alibi', [True, False])
def test_forward_with_cache_and_padding(alibi):
    # Tests that the result is the same with or without padding when using kv caching
    hf_config = MPTConfig(
        init_device='cpu',
        d_model=128,
        n_heads=4,
        n_layers=2,
        expansion_ratio=2,
        max_seq_len=2048,
        emb_pdrop=0.1,
        resid_pdrop=0.2,
        attn_config={
            'attn_impl': 'torch',
            'alibi': alibi,
        },
        use_cache=True,
        init_config={
            'name': 'baseline_',
            'init_std': 0.02,
        },
    )

    mpt = MPTForCausalLM(hf_config)
    mpt.eval()

    first_input_ids_no_padding = torch.tensor([[11274, 16390, 11]])
    first_attention_mask_no_padding = torch.tensor([[1, 1, 1]]).bool()

    # start with passing the first three tokens through (no padding)
    first_output_no_padding = mpt(
        first_input_ids_no_padding,
        attention_mask=first_attention_mask_no_padding)

    second_input_ids_no_padding = torch.tensor([[11274, 16390, 11, 11274]])
    second_attention_mask_no_padding = torch.tensor([[1, 1, 1, 1]]).bool()

    # pass through the fourth token by itself, using the key-value cache (no padding)
    second_output_no_padding = mpt(
        second_input_ids_no_padding[:, -1].unsqueeze(-1),
        attention_mask=second_attention_mask_no_padding,
        past_key_values=first_output_no_padding.past_key_values)

    first_input_ids_padding = torch.tensor([[50256, 11274, 16390, 11]])
    first_attention_mask_padding = torch.tensor([[0, 1, 1, 1]]).bool()

    # start with passing the first three tokens through (with left padding)
    first_output_padding = mpt(first_input_ids_padding,
                               attention_mask=first_attention_mask_padding)

    second_input_ids_padding = torch.tensor([[50256, 11274, 16390, 11, 11274]])
    second_attention_mask_padding = torch.tensor([[0, 1, 1, 1, 1]]).bool()

    # pass through the fourth token by itself, using the key-value cache (with left padding)
    second_output_padding = mpt(
        second_input_ids_padding[:, -1].unsqueeze(-1),
        attention_mask=second_attention_mask_padding,
        past_key_values=first_output_padding.past_key_values)

    # check that the outputs are the same with or without padding
    torch.testing.assert_close(second_output_no_padding.logits,
                               second_output_padding.logits[:,
                                                            -1, :].unsqueeze(1),
                               atol=1e-6,
                               rtol=1e-6)


@pytest.mark.parametrize('attn_impl,device', [
    ('torch', 'cpu'),
    ('flash', 'gpu'),
    ('triton', 'gpu'),
    ('torch', 'gpu'),
])
@pytest.mark.parametrize('alibi', [True, False])
def test_forward_with_cache(attn_impl, device, alibi):
    # Test that model forward with and without the key-value cache produces the
    # same output.
    if not torch.cuda.is_available() and device == 'gpu':
        pytest.skip(
            f'This test requires CUDA to be available in order to run with {attn_impl} attention.'
        )
    if alibi and attn_impl == 'flash':
        pytest.skip(f'alibi only implemented with torch and triton attention.')

    device = get_device(device)

    hf_config = MPTConfig(
        init_device='cpu',
        d_model=128,
        n_heads=4,
        n_layers=2,
        expansion_ratio=2,
        max_seq_len=2048,
        emb_pdrop=0.1,
        resid_pdrop=0.2,
        attn_config={
            'attn_impl': attn_impl,
            'alibi': alibi,
        },
        attn_impl=attn_impl,
        alibi=alibi,
        use_cache=True,
        init_config={
            'name': 'baseline_',
            'init_std': 0.02,
        },
    )
    reproducibility.seed_all(1234)
    mpt = MPTForCausalLM(hf_config)
    mpt = device.module_to_device(mpt)
    mpt.eval()

    with get_precision_context('amp_bf16' if device.name == 'gpu' else 'fp32'):
        reproducibility.seed_all(1234)
        first_input_ids = torch.tensor([[11274, 16390, 11]])
        first_input_ids = device.tensor_to_device(first_input_ids)
        first_attention_mask = torch.tensor([[1, 1, 1]]).bool()
        first_attention_mask = device.tensor_to_device(first_attention_mask)

        # start with passing the first three tokens through
        first_output = mpt(first_input_ids, attention_mask=first_attention_mask)

        assert first_output.logits.shape == (1, 3, hf_config.vocab_size)
        assert len(first_output.past_key_values) == hf_config.n_layers
        assert all(
            len(past_key_value) == 2
            for past_key_value in first_output.past_key_values)
        if attn_impl == 'torch':
            assert all(past_key_value[0].shape == (1, 4, 32, 3)
                       for past_key_value in first_output.past_key_values)
            assert all(past_key_value[1].shape == (1, 4, 3, 32)
                       for past_key_value in first_output.past_key_values)
        else:
            assert all(past_key_value[0].shape == (1, 3, 128)
                       for past_key_value in first_output.past_key_values)
            assert all(past_key_value[1].shape == (1, 3, 128)
                       for past_key_value in first_output.past_key_values)

        reproducibility.seed_all(1234)
        second_input_ids = torch.tensor([[11274, 16390, 11, 11274]])
        second_input_ids = device.tensor_to_device(second_input_ids)
        second_attention_mask = torch.tensor([[1, 1, 1, 1]]).bool()
        second_attention_mask = device.tensor_to_device(second_attention_mask)

        # pass through the fourth token by itself, using the key-value cache
        second_output = mpt(
            second_input_ids[:, -1].unsqueeze(-1),
            past_key_values=first_output.past_key_values,
            attention_mask=second_attention_mask,
        )

        assert second_output.logits.shape == (1, 1, hf_config.vocab_size)
        assert len(second_output.past_key_values) == hf_config.n_layers
        assert all(
            len(past_key_value) == 2
            for past_key_value in second_output.past_key_values)
        if attn_impl == 'torch':
            assert all(past_key_value[0].shape == (1, 4, 32, 4)
                       for past_key_value in second_output.past_key_values)
            assert all(past_key_value[1].shape == (1, 4, 4, 32)
                       for past_key_value in second_output.past_key_values)
        else:
            assert all(past_key_value[0].shape == (1, 4, 128)
                       for past_key_value in second_output.past_key_values)
            assert all(past_key_value[1].shape == (1, 4, 128)
                       for past_key_value in second_output.past_key_values)

        reproducibility.seed_all(1234)
        # pass through the first four tokens without the key-value cache
        full_output = mpt(second_input_ids,
                          attention_mask=second_attention_mask)

        # check that the output is the same whether using the key-value cache or not
        torch.testing.assert_close(
            second_output.logits,
            full_output.logits[:, -1, :].unsqueeze(1),
            atol=1e-2,
            rtol=1e-2,
        )


@pytest.mark.parametrize('alibi', [True, False])
def test_generate_with_past_kv(alibi):
    hf_config = MPTConfig(
        init_device='cpu',
        d_model=128,
        n_heads=4,
        n_layers=2,
        expansion_ratio=2,
        max_seq_len=2048,
        emb_pdrop=0.1,
        resid_pdrop=0.2,
        attn_config={
            'attn_impl': 'torch',
            'alibi': alibi,
        },
        use_cache=True,
        init_config={
            'name': 'baseline_',
            'init_std': 0.02,
        },
    )
    mpt = MPTForCausalLM(hf_config)
    mpt.eval()

    # no padding in the input
    no_padding_input_ids = torch.tensor([[11274, 16390, 11]])
    no_padding_attention_mask = torch.tensor([[1, 1, 1]])

    with mock.patch.object(MPTForCausalLM, 'forward',
                           autospec=True) as forward_mocked:
        forward_mocked.return_value = CausalLMOutputWithPast(
            logits=torch.randn((1, 3, hf_config.vocab_size)),
            past_key_values=[(torch.randn(1, 3, hf_config.d_model),
                              torch.randn(1, 3, hf_config.d_model))
                             for _ in range(hf_config.n_layers)])
        _ = mpt.generate(input_ids=no_padding_input_ids,
                         attention_mask=no_padding_attention_mask,
                         max_new_tokens=2)

        assert forward_mocked.call_count == 2
        _, _, kwargs = forward_mocked.mock_calls[0]
        assert kwargs['past_key_values'] is None
        _, _, kwargs = forward_mocked.mock_calls[1]
        assert kwargs['past_key_values'] is not None
        assert len(kwargs['past_key_values']) == hf_config.n_layers
        assert kwargs['past_key_values'][0][0].shape == (1, 3,
                                                         hf_config.d_model)


@pytest.mark.parametrize('generation_kwargs', [{
    'max_new_tokens': 2,
    'num_beams': 4
}, {
    'max_new_tokens': 2,
    'top_k': 5,
    'penalty_alpha': 0.4
}, {
    'do_sample': True,
    'top_p': 0.95
}])
@pytest.mark.parametrize('alibi', [True, False])
def test_generation_kwargs_dont_crash(generation_kwargs, alibi):
    hf_config = MPTConfig(
        init_device='cpu',
        d_model=128,
        n_heads=4,
        n_layers=2,
        expansion_ratio=2,
        max_seq_len=2048,
        emb_pdrop=0.1,
        resid_pdrop=0.2,
        attn_config={
            'attn_impl': 'torch',
            'alibi': alibi,
        },
        use_cache=True,
    )
    mpt = MPTForCausalLM(hf_config)
    mpt.eval()

    # no padding in the input
    no_padding_input_ids = torch.tensor([[11274, 16390, 11]])
    no_padding_attention_mask = torch.tensor([[1, 1, 1]])

    _ = mpt.generate(input_ids=no_padding_input_ids,
                     attention_mask=no_padding_attention_mask,
                     **generation_kwargs)


@pytest.mark.gpu
@pytest.mark.parametrize('attention_impl', ['torch', 'flash', 'triton'])
@pytest.mark.parametrize('alibi', [True, False])
def test_model_to(attention_impl, alibi):
    # test that moving the model to diff devices and dtypes in diff ways does not break the model
    if not torch.cuda.is_available():
        pytest.skip(
            f'This test requires CUDA to be available in order to run with {attention_impl} attention.'
        )
    if alibi and attention_impl == 'flash':
        pytest.skip(f'alibi only implemented with torch and triton attention.')

    hf_config = MPTConfig(
        init_device='cpu',
        d_model=128,
        n_heads=4,
        n_layers=2,
        expansion_ratio=2,
        max_seq_len=2048,
        emb_pdrop=0.1,
        resid_pdrop=0.2,
        attn_config={
            'attn_impl': attention_impl,
            'alibi': alibi,
        },
        use_cache=True,
        init_config={
            'name': 'baseline_',
            'init_std': 0.02,
        },
    )
    reproducibility.seed_all(1234)
    mpt = MPTForCausalLM(hf_config)
    mpt = mpt.bfloat16()
    mpt = mpt.to('cuda')
    mpt.eval()

    # gen input data
    input_ids = torch.tensor([[11274, 16390, 11]]).to('cuda')
    attention_mask = torch.tensor([[1, 1, 1]]).bool().to('cuda')

    # with get_precision_context('amp_bf16'):
    _ = mpt(input_ids, attention_mask=attention_mask)

    # move the model around using different methods
    mpt = mpt.bfloat16()
    mpt = mpt.to('cpu')

    # verify the model still works
    if attention_impl == 'torch':
        with torch.autocast('cpu', dtype=torch.bfloat16, enabled=True):
            _ = mpt(input_ids.to('cpu'),
                    attention_mask=attention_mask.to('cpu'))

    mpt = mpt.cuda()
    mpt = mpt.bfloat16()

    # verify the model still works
    if attention_impl == 'torch':
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
            _ = mpt(input_ids, attention_mask=attention_mask)

    mpt = mpt.to('cpu')
    mpt = mpt.float()

    # verify the model still works
    if attention_impl == 'torch':
        _ = mpt(input_ids.to('cpu'), attention_mask=attention_mask.to('cpu'))

    mpt = mpt.half()
    mpt = mpt.to(0)  # move to rank0
    mpt = mpt.bfloat16()

    # verify the model still works
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
        _ = mpt(input_ids, attention_mask=attention_mask)


def test_alibi_vs_hf():
    # compare alibi-bias generation vs HF Bloom model alibi-bias for diff seq len and n_heads
    for n_heads in range(1, 64):
        for seq_len in [1, 2, 8, 13, 64, 195, 256]:
            # hf bloom alibi bais
            alibi_bias_hf = build_alibi_tensor(
                torch.ones(seq_len)[None, ...], n_heads, torch.float32)
            alibi_bias_hf = alibi_bias_hf - alibi_bias_hf.max(
                dim=2, keepdim=True).values

            # mosaicml alibi bais
            alibi_bias_m = build_alibi_bias(n_heads,
                                            seq_len,
                                            dtype=torch.float32)
            alibi_bias_m = alibi_bias_m[0]

            torch.testing.assert_close(alibi_bias_hf, alibi_bias_m)


@pytest.mark.parametrize('attn_impl,device', [
    ('torch', 'cpu'),
    ('flash', 'gpu'),
    ('triton', 'gpu'),
    ('torch', 'gpu'),
])
@pytest.mark.parametrize('alibi', [True, False])
@pytest.mark.parametrize('output_attentions', [True, False])
@pytest.mark.parametrize('output_hidden_states', [True, False])
def test_forward_with_output_attentions_and_output_hidden_states(
        attn_impl, device, alibi, output_attentions, output_hidden_states):
    # Test that model forward with output_attentions_and_output_hidden_states
    if not torch.cuda.is_available() and device == 'gpu':
        pytest.skip(
            f'This test requires CUDA to be available in order to run with {attn_impl} attention.'
        )
    if alibi and attn_impl == 'flash':
        pytest.skip(f'alibi only implemented with torch and triton attention.')
    if output_attentions and attn_impl in ['flash', 'triton']:
        pytest.skip(f'output_attentions only implemented with torch attention.')

    device = get_device(device)

    n_layers = 2

    hf_config = MPTConfig(
        init_device='cpu',
        d_model=128,
        n_heads=4,
        n_layers=n_layers,
        expansion_ratio=2,
        max_seq_len=2048,
        emb_pdrop=0.1,
        resid_pdrop=0.2,
        attn_config={
            'attn_impl': attn_impl,
            'alibi': alibi,
        },
        attn_impl=attn_impl,
        alibi=alibi,
        use_cache=True,
        init_config={
            'name': 'baseline_',
            'init_std': 0.02,
        },
    )
    reproducibility.seed_all(1234)
    mpt = MPTForCausalLM(hf_config)
    mpt = device.module_to_device(mpt)
    mpt.eval()

    with get_precision_context('amp_bf16' if device.name == 'gpu' else 'fp32'):
        reproducibility.seed_all(1234)
        input_ids = torch.tensor([[11274, 16390, 11]])
        input_ids = device.tensor_to_device(input_ids)
        attention_mask = torch.tensor([[1, 1, 1]]).bool()
        attention_mask = device.tensor_to_device(attention_mask)

        # start with passing the first three tokens through
        outputs = mpt(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if output_attentions:
            assert len(outputs.attentions) == n_layers
        if output_hidden_states:
            assert len(outputs.hidden_states) == n_layers + 1


@pytest.mark.gpu
@pytest.mark.parametrize('init_device', ['cpu', 'meta', 'mixed'])
@pytest.mark.parametrize('world_size', [2])
def test_hf_init(tmp_path,
                 init_device: str,
                 world_size: int,
                 batch_size: int = 1):
    if not torch.cuda.is_available():
        pytest.skip(f'This test requires CUDA to be available.')
    if not torch.cuda.device_count() >= world_size:
        pytest.skip(f'This test requires {world_size} GPUs.')

    torch.cuda.empty_cache()
    gc.collect()  #just in case
    torch.cuda.synchronize()

    test_cfg = get_config(conf_path='scripts/train/yamls/pretrain/testing.yaml')
    test_cfg.device = torch.cuda.current_device()

    device = get_device(None)
    dist.initialize_dist(device, timeout=30)

    fsdp_config = {
        'sharding_strategy': 'FULL_SHARD',
    }

    save_path = tmp_path / 'test-hf-device-init'

    if init_device == 'mixed':
        if dist.get_local_rank() != 0:
            init_device = 'meta'
        else:
            init_device = 'cpu'

    precision = Precision('amp_bf16')

    hf_config = MPTConfig(
        init_device=init_device,
        d_model=32,
        n_heads=4,
        n_layers=1,
        expansion_ratio=2,
        max_seq_len=128,
        emb_pdrop=0.1,
        resid_pdrop=0.2,
        attn_config={
            'attn_impl': 'torch',
        },
    )

    mpt = MPTForCausalLM(hf_config)
    mpt.save_pretrained(save_path)

    AutoConfig.register('mpt', MPTConfig)
    AutoModelForCausalLM.register(MPTConfig, MPTForCausalLM)

    context = contextlib.nullcontext()
    if init_device == 'meta':
        context = init_empty_weights(include_buffers=False)

    # Load in a pretrained model with a given context
    with context:
        model = AutoModelForCausalLM.from_pretrained(save_path,
                                                     trust_remote_code=True)

    tokenizer = build_tokenizer(test_cfg.tokenizer)
    optimizer = DecoupledAdamW(model.parameters(), lr=1e-5, betas=[0.9, 0.99])

    prepare_fsdp_module(model, optimizer, fsdp_config, precision, device, False)

    model = HuggingFaceModelWithZLoss(model, tokenizer)

    batch = gen_random_batch(batch_size, test_cfg)

    original_params = next(model.parameters()).clone().data

    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
        outputs = model(batch)
    loss = model.loss(outputs, batch)
    loss.backward()
    optimizer.step()

    updated_params = next(model.parameters()).clone().data

    assert not torch.equal(original_params, updated_params)
