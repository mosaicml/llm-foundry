# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import copy
import os
import warnings
from typing import cast
from unittest import mock

import pytest
import torch
import torch.nn as nn
from composer.core.precision import get_precision_context
from composer.optim import DecoupledAdamW
from composer.utils import get_device, reproducibility
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from transformers.modeling_outputs import CausalLMOutputWithPast

from examples.llm import (COMPOSER_MODEL_REGISTRY, ComposerHFCausalLM,
                          ComposerHFPrefixLM)
from examples.llm.src.models.mosaic_gpt import MosaicGPT, MosaicGPTConfig


def get_config(conf_path='yamls/mosaic_gpt/testing.yaml') -> DictConfig:
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    print(conf_path)
    with open(conf_path) as f:
        test_cfg = om.load(f)
    return cast(DictConfig, test_cfg)


def get_objs(conf_path='yamls/mosaic_gpt/testing.yaml'):
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
    test_cfg.model.attn_impl = 'torch'
    # device = 'cuda'
    # test_cfg.precision = 'amp'
    test_cfg.model.init_device = device
    test_cfg.device = device

    test_cfg.global_train_batch_size = 2
    test_cfg.device_eval_batch_size = 2
    test_cfg.device_train_microbatch_size = 2

    model = COMPOSER_MODEL_REGISTRY[test_cfg.model.name](test_cfg.model,
                                                         test_cfg.tokenizer)
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
        conf_path='yamls/mosaic_gpt/testing.yaml')

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
    test_cfg, model, _ = get_objs(conf_path='yamls/mosaic_gpt/testing.yaml')

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

    attn_bias, attention_mask = model.model._attn_bias(
        device=x.device, dtype=x.dtype, attention_mask=attention_mask)

    for block in model.model.transformer.blocks:
        a = block.ln_1(x)
        b, attention_weights, _ = block.attn(a,
                                             past_key_value=None,
                                             attn_bias=attn_bias,
                                             attention_mask=attention_mask,
                                             is_causal=model.model.is_causal,
                                             needs_weights=True)

        zerod_weights = (attention_weights == 0)
        assert torch.equal(expected_zerod_weights.expand(*zerod_weights.shape),
                           zerod_weights)
        x = x + block.resid_attn_dropout(b)
        m = block.ln_2(x)
        n = block.mlp(m)
        x = x + block.resid_mlp_dropout(n)


@pytest.mark.parametrize('prefixlm', [False, True])
def test_full_forward_and_backward_gpt2_small(prefixlm, batch_size=2):
    warnings.filterwarnings(
        action='ignore',
        message='Torchmetrics v0.9 introduced a new argument class property')
    conf_path = 'yamls/hf_causal_lm/gpt2-small.yaml'
    with open(conf_path) as f:
        neo_cfg = om.load(f)

    device = 'cpu'
    neo_cfg.device = device
    neo_cfg.max_seq_len = 256

    if prefixlm:
        neo_cfg.model.name = 'hf_prefix_lm'
    else:
        neo_cfg.model.name = 'hf_causal_lm'

    model = COMPOSER_MODEL_REGISTRY[neo_cfg.model.name](
        neo_cfg.model, neo_cfg.tokenizer).to(device)

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
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({
        'model': {
            'pretrained_model_name_or_path': 't5-small',
            'pretrained': False,
            'z_loss': 0.0001,
        },
        'optimizer': {
            'lr': 0.0001,
            'betas': [0.9, 0.99],
            'eps': 1e-6,
            'weight_decay': 0.00001
        },
        'tokenizer': {
            'name': 't5-small',
        }
    })

    device = 'cpu'
    max_seq_len = 16

    model = COMPOSER_MODEL_REGISTRY['hf_t5'](cfg.model,
                                             cfg.tokenizer).to(device)

    optimizer = DecoupledAdamW(model.parameters(),
                               lr=cfg.optimizer.lr,
                               betas=cfg.optimizer.betas,
                               eps=cfg.optimizer.eps,
                               weight_decay=cfg.optimizer.weight_decay)

    # set vocab size using model num_embeddings
    batch = gen_random_enc_dec_batch(batch_size, model.model.config.vocab_size,
                                     max_seq_len, device)

    assert batch['input_ids'].shape == torch.Size([batch_size, max_seq_len])
    model.train()
    original_params = next(model.parameters()).clone().data
    outputs = model(batch)
    loss = model.loss(outputs, batch)
    loss.backward()
    optimizer.step()
    updated_params = next(model.parameters()).clone().data
    assert not torch.equal(original_params, updated_params)


@pytest.mark.parametrize(
    'attention_type,precision',
    [('torch', torch.float16), ('torch', torch.bfloat16),
     pytest.param('flash', torch.float16, marks=pytest.mark.gpu),
     pytest.param('flash', torch.bfloat16, marks=pytest.mark.gpu)])
def test_determinism(attention_type: str, precision):
    if not torch.cuda.is_available():
        pytest.skip(
            'This test requires CUDA to be available in order to run with bfloat16 precision.'
        )
    reproducibility.seed_all(1111)

    conf_path = 'yamls/mosaic_gpt/testing.yaml'
    with open(conf_path) as f:
        test_cfg = om.load(f)

    test_cfg.model.attn_impl = attention_type
    test_cfg.model.init_device = 'cuda:0'
    test_cfg.device = 'cuda:0'

    model_1 = COMPOSER_MODEL_REGISTRY[test_cfg.model.name](test_cfg.model,
                                                           test_cfg.tokenizer)
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

    reproducibility.seed_all(1111)

    conf_path = 'yamls/mosaic_gpt/testing.yaml'
    with open(conf_path) as f:
        test_cfg = om.load(f)

    test_cfg.device = 'cuda:0'
    test_cfg.model.init_device = 'cuda:0'
    test_cfg.model.param_init_fn = 'baseline_'
    test_cfg.model.init_std = 0.02

    model_1 = COMPOSER_MODEL_REGISTRY[test_cfg.model.name](test_cfg.model,
                                                           test_cfg.tokenizer)
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

    for i in range(25):
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

    if prefixlm:
        model = ComposerHFPrefixLM(config.model, config.tokenizer)
    else:
        model = ComposerHFCausalLM(config.model, config.tokenizer)

    # check that all the modules we except are blocked from FSDP wrapping
    assert not model.model.model._fsdp_wrap
    assert not model.model.model.decoder._fsdp_wrap
    assert not model.model.model.decoder.embed_tokens._fsdp_wrap
    assert not model.model.lm_head._fsdp_wrap


def test_mosaic_gpt_creation():
    # Test that the config constructs the model as expected.
    hf_config = MosaicGPTConfig(
        init_device='cpu',
        d_model=128,
        n_heads=4,
        n_layers=2,
        mlp_ratio=2,
        max_seq_len=2048,
        emb_pdrop=0.1,
        resid_pdrop=0.2,
        attn_impl='torch',
    )
    mosaic_gpt = MosaicGPT(hf_config)

    assert mosaic_gpt.config.d_model == 128
    assert mosaic_gpt.config.n_heads == 4
    assert mosaic_gpt.config.n_layers == 2
    assert mosaic_gpt.config.mlp_ratio == 2
    assert mosaic_gpt.config.max_seq_len == 2048

    assert mosaic_gpt.transformer.wte.weight.shape == torch.Size(  # type: ignore
        [hf_config.vocab_size, hf_config.d_model])
    assert mosaic_gpt.transformer.wpe.weight.shape == torch.Size(  # type: ignore
        [hf_config.max_seq_len, hf_config.d_model])
    assert mosaic_gpt.transformer.emb_drop.p == 0.1  # type: ignore
    assert len(mosaic_gpt.transformer.blocks) == 2  # type: ignore

    d_model = hf_config.d_model
    for block in mosaic_gpt.transformer.blocks:  # type: ignore
        assert block.ln_1.weight.shape == torch.Size([d_model])  # type: ignore
        assert block.ln_2.weight.shape == torch.Size([d_model])  # type: ignore
        assert block.mlp.mlp_up.weight.shape == torch.Size(  # type: ignore
            [hf_config.d_model * hf_config.mlp_ratio, hf_config.d_model])
        assert block.mlp.mlp_down.weight.shape == torch.Size(  # type: ignore
            [hf_config.d_model, hf_config.d_model * hf_config.mlp_ratio])
        assert block.resid_attn_dropout.p == 0.2  # type: ignore
        assert block.resid_mlp_dropout.p == 0.2  # type: ignore


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

    hf_config = MosaicGPTConfig(
        init_device='cpu',
        d_model=128,
        n_heads=1,
        n_layers=2,
        mlp_ratio=2,
        max_seq_len=2048,
        emb_pdrop=0.1,
        resid_pdrop=0.2,
        attn_impl=attention_impl,
        alibi=alibi,
        param_init_fn='baseline_',
        init_std=0.02,
    )
    mosaic_gpt = MosaicGPT(hf_config)
    mosaic_gpt.eval()
    mosaic_gpt = device.module_to_device(mosaic_gpt)

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

        right_padding_output = mosaic_gpt(
            right_padding_input_ids,
            attention_mask=right_padding_attention_mask).logits
        middle_padding_output = mosaic_gpt(
            middle_padding_input_ids,
            attention_mask=middle_padding_attention_mask).logits
        left_padding_output = mosaic_gpt(
            left_padding_input_ids,
            attention_mask=left_padding_attention_mask).logits
        batched_output = mosaic_gpt(
            batched_input_ids, attention_mask=batched_attention_mask).logits

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
    hf_config = MosaicGPTConfig(init_device='cpu',
                                d_model=16,
                                n_heads=1,
                                n_layers=1,
                                mlp_ratio=1,
                                max_seq_len=256,
                                emb_pdrop=0.0,
                                resid_pdrop=0.0,
                                attn_impl=attention_impl,
                                prefix_lm=True,
                                attn_uses_sequence_id=True,
                                alibi=False)
    mosaic_gpt = MosaicGPT(hf_config)
    mosaic_gpt.eval()

    prefix_mask = torch.ByteTensor([[1, 1, 0, 0, 1, 1, 1, 0]])
    sequence_id = torch.LongTensor([[0, 0, 0, 0, 1, 1, 1, 1]])

    attn_bias, _ = mosaic_gpt._attn_bias(device=mosaic_gpt.device,
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

    hf_config = MosaicGPTConfig(
        init_device='cpu',
        d_model=128,
        n_heads=4,
        n_layers=2,
        mlp_ratio=2,
        max_seq_len=2048,
        emb_pdrop=0.1,
        resid_pdrop=0.2,
        attn_impl=attention_impl,
        alibi=alibi,
    )
    mosaic_gpt = MosaicGPT(hf_config)
    mosaic_gpt.eval()
    mosaic_gpt = device.module_to_device(mosaic_gpt)

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
        batched_generation = mosaic_gpt.generate(
            input_ids=batched_input_ids,
            attention_mask=batched_attention_mask,
            max_new_tokens=5,
            use_cache=False)
        assert batched_generation.shape == (2, 6 + 5)

        reproducibility.seed_all(1234)
        generation_with_left_padding = mosaic_gpt.generate(
            input_ids=left_padding_input_ids,
            attention_mask=left_padding_attention_mask,
            max_new_tokens=5,
            use_cache=False)
        assert generation_with_left_padding.shape == (2, 6 + 5)
        reproducibility.seed_all(1234)
        generation_with_no_padding = mosaic_gpt.generate(
            input_ids=no_padding_input_ids,
            attention_mask=no_padding_attention_mask,
            max_new_tokens=5,
            use_cache=False)
        assert generation_with_no_padding.shape == (2, 3 + 5)

        # check that left padding and no padding produce the same output
        assert generation_with_no_padding[:, 3:].equal(
            generation_with_left_padding[:, 6:])


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
    # Test that MosaicGPT can be used with the HuggingFace
    # save_pretrained/from_pretrained api.
    hf_config = MosaicGPTConfig(
        init_device='cpu',
        d_model=128,
        n_heads=4,
        n_layers=2,
        mlp_ratio=2,
        max_seq_len=2048,
        emb_pdrop=0.1,
        resid_pdrop=0.2,
        attn_impl='torch',
    )
    mosaic_gpt = MosaicGPT(hf_config)

    mosaic_gpt.save_pretrained(tmp_path / 'test-save-pretrained')
    mosaic_gpt2 = MosaicGPT.from_pretrained(tmp_path / 'test-save-pretrained')

    check_hf_model_equivalence(mosaic_gpt, mosaic_gpt2)


@pytest.mark.parametrize('alibi', [True, False])
def test_forward_with_cache_and_padding(alibi):
    # Tests that the result is the same with or without padding when using kv caching
    hf_config = MosaicGPTConfig(
        init_device='cpu',
        d_model=128,
        n_heads=4,
        n_layers=2,
        mlp_ratio=2,
        max_seq_len=2048,
        emb_pdrop=0.1,
        resid_pdrop=0.2,
        attn_impl='torch',
        alibi=alibi,
        use_cache=True,
        param_init_fn='baseline_',
        init_std=0.02,
    )

    mosaic_gpt = MosaicGPT(hf_config)
    mosaic_gpt.eval()

    first_input_ids_no_padding = torch.tensor([[11274, 16390, 11]])
    first_attention_mask_no_padding = torch.tensor([[1, 1, 1]]).bool()

    # start with passing the first three tokens through (no padding)
    first_output_no_padding = mosaic_gpt(
        first_input_ids_no_padding,
        attention_mask=first_attention_mask_no_padding)

    second_input_ids_no_padding = torch.tensor([[11274, 16390, 11, 11274]])
    second_attention_mask_no_padding = torch.tensor([[1, 1, 1, 1]]).bool()

    # pass through the fourth token by itself, using the key-value cache (no padding)
    second_output_no_padding = mosaic_gpt(
        second_input_ids_no_padding[:, -1].unsqueeze(-1),
        attention_mask=second_attention_mask_no_padding,
        past_key_values=first_output_no_padding.past_key_values)

    first_input_ids_padding = torch.tensor([[50256, 11274, 16390, 11]])
    first_attention_mask_padding = torch.tensor([[0, 1, 1, 1]]).bool()

    # start with passing the first three tokens through (with left padding)
    first_output_padding = mosaic_gpt(
        first_input_ids_padding, attention_mask=first_attention_mask_padding)

    second_input_ids_padding = torch.tensor([[50256, 11274, 16390, 11, 11274]])
    second_attention_mask_padding = torch.tensor([[0, 1, 1, 1, 1]]).bool()

    # pass through the fourth token by itself, using the key-value cache (with left padding)
    second_output_padding = mosaic_gpt(
        second_input_ids_padding[:, -1].unsqueeze(-1),
        attention_mask=second_attention_mask_padding,
        past_key_values=first_output_padding.past_key_values)

    # check that the outputs are the same with or without padding
    torch.testing.assert_close(second_output_no_padding.logits,
                               second_output_padding.logits[:,
                                                            -1, :].unsqueeze(1),
                               atol=1e-6,
                               rtol=1e-6)


@pytest.mark.parametrize('attention_impl,device', [('torch', 'cpu'),
                                                   ('flash', 'gpu'),
                                                   ('triton', 'gpu'),
                                                   ('torch', 'gpu')])
@pytest.mark.parametrize('alibi', [True, False])
def test_forward_with_cache(attention_impl, device, alibi):
    # Test that model forward with and without the key-value cache produces the
    # same output.
    if not torch.cuda.is_available() and device == 'gpu':
        pytest.skip(
            f'This test requires CUDA to be available in order to run with {attention_impl} attention.'
        )
    if alibi and attention_impl == 'flash':
        pytest.skip(f'alibi only implemented with torch and triton attention.')

    device = get_device(device)

    hf_config = MosaicGPTConfig(
        init_device='cpu',
        d_model=128,
        n_heads=4,
        n_layers=2,
        mlp_ratio=2,
        max_seq_len=2048,
        emb_pdrop=0.1,
        resid_pdrop=0.2,
        attn_impl=attention_impl,
        alibi=alibi,
        use_cache=True,
        param_init_fn='baseline_',
        init_std=0.02,
    )
    reproducibility.seed_all(1234)
    mosaic_gpt = MosaicGPT(hf_config)
    mosaic_gpt.eval()
    mosaic_gpt = device.module_to_device(mosaic_gpt)

    with get_precision_context('amp_bf16' if device.name == 'gpu' else 'fp32'):
        reproducibility.seed_all(1234)
        first_input_ids = torch.tensor([[11274, 16390, 11]])
        first_input_ids = device.tensor_to_device(first_input_ids)
        first_attention_mask = torch.tensor([[1, 1, 1]]).bool()
        first_attention_mask = device.tensor_to_device(first_attention_mask)

        # start with passing the first three tokens through
        first_output = mosaic_gpt(first_input_ids,
                                  attention_mask=first_attention_mask)

        assert first_output.logits.shape == (1, 3, hf_config.vocab_size)
        assert len(first_output.past_key_values) == 2
        assert all(
            len(past_key_value) == 2
            for past_key_value in first_output.past_key_values)
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
        second_output = mosaic_gpt(second_input_ids[:, -1].unsqueeze(-1),
                                   attention_mask=second_attention_mask,
                                   past_key_values=first_output.past_key_values)

        assert second_output.logits.shape == (1, 1, hf_config.vocab_size)
        assert len(second_output.past_key_values) == 2
        assert all(
            len(past_key_value) == 2
            for past_key_value in second_output.past_key_values)
        assert all(past_key_value[0].shape == (1, 4, 128)
                   for past_key_value in second_output.past_key_values)
        assert all(past_key_value[1].shape == (1, 4, 128)
                   for past_key_value in second_output.past_key_values)

        reproducibility.seed_all(1234)
        # pass through the first four tokens without the key-value cache
        full_output = mosaic_gpt(second_input_ids,
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
    hf_config = MosaicGPTConfig(
        init_device='cpu',
        d_model=128,
        n_heads=4,
        n_layers=2,
        mlp_ratio=2,
        max_seq_len=2048,
        emb_pdrop=0.1,
        resid_pdrop=0.2,
        attn_impl='torch',
        alibi=alibi,
        use_cache=True,
        param_init_fn='baseline_',
        init_std=0.02,
    )
    mosaic_gpt = MosaicGPT(hf_config)
    mosaic_gpt.eval()

    # no padding in the input
    no_padding_input_ids = torch.tensor([[11274, 16390, 11]])
    no_padding_attention_mask = torch.tensor([[1, 1, 1]])

    with mock.patch.object(MosaicGPT, 'forward',
                           autospec=True) as forward_mocked:
        forward_mocked.return_value = CausalLMOutputWithPast(
            logits=torch.randn((1, 3, hf_config.vocab_size)),
            past_key_values=[(torch.randn(1, 3, hf_config.d_model),
                              torch.randn(1, 3, hf_config.d_model))
                             for _ in range(hf_config.n_layers)])
        _ = mosaic_gpt.generate(input_ids=no_padding_input_ids,
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
    hf_config = MosaicGPTConfig(
        init_device='cpu',
        d_model=128,
        n_heads=4,
        n_layers=2,
        mlp_ratio=2,
        max_seq_len=2048,
        emb_pdrop=0.1,
        resid_pdrop=0.2,
        attn_impl='torch',
        alibi=alibi,
        use_cache=True,
    )
    mosaic_gpt = MosaicGPT(hf_config)
    mosaic_gpt.eval()

    # no padding in the input
    no_padding_input_ids = torch.tensor([[11274, 16390, 11]])
    no_padding_attention_mask = torch.tensor([[1, 1, 1]])

    _ = mosaic_gpt.generate(input_ids=no_padding_input_ids,
                            attention_mask=no_padding_attention_mask,
                            **generation_kwargs)


def test_tokenizer_max_length_load(max_seq_len=2048):
    conf_path = 'yamls/mosaic_gpt/testing.yaml'
    with open(conf_path) as f:
        test_cfg = om.load(f)

    test_cfg.max_seq_len = max_seq_len

    model = COMPOSER_MODEL_REGISTRY[test_cfg.model.name](test_cfg.model,
                                                         test_cfg.tokenizer)
    assert model.tokenizer.model_max_length == max_seq_len
