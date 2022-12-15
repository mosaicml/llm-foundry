# Copyright 2022 MosaicML Benchmarks authors
# SPDX-License-Identifier: Apache-2.0

import copy
import os
import pytest
import warnings
from typing import cast

import torch
import torch.nn as nn
from composer.optim import DecoupledAdamW
from composer.utils import reproducibility
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from src.model_registry import COMPOSER_MODEL_REGISTRY
from src.tokenizer import TOKENIZER_REGISTRY


def get_config(conf_path='yamls/mosaic_gpt/125m.yaml') -> DictConfig:
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    print(conf_path)
    with open(conf_path) as f:
        test_cfg = om.load(f)
    return cast(DictConfig, test_cfg)


def get_objs(conf_path='yamls/mosaic_gpt/125m.yaml'):
    warnings.filterwarnings(
        action='ignore',
        message='Torchmetrics v0.9 introduced a new argument class property')
    test_cfg = get_config(conf_path=conf_path)
    _ = TOKENIZER_REGISTRY[test_cfg.tokenizer.type](
        **test_cfg.tokenizer.args)  # make sure tokenizer in registry

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
    test_cfg.model.device = device
    test_cfg.device = device

    test_cfg.global_train_batch_size = 2
    test_cfg.device_eval_batch_size = 2
    test_cfg.device_train_microbatch_size = 2

    model = COMPOSER_MODEL_REGISTRY[test_cfg.model.name](test_cfg.model)
    # Optimizer
    assert test_cfg.optimizer.name == 'decoupled_adamw'
    optimizer = DecoupledAdamW(model.parameters(),
                               lr=test_cfg.optimizer.lr,
                               betas=test_cfg.optimizer.betas,
                               eps=test_cfg.optimizer.eps,
                               weight_decay=test_cfg.optimizer.weight_decay)

    return test_cfg, model, optimizer


def gen_random_batch(batch_size, test_cfg):
    # generate input batch of random data
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
    return batch


def test_full_forward_and_backward(batch_size=2):
    test_cfg, model, optimizer = get_objs(
        conf_path='yamls/mosaic_gpt/125m.yaml')

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
    test_cfg, model, _ = get_objs(conf_path='yamls/mosaic_gpt/125m.yaml')

    batch = gen_random_batch(batch_size, test_cfg)

    model.eval()
    # run a partial forward where we explicitly inspect the attention_mask from the causal_attn block
    input_ids, key_padding_mask = batch['input_ids'], batch[
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
        [(~key_padding_mask).reshape(batch_size, 1, test_cfg.max_seq_len)],
        axis=1)
    expected_zerod_weights |= torch_key_padding

    for block in model.model.transformer.blocks:
        a = block.ln_1(x)
        b, attention_weights = block.causal_attn(a, key_padding_mask)

        zerod_weights = (attention_weights == 0)
        assert torch.equal(expected_zerod_weights, zerod_weights)
        x = x + block.resid_attn_dropout(b)
        m = block.ln_2(x)
        n = block.mlp(m)
        x = x + block.resid_mlp_dropout(n)


def test_full_forward_and_backward_gpt_neo(batch_size=2):
    warnings.filterwarnings(
        action='ignore',
        message='Torchmetrics v0.9 introduced a new argument class property')
    conf_path = 'yamls/hf_causal_lm/gpt-neo-125m.yaml'
    with open(conf_path) as f:
        neo_cfg = om.load(f)

    device = 'cpu'
    neo_cfg.device = device

    model = COMPOSER_MODEL_REGISTRY[neo_cfg.model.name](
        neo_cfg.model).to(device)

    assert neo_cfg.optimizer.name == 'decoupled_adamw'
    optimizer = DecoupledAdamW(model.parameters(),
                               lr=neo_cfg.optimizer.lr,
                               betas=neo_cfg.optimizer.betas,
                               eps=neo_cfg.optimizer.eps,
                               weight_decay=neo_cfg.optimizer.weight_decay)

    # set vacab size using model num_embeddings
    neo_cfg.model.vocab_size = model.model.transformer.wte.num_embeddings
    batch = gen_random_batch(batch_size, neo_cfg)

    batch['input_ids'].shape == torch.Size([batch_size, neo_cfg.max_seq_len])
    model.train()
    original_params = next(model.parameters()).clone().data
    outputs = model(batch)
    loss = model.loss(outputs, batch)
    loss.backward()
    optimizer.step()
    updated_params = next(model.parameters()).clone().data
    assert not torch.equal(original_params, updated_params)

@pytest.mark.parametrize('attention_type,precision', [
    ('torch', torch.float16),
    ('torch', torch.bfloat16),
    ('flash', torch.float16),
    # Note: Whether this test fails or not depends on the random seed, how many steps are run for,
    # and possibly other stuff like torch/cuda version. It is flaky.
    pytest.param('flash', torch.bfloat16, marks=pytest.mark.xfail)
])
def test_determinism(attention_type: str, precision):
    if not torch.cuda.is_available():
        pytest.skip('This test requires CUDA to be available in order to run with bfloat16 precision.')
    reproducibility.seed_all(1111)

    conf_path='yamls/mosaic_gpt/125m.yaml'
    with open(conf_path) as f:
        test_cfg = om.load(f)

    test_cfg.model.attn_impl = attention_type
    test_cfg.model.device = 'cuda:0'
    test_cfg.device = 'cuda:0'

    model_1 = COMPOSER_MODEL_REGISTRY[test_cfg.model.name](test_cfg.model).to(test_cfg.model.device)
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
            assert output_1.allclose(output_2, rtol=0.0, atol=0.0), f'differed at step {i}'

            loss_1 = model_1.loss(output_1, batch)
            loss_2 = model_2.loss(output_2, batch)
            assert loss_1 == loss_2
            loss_1.backward()
            loss_2.backward()
            optimizer_1.step()
            optimizer_2.step()


