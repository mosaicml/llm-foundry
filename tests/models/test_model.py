# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
import contextlib
import copy
import os
import pathlib
import warnings
from functools import partial
from typing import Any, Dict, List, Optional, Union, cast
from unittest import mock

import pytest
import torch
import torch.nn as nn
from accelerate import init_empty_weights
from composer.core.precision import Precision, get_precision_context
from composer.distributed.dist_strategy import prepare_fsdp_module
from composer.models.huggingface import maybe_get_underlying_model
from composer.optim import DecoupledAdamW
from composer.utils import (
    FSDPConfig,
    dist,
    get_device,
    reproducibility,
)
from omegaconf import DictConfig, ListConfig
from omegaconf import OmegaConf as om
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    pipeline,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.bloom.modeling_bloom import build_alibi_tensor

from llmfoundry import ComposerHFCausalLM
from llmfoundry.layers_registry import norms
from llmfoundry.models.hf.model_wrapper import HuggingFaceModelWithFSDP
from llmfoundry.models.layers import build_alibi_bias
from llmfoundry.models.layers.attention import (
    check_alibi_support,
    is_flash_v2_installed,
)
from llmfoundry.models.layers.blocks import MPTBlock
from llmfoundry.models.mpt import MPTConfig, MPTForCausalLM, MPTModel
from llmfoundry.models.mpt.modeling_mpt import HFRotaryEmbeddingMP
from llmfoundry.utils import build_tokenizer
from llmfoundry.utils.builders import build_composer_model
from llmfoundry.utils.config_utils import to_dict_container


def get_config(
    conf_path: str = 'scripts/train/yamls/pretrain/testing.yaml',
) -> DictConfig:
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    print(conf_path)
    with open(conf_path) as f:
        test_cfg = om.load(f)
    return cast(DictConfig, test_cfg)


def _load_tokenizer_cfg(cfg: Union[Dict[str, Any], DictConfig]) -> Dict:
    if isinstance(cfg, DictConfig):
        config = to_dict_container(cfg)
    else:
        assert isinstance(cfg, dict)
        config = cfg
    return config


def _get_objs(
    request: pytest.FixtureRequest,
    conf_path: str = 'scripts/train/yamls/pretrain/testing.yaml',
    model_config_overrides: Optional[Dict] = None,
    attn_impl: str = 'torch',
):
    warnings.filterwarnings(
        action='ignore',
        message='Torchmetrics v0.9 introduced a new argument class property',
    )
    test_cfg = get_config(conf_path=conf_path)
    if model_config_overrides is not None:
        for k, v in model_config_overrides.items():
            test_cfg.model[k] = v

    # Read FSDP Config as a dict
    fsdp_config = test_cfg.get('fsdp_config', None)
    fsdp_config = om.to_container(
        fsdp_config,
        resolve=True,
    ) if fsdp_config else None

    # Check if we are running on GPU
    is_gpu = False
    for item in request.session.items:
        is_gpu |= item.get_closest_marker('gpu') is not None

    # Build Model
    # For fast initialization, use `meta` device
    print('Initializing model...')
    device = 'cuda' if is_gpu else 'cpu'
    test_cfg.precision = 'amp_bf16' if is_gpu else 'fp32'
    test_cfg.model.attn_config = {
        'attn_impl': attn_impl,
    }
    test_cfg.model.init_device = device
    test_cfg.device = device

    test_cfg.global_train_batch_size = 2
    test_cfg.device_eval_batch_size = 2
    test_cfg.device_train_microbatch_size = 2

    tokenizer_cfg: Dict[str, Any] = _load_tokenizer_cfg(test_cfg.tokenizer)
    tokenizer = build_tokenizer(
        test_cfg.tokenizer.name,
        tokenizer_cfg.get('kwargs', {}),
    )

    name = test_cfg.model.pop('name')
    model = build_composer_model(
        name=name,
        cfg=to_dict_container(test_cfg.model),
        tokenizer=tokenizer,
    )

    # Optimizer
    assert test_cfg.optimizer.name == 'decoupled_adamw'
    optimizer = DecoupledAdamW(
        model.parameters(),
        lr=test_cfg.optimizer.lr,
        betas=test_cfg.optimizer.betas,
        eps=test_cfg.optimizer.eps,
        weight_decay=test_cfg.optimizer.weight_decay,
    )

    return test_cfg, model, optimizer


def gen_random_batch(
    batch_size: int,
    test_cfg: Union[DictConfig, ListConfig],
    inputs: Optional[List[str]] = None,
):
    # inputs can be [], ['input_ids'], ['input_ids', 'inputs_embeds'], and ['inputs_embeds']
    # default to only input ids
    if inputs == None:
        inputs = ['input_ids']
    # generate input batch of random data, suitable for a Causal LM
    batch = {}
    for inp in inputs:
        if inp == 'input_ids':
            batch['input_ids'] = torch.randint(
                low=0,
                high=test_cfg.model.vocab_size,
                size=(batch_size, test_cfg.max_seq_len),
            ).to(test_cfg.device)
        if inp == 'inputs_embeds':
            batch['inputs_embeds'] = torch.randn(
                batch_size,
                test_cfg.max_seq_len,
                test_cfg.model.d_model,
            ).to(test_cfg.device)

    batch['labels'] = torch.randint(
        low=0,
        high=test_cfg.model.vocab_size,
        size=(batch_size, test_cfg.max_seq_len),
    ).to(test_cfg.device)
    batch['attention_mask'] = torch.ones(
        size=(batch_size, test_cfg.max_seq_len),
        dtype=torch.int64,
    ).to(test_cfg.device)
    return batch


def gen_random_enc_dec_batch(
    batch_size: int,
    vocab_size: int,
    max_seq_len: int,
    device: str,
):
    # generate input batch of random data, suitable for a T5
    batch = {}
    batch['input_ids'] = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, max_seq_len),
    ).to(device)
    batch['labels'] = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, max_seq_len),
    ).to(device)
    batch['decoder_input_ids'] = torch.zeros_like(batch['labels'])
    batch['decoder_input_ids'][:, 1:] = batch['labels'][:, :-1]
    batch['attention_mask'] = torch.ones(
        size=(batch_size, max_seq_len),
        dtype=torch.int64,
    ).to(device)
    batch['decoder_attention_mask'] = batch['attention_mask'].clone()
    return batch


@pytest.mark.parametrize(
    'conf_path',
    [
        'scripts/train/yamls/pretrain/testing.yaml',
    ],
)
def test_full_forward_and_backward(
    request: pytest.FixtureRequest,
    conf_path: str,
    batch_size: int = 2,
):
    test_cfg, model, optimizer = _get_objs(request=request, conf_path=conf_path)

    batch = gen_random_batch(batch_size, test_cfg)

    assert batch['input_ids'].shape == torch.Size([
        batch_size,
        test_cfg.max_seq_len,
    ])
    model.train()
    original_params = next(model.parameters()).clone().data
    outputs = model(batch)
    loss = model.loss(outputs, batch)
    assert isinstance(loss, torch.Tensor)
    loss.backward()
    optimizer.step()
    updated_params = next(model.parameters()).clone().data
    assert not torch.equal(original_params, updated_params)


def test_full_forward_and_backward_with_inputs_embeds(
    request: pytest.FixtureRequest,
    batch_size: int = 2,
):
    test_cfg, model, optimizer = _get_objs(
        request=request,
        conf_path='scripts/train/yamls/pretrain/testing.yaml',
    )

    batch = gen_random_batch(batch_size, test_cfg, inputs=['inputs_embeds'])

    model.train()
    original_params = next(model.parameters()).clone().data
    outputs = model(batch)
    loss = model.loss(outputs, batch)
    assert isinstance(loss, torch.Tensor)
    loss.backward()
    optimizer.step()
    updated_params = next(model.parameters()).clone().data
    assert not torch.equal(original_params, updated_params)


@pytest.mark.parametrize('inputs', [[], ['input_ids', 'inputs_embeds']])
def test_invalid_inputs_embeds_input_ids_combinations(
    request: pytest.FixtureRequest,
    inputs: List[str],
):
    test_cfg, model, _ = _get_objs(
        request=request,
        conf_path='scripts/train/yamls/pretrain/testing.yaml',
    )

    batch = gen_random_batch(2, test_cfg, inputs=inputs)

    model.train()
    with pytest.raises(ValueError):
        _ = model(batch)


@pytest.mark.parametrize(
    'conf_path',
    [
        'scripts/train/yamls/pretrain/testing.yaml',
        pytest.param(
            'scripts/train/yamls/pretrain/testing-moe.yaml',
            marks=pytest.mark.gpu,
        ),
    ],
)
def test_attention_mechanism(
    request: pytest.FixtureRequest,
    conf_path: str,
    batch_size: int = 2,
):
    test_cfg, model, _ = _get_objs(request=request, conf_path=conf_path)

    batch = gen_random_batch(batch_size, test_cfg)

    model.eval()
    # run a partial forward where we explicitly inspect the attention_mask from the causal_attn block
    input_ids, attention_mask = batch['input_ids'], batch['attention_mask'
                                                         ].bool()

    _, S = input_ids.size()
    assert (
        S <= test_cfg.max_seq_len
    ), f'Cannot forward input with seq_len={S}, this model only supports seq_len<={test_cfg.max_seq_len}'
    pos = torch.arange(0, S, dtype=torch.long,
                       device=input_ids.device).unsqueeze(0)

    with get_precision_context(test_cfg.precision):
        tok_emb = model.model.transformer.wte(input_ids)
        pos_emb = model.model.transformer.wpe(pos)
        x = model.model.transformer.emb_drop(tok_emb + pos_emb)

        # basically the attention mask should be a tensor shape (bsz, seqlen, seqlen)
        # wih -inf along the upper triangle as well as wherever there are any pad tokens
        # and with 0 everywhere else
        expected_zerod_weights = nn.Transformer.generate_square_subsequent_mask(test_cfg.max_seq_len, device=test_cfg.device)\
            .reshape(1, test_cfg.max_seq_len, test_cfg.max_seq_len)
        expected_zerod_weights = torch.isneginf(
            torch.cat(batch_size * [expected_zerod_weights]),
        )
        torch_key_padding = torch.cat(  # type: ignore
            test_cfg.max_seq_len *
            [(~attention_mask).reshape(batch_size, 1, test_cfg.max_seq_len)],
            axis=1)
        expected_zerod_weights |= torch_key_padding

        attn_bias, attention_mask = model.model.transformer._attn_bias(
            device=x.device,
            dtype=x.dtype,
            attention_mask=attention_mask,
        )

        for block in model.model.transformer.blocks:
            a = block.norm_1(x)
            b, attention_weights, _ = block.attn(
                a,
                past_key_value=None,
                attn_bias=attn_bias,
                attention_mask=attention_mask,
                is_causal=model.model.transformer.is_causal,
                needs_weights=True,
            )

            zerod_weights = (attention_weights == 0)
            assert torch.equal(
                expected_zerod_weights.expand(*zerod_weights.shape),
                zerod_weights,
            )
            x = x + block.resid_attn_dropout(b)
            m = block.norm_2(x)
            n = block.ffn(m)
            x = x + block.resid_ffn_dropout(n)


def test_full_forward_and_backward_gpt2_small(batch_size: int = 2):
    warnings.filterwarnings(
        action='ignore',
        message='Torchmetrics v0.9 introduced a new argument class property',
    )
    conf_path = 'scripts/train/yamls/pretrain/gpt2-small.yaml'
    with open(conf_path) as f:
        neo_cfg = om.load(f)

    device = 'cpu'
    neo_cfg.device = device
    neo_cfg.max_seq_len = 256
    neo_cfg.model.name = 'hf_causal_lm'

    tokenizer_cfg: Dict[str, Any] = _load_tokenizer_cfg(neo_cfg.tokenizer)
    tokenizer = build_tokenizer(
        neo_cfg.tokenizer.name,
        tokenizer_cfg.get('kwargs', {}),
    )

    name = neo_cfg.model.pop('name')
    model = build_composer_model(
        name=name,
        cfg=to_dict_container(neo_cfg.model),
        tokenizer=tokenizer,
    ).to(device)

    assert isinstance(
        model.tokenizer,
        (PreTrainedTokenizer, PreTrainedTokenizerFast),
    )

    assert neo_cfg.optimizer.name == 'decoupled_adamw'
    optimizer = DecoupledAdamW(
        model.parameters(),
        lr=neo_cfg.optimizer.lr,
        betas=neo_cfg.optimizer.betas,
        eps=neo_cfg.optimizer.eps,
        weight_decay=neo_cfg.optimizer.weight_decay,
    )

    # set vocab size using model num_embeddings
    neo_cfg.model.vocab_size = model.model.transformer.wte.num_embeddings
    batch = gen_random_batch(batch_size, neo_cfg)

    assert batch['input_ids'].shape == torch.Size([
        batch_size,
        neo_cfg.max_seq_len,
    ])
    model.train()
    original_params = next(model.parameters()).clone().data
    outputs = model(batch)
    loss = model.loss(outputs, batch)
    assert isinstance(loss, torch.Tensor)
    loss.backward()
    optimizer.step()
    updated_params = next(model.parameters()).clone().data
    assert not torch.equal(original_params, updated_params)


def test_full_forward_and_backward_t5_small(batch_size: int = 2):
    warnings.filterwarnings(
        action='ignore',
        message='Torchmetrics v0.9 introduced a new argument class property',
    )
    conf_path = 'scripts/train/yamls/finetune/t5-small_dolly_sft.yaml'
    with open(conf_path) as f:
        t5_cfg = om.load(f)

    device = 'cpu'
    t5_cfg.device = device
    t5_cfg.max_seq_len = 16

    tokenizer_cfg: Dict[str, Any] = _load_tokenizer_cfg(t5_cfg.tokenizer)
    tokenizer = build_tokenizer(
        t5_cfg.tokenizer.name,
        tokenizer_cfg.get('kwargs', {}),
    )

    name = t5_cfg.model.pop('name')
    model = build_composer_model(
        name=name,
        cfg=to_dict_container(t5_cfg.model),
        tokenizer=tokenizer,
    ).to(device)

    assert isinstance(
        model.tokenizer,
        (PreTrainedTokenizer, PreTrainedTokenizerFast),
    )

    optimizer = DecoupledAdamW(
        model.parameters(),
        lr=t5_cfg.optimizer.lr,
        betas=t5_cfg.optimizer.betas,
        eps=t5_cfg.optimizer.eps,
        weight_decay=t5_cfg.optimizer.weight_decay,
    )

    # set vocab size using model num_embeddings
    batch = gen_random_enc_dec_batch(
        batch_size,
        model.model.config.vocab_size,
        t5_cfg.max_seq_len,
        device,
    )

    assert batch['input_ids'].shape == torch.Size([
        batch_size,
        t5_cfg.max_seq_len,
    ])
    model.train()
    original_params = next(model.parameters()).clone().data
    outputs = model(batch)
    loss = model.loss(outputs, batch)
    assert isinstance(loss, torch.Tensor)
    loss.backward()
    optimizer.step()
    updated_params = next(model.parameters()).clone().data
    assert not torch.equal(original_params, updated_params)


@pytest.mark.gpu
@pytest.mark.parametrize(
    'attn_impl,precision',
    [('torch', torch.float16), ('torch', torch.bfloat16),
     pytest.param('flash', torch.float16, marks=pytest.mark.gpu),
     pytest.param('flash', torch.bfloat16, marks=pytest.mark.gpu)],
)
@pytest.mark.parametrize('ffn_type', ['mptmlp', 'mptglu'])
@pytest.mark.parametrize(
    'ffn_act_fn',
    [
        None,
        {
            'name': 'gelu',
            'approximate': 'tanh',
        },
        {
            'name': 'silu',
        },
        {
            'name': 'relu',
            'inplace': True,
        },
        pytest.param({'name': 'relu5'},
                     marks=pytest.mark.xfail(
                         reason='invalid choice.',
                         strict=True,
                     )),
    ],
)
def test_determinism(
    attn_impl: str,
    precision: torch.dtype,
    ffn_type: str,
    ffn_act_fn: dict,
):
    conf_path = 'scripts/train/yamls/pretrain/testing.yaml'
    with open(conf_path) as f:
        test_cfg = om.load(f)

    test_cfg.model.attn_config = {
        'attn_impl': attn_impl,
    }
    if hasattr(test_cfg.model, 'ffn_config'):
        test_cfg.model.ffn_config['ffn_type'] = ffn_type
    else:
        test_cfg.model.setdefault('ffn_config', {'ffn_type': ffn_type})
    test_cfg.model.ffn_config['ffn_act_fn'] = ffn_act_fn
    test_cfg.model.init_device = 'cuda:0'
    test_cfg.device = 'cuda:0'

    tokenizer_cfg: Dict[str, Any] = _load_tokenizer_cfg(test_cfg.tokenizer)
    tokenizer = build_tokenizer(
        test_cfg.tokenizer.name,
        tokenizer_cfg.get('kwargs', {}),
    )

    name = test_cfg.model.pop('name')
    model_1 = build_composer_model(
        name=name,
        cfg=to_dict_container(test_cfg.model),
        tokenizer=tokenizer,
    )
    model_2 = copy.deepcopy(model_1)

    optimizer_1 = DecoupledAdamW(
        model_1.parameters(),
        lr=test_cfg.optimizer.lr,
        betas=test_cfg.optimizer.betas,
        eps=test_cfg.optimizer.eps,
        weight_decay=test_cfg.optimizer.weight_decay,
    )
    optimizer_2 = DecoupledAdamW(
        model_2.parameters(),
        lr=test_cfg.optimizer.lr,
        betas=test_cfg.optimizer.betas,
        eps=test_cfg.optimizer.eps,
        weight_decay=test_cfg.optimizer.weight_decay,
    )

    for i in range(5):
        with torch.cuda.amp.autocast(True, precision):
            batch = gen_random_batch(2, test_cfg)
            output_1 = model_1(batch)
            output_2 = model_2(batch)
            assert output_1.logits.allclose(
                output_2.logits,
                rtol=0.0,
                atol=0.0,
            ), f'differed at step {i}'
            loss_1 = model_1.loss(output_1, batch)
            loss_2 = model_2.loss(output_2, batch)
            assert isinstance(loss_1, torch.Tensor)
            assert isinstance(loss_2, torch.Tensor)
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
    from torch.backends import cuda, cudnn
    cuda.matmul.allow_tf32 = False
    cudnn.allow_tf32 = False

    conf_path = 'scripts/train/yamls/pretrain/testing.yaml'
    with open(conf_path) as f:
        test_cfg = om.load(f)

    assert isinstance(test_cfg, DictConfig)

    test_cfg.device = 'cuda:0'
    test_cfg.model.init_device = 'cpu'
    test_cfg.model.init_config = {
        'name': 'baseline_',
        'init_std': 0.02,
    }

    tokenizer_cfg: Dict[str, Any] = _load_tokenizer_cfg(test_cfg.tokenizer)
    tokenizer = build_tokenizer(
        test_cfg.tokenizer.name,
        tokenizer_cfg.get('kwargs', {}),
    )

    name = test_cfg.model.pop('name')
    model_1 = build_composer_model(
        name=name,
        cfg=to_dict_container(test_cfg.model),
        tokenizer=tokenizer,
    )
    model_2 = copy.deepcopy(model_1)

    model_1.to(test_cfg.device)
    model_2.to(test_cfg.device)

    assert isinstance(model_1.loss_fn, torch.nn.CrossEntropyLoss)
    model_2.loss_fn = FusedCrossEntropyLoss(ignore_index=-100, reduction='none')

    optimizer_1 = DecoupledAdamW(
        model_1.parameters(),
        lr=test_cfg.optimizer.lr,
        betas=test_cfg.optimizer.betas,
        eps=test_cfg.optimizer.eps,
        weight_decay=test_cfg.optimizer.weight_decay,
    )
    optimizer_2 = DecoupledAdamW(
        model_2.parameters(),
        lr=test_cfg.optimizer.lr,
        betas=test_cfg.optimizer.betas,
        eps=test_cfg.optimizer.eps,
        weight_decay=test_cfg.optimizer.weight_decay,
    )

    for i in range(15):
        batch = gen_random_batch(2, test_cfg)
        output_1 = model_1(batch)
        output_2 = model_2(batch)
        assert output_1.logits.allclose(
            output_2.logits,
            rtol=1e-4,
            atol=1e-4,
        ), f'differed at step {i}'

        loss_1 = model_1.loss(output_1, batch)
        loss_2 = model_2.loss(output_2, batch)
        assert isinstance(loss_1, torch.Tensor)
        assert isinstance(loss_2, torch.Tensor)
        assert loss_1.allclose(
            loss_2,
            rtol=1e-3,
            atol=1e-3,
        ), f'differed at step {i}'
        loss_1.backward()
        loss_2.backward()
        optimizer_1.step()
        optimizer_2.step()

        for p1, p2 in zip(model_1.parameters(), model_2.parameters()):
            assert p1.data.shape == p2.data.shape
            assert p1.data.allclose(
                p2.data,
                rtol=1e-5,
                atol=1e-4,
            ), f'differed at step {i}'


@pytest.mark.gpu
@pytest.mark.parametrize(
    'loss_fn_config',
    ['torch_crossentropy', 'fused_crossentropy'],
)
def test_loss_reduction(loss_fn_config: str):
    """Tests the Fused CrossEntropy vs torch.nn.CrossEntropy loss function.

    We provide non-zero tolerances to account for small numerics differences
    between the two loss implementations.
    """
    try:
        from flash_attn.losses.cross_entropy import CrossEntropyLoss as FusedCrossEntropyLoss  # type: ignore # isort: skip
    except:
        pytest.skip('Fused cross entropy was not installed')

    # run numerical test in pure fp32
    from torch.backends import cuda, cudnn
    cuda.matmul.allow_tf32 = False
    cudnn.allow_tf32 = False

    conf_path = 'scripts/train/yamls/pretrain/testing.yaml'
    with open(conf_path) as f:
        test_cfg = om.load(f)

    assert isinstance(test_cfg, DictConfig)

    test_cfg.model.loss_fn = loss_fn_config

    test_cfg.device = 'cuda:0'
    test_cfg.model.init_device = 'cpu'
    test_cfg.model.init_config = {
        'name': 'baseline_',
        'init_std': 0.02,
    }

    tokenizer_cfg: Dict[str, Any] = _load_tokenizer_cfg(test_cfg.tokenizer)
    tokenizer = build_tokenizer(
        test_cfg.tokenizer.name,
        tokenizer_cfg.get('kwargs', {}),
    )

    name = test_cfg.model.pop('name')
    model_1 = build_composer_model(
        name=name,
        cfg=to_dict_container(test_cfg.model),
        tokenizer=tokenizer,
    )
    model_2 = copy.deepcopy(model_1)

    model_1.to(test_cfg.device)
    model_2.to(test_cfg.device)

    # Reduce the loss in FusedCrossEntropyLoss
    if loss_fn_config == 'fused_crossentropy':
        assert isinstance(model_1.loss_fn, FusedCrossEntropyLoss)
        model_2.loss_fn = FusedCrossEntropyLoss(
            ignore_index=-100,
            reduction='mean',
        )
    else:
        assert isinstance(model_1.loss_fn, torch.nn.CrossEntropyLoss)
        model_2.loss_fn = torch.nn.CrossEntropyLoss(
            ignore_index=-100,
            reduction='mean',
        )

    optimizer_1 = DecoupledAdamW(
        model_1.parameters(),
        lr=test_cfg.optimizer.lr,
        betas=test_cfg.optimizer.betas,
        eps=test_cfg.optimizer.eps,
        weight_decay=test_cfg.optimizer.weight_decay,
    )
    optimizer_2 = DecoupledAdamW(
        model_2.parameters(),
        lr=test_cfg.optimizer.lr,
        betas=test_cfg.optimizer.betas,
        eps=test_cfg.optimizer.eps,
        weight_decay=test_cfg.optimizer.weight_decay,
    )

    for i in range(3):
        batch = gen_random_batch(2, test_cfg)
        output_1 = model_1(batch)
        output_2 = model_2(batch)
        assert output_1.logits.allclose(
            output_2.logits,
            rtol=1e-4,
            atol=1e-4,
        ), f'differed at step {i}'

        loss_1 = model_1.loss(output_1, batch)

        # Loss for model_2 gets reduced within the loss_fn, so we handle it separately
        targets = model_2.get_targets(batch)
        loss_2 = model_2.loss_fn(
            output_2.logits.view(-1, output_2.logits.size(-1)),
            targets.view(-1),
        )

        assert isinstance(loss_1, torch.Tensor)
        assert isinstance(loss_2, torch.Tensor)
        assert loss_1.allclose(
            loss_2,
            rtol=1e-3,
            atol=1e-3,
        ), f'differed at step {i}'
        loss_1.backward()
        loss_2.backward()
        optimizer_1.step()
        optimizer_2.step()

        for p1, p2 in zip(model_1.parameters(), model_2.parameters()):
            assert p1.data.shape == p2.data.shape
            assert p1.data.allclose(
                p2.data,
                rtol=1e-5,
                atol=1e-4,
            ), f'differed at step {i}'


@pytest.mark.parametrize(
    'peft_config',
    [
        None,
        {
            'peft_type': 'LORA',
            'task_type': 'CAUSAL_LM',
        },
    ],
)
def test_opt_wrapping(peft_config: Optional[dict[str, str]]):
    if peft_config is not None:
        _ = pytest.importorskip('peft')

    conf: dict[str, dict[str, Any]] = {
        'model': {
            'name': 'hf_causal_lm',
            'pretrained_model_name_or_path': 'facebook/opt-125m',
            'pretrained': False,
        },
        'tokenizer': {
            'name': 'facebook/opt-125m',
        },
    }
    if peft_config is not None:
        conf['model']['peft_config'] = peft_config

    tokenizer_cfg: Dict[str, Any] = _load_tokenizer_cfg(conf['tokenizer'])
    tokenizer = build_tokenizer(
        conf['tokenizer']['name'],
        tokenizer_cfg.get('kwargs', {}),
    )

    conf['model'].pop('name')
    model = ComposerHFCausalLM(**conf['model'], tokenizer=tokenizer)

    # check that all the modules we except are blocked from FSDP wrapping
    underlying_model = maybe_get_underlying_model(model.model)
    assert not underlying_model.model._fsdp_wrap
    assert not underlying_model.model.decoder._fsdp_wrap
    assert not underlying_model.model.decoder.embed_tokens._fsdp_wrap
    assert not underlying_model.lm_head._fsdp_wrap


def test_lora_id():
    peft = pytest.importorskip('peft')

    conf: dict[str, dict[str, Union[str, dict]]] = {
        'model': {
            'name': 'hf_causal_lm',
            'pretrained_model_name_or_path': 'facebook/opt-350m',
            'pretrained': 'false',
            'pretrained_lora_id_or_path': 'ybelkada/opt-350m-lora',
        },
        'tokenizer': {
            'name': 'facebook/opt-350m',
        },
    }

    config = DictConfig(conf)

    tokenizer_cfg: Dict[str, Any] = _load_tokenizer_cfg(config.tokenizer)
    tokenizer = build_tokenizer(
        config.tokenizer.name,
        tokenizer_cfg.get('kwargs', {}),
    )

    config.model.pop('name')
    model = ComposerHFCausalLM(**config.model, tokenizer=tokenizer)

    assert isinstance(model.model, peft.PeftModelForCausalLM)


@pytest.mark.parametrize('norm_type', norms.get_all())
@pytest.mark.parametrize('no_bias', [False, True])
@pytest.mark.parametrize('tie_word_embeddings', [True, False])
@pytest.mark.parametrize(
    'expansion_ratio,ffn_hidden_size',
    [
        (2, None),
        pytest.param(
            1.231,
            None,
            marks=pytest.mark.xfail(
                reason='d_model * expansion_ratio must be an integer.',
                strict=True,
            ),
        ),
        (2, 128),
        (2, 256),
    ],
)
@pytest.mark.parametrize(
    'ffn_act_fn',
    [
        None,
        {
            'name': 'gelu',
            'approximate': 'tanh',
        },
        {
            'name': 'silu',
        },
        {
            'name': 'relu',
            'inplace': True,
        },
        pytest.param({'name': 'relu5'},
                     marks=pytest.mark.xfail(
                         reason='invalid choice.',
                         strict=True,
                     )),
    ],
)
def test_mpt_creation(
    norm_type: str,
    no_bias: bool,
    tie_word_embeddings: bool,
    expansion_ratio: Union[int, float],
    ffn_hidden_size: int,
    ffn_act_fn: dict,
):
    if norm_type == 'triton_rmsnorm' and not is_flash_v2_installed():
        pytest.skip(
            f'norm_type=triton_rmsnorm requires flash Attention to be installed',
        )

    # Test that the config constructs the model as expected.
    hf_config = MPTConfig(
        init_device='cpu',
        d_model=128,
        n_heads=4,
        n_layers=2,
        expansion_ratio=expansion_ratio,
        max_seq_len=2048,
        emb_pdrop=0.1,
        resid_pdrop=0.2,
        attn_config={
            'attn_impl': 'torch',
        },
        norm_type=norm_type,
        no_bias=no_bias,
        tie_word_embeddings=tie_word_embeddings,
        ffn_config={
            'ffn_type': 'mptmlp',
            'ffn_hidden_size': ffn_hidden_size,
            'ffn_act_fn': ffn_act_fn,
        },
    )

    mpt = MPTForCausalLM(hf_config)

    assert mpt.config.d_model == 128
    assert mpt.config.n_heads == 4
    assert mpt.config.n_layers == 2
    if ffn_hidden_size is None:
        assert mpt.config.expansion_ratio == expansion_ratio
    else:
        assert mpt.config.ffn_config['ffn_hidden_size'] == ffn_hidden_size
    assert mpt.config.max_seq_len == 2048

    assert mpt.transformer.wte.weight.shape == torch.Size([
        hf_config.vocab_size,
        hf_config.d_model,
    ])
    if not tie_word_embeddings:
        assert mpt.lm_head is not None
        assert mpt.lm_head.weight.shape == mpt.transformer.wte.weight.shape
    assert mpt.transformer.wpe.weight.shape == torch.Size([
        hf_config.max_seq_len,
        hf_config.d_model,
    ])
    assert mpt.transformer.emb_drop.p == 0.1
    assert len(mpt.transformer.blocks) == 2

    d_model = hf_config.d_model
    if ffn_hidden_size is None:  # type: ignore (sometimes it may not be none)
        ffn_hidden_size = int(hf_config.d_model * hf_config.expansion_ratio)
    for block in mpt.transformer.blocks:
        assert isinstance(block, MPTBlock)
        assert block.norm_1.weight.shape == torch.Size([d_model])
        assert block.norm_2 is not None
        assert block.norm_2.weight.shape == torch.Size([d_model])
        assert isinstance(block.ffn.up_proj, nn.Linear)
        assert block.ffn.up_proj.weight.shape == torch.Size([
            ffn_hidden_size,
            hf_config.d_model,
        ])
        assert isinstance(block.ffn.down_proj, nn.Linear)
        assert block.ffn.down_proj.weight.shape == torch.Size([
            hf_config.d_model,
            ffn_hidden_size,
        ])
        assert block.resid_attn_dropout.p == 0.2
        assert block.resid_ffn_dropout.p == 0.2


@pytest.mark.gpu
def test_mb_mpt_creation():
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
        norm_type='low_precision_layernorm',
        no_bias=True,
        tie_word_embeddings=False,
        ffn_config={
            'ffn_type': 'mb_moe',
            'ffn_hidden_size': 1024,
            'ffn_act_fn': {
                'name': 'gelu',
            },
            'moe_world_size': 1,
        },
    )

    _ = MPTForCausalLM(hf_config)


@pytest.mark.gpu
@pytest.mark.parametrize('attention_impl', ['flash', 'torch'])
@pytest.mark.parametrize(
    'pos_emb_config',
    [{
        'alibi': True,
        'rope': False,
    }, {
        'alibi': False,
        'rope': True,
        'rope_theta': 10000,
        'rope_impl': 'dail',
        'rope_dail_config': {
            'type': 'original',
            'pos_idx_in_fp32': True,
            'xpos_scale_base': 512,
        },
    }, {
        'alibi': False,
        'rope': True,
        'rope_theta': 10000,
        'rope_impl': 'hf',
        'rope_hf_config': {
            'type': 'no_scaling',
            'factor': 1.0,
        },
    }],
)
def test_sequence_id_based_masking(attention_impl: str, pos_emb_config: dict):
    # Testing the output of concatenated sequence with sequence id masking vs individual sequences.
    alibi = pos_emb_config['alibi']
    if alibi and not check_alibi_support(attention_impl):
        pytest.skip(f'flash attention below v2.4.2 does not support alibi.')

    rope = pos_emb_config['rope']
    if rope and pos_emb_config['rope_impl'
                              ] == 'dail' and not is_flash_v2_installed():
        pytest.skip(
            f'dail implementation of rope requires gpu and flash attention 2.',
        )

    if attention_impl == 'flash' and (
        not is_flash_v2_installed(v2_version='v2.1.2')
    ):
        pytest.skip(
            'Using sequence id with flash attention requires flash attention v2.1.2 or higher.',
        )

    composer_device = get_device(None)

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
            'attn_uses_sequence_id': True,
            **pos_emb_config,
        },
        init_config={
            'name': 'baseline_',
            'init_std': 0.02,
        },
    )
    mpt = MPTForCausalLM(hf_config)
    mpt.eval()
    mpt = composer_device.module_to_device(mpt)

    with get_precision_context(
        'amp_bf16' if composer_device.name == 'gpu' else 'fp32',
    ):
        # padding on the right side of the input
        concatenated_seq_ids = torch.tensor([[11274, 16390, 11, 4332, 323, 423],
                                             [2342, 12, 111, 123, 50256, 342]])
        concatenated_seq_ids = composer_device.tensor_to_device(
            concatenated_seq_ids,
        )

        sequence_id = torch.tensor([[0, 0, 0, 1, 2, 2], [0, 0, 0, 1, 2, 2]])
        sequence_id = composer_device.tensor_to_device(sequence_id)

        first_seq_ids = torch.tensor([[11274, 16390, 11], [2342, 12, 111]])
        first_seq_ids = composer_device.tensor_to_device(first_seq_ids)

        second_seq_ids = torch.tensor([[4332], [123]])
        second_seq_ids = composer_device.tensor_to_device(second_seq_ids)

        third_seq_ids = torch.tensor([[323, 423], [50256, 342]])
        third_seq_ids = composer_device.tensor_to_device(third_seq_ids)

        concatenated_seq_output = mpt(
            concatenated_seq_ids,
            sequence_id=sequence_id,
        ).logits
        first_seq_output = mpt(first_seq_ids).logits
        second_seq_output = mpt(second_seq_ids).logits
        third_seq_output = mpt(third_seq_ids).logits

        assert torch.allclose(
            concatenated_seq_output[:, :3],
            first_seq_output,
            atol=2e-6 if attention_impl == 'torch' else 1e-8,
        )
        assert torch.allclose(
            concatenated_seq_output[:, 3:4],
            second_seq_output,
            atol=2e-6 if attention_impl == 'torch' else 1e-8,
        )
        atol = 1e-8
        if attention_impl == 'torch':
            atol = 2e-6
        elif pos_emb_config['rope']:
            atol = 2e-2
        assert torch.allclose(
            concatenated_seq_output[:, 4:6],
            third_seq_output,
            atol=atol,
        )


@pytest.mark.parametrize(
    'attention_impl',
    [
        'torch',
        pytest.param('flash', marks=pytest.mark.gpu),
        pytest.param('torch', marks=pytest.mark.gpu),
    ],
)
@pytest.mark.parametrize(
    'pos_emb_config',
    [{
        'alibi': False,
        'rope': False,
    }, {
        'alibi': True,
        'rope': False,
    }, {
        'alibi': False,
        'rope': True,
        'rope_theta': 10000,
        'rope_impl': 'dail',
        'rope_dail_config': {
            'type': 'original',
            'pos_idx_in_fp32': True,
            'xpos_scale_base': 512,
        },
    }, {
        'alibi': False,
        'rope': True,
        'rope_theta': 10000,
        'rope_impl': 'hf',
        'rope_hf_config': {
            'type': 'no_scaling',
            'factor': 1.0,
        },
    }],
)
@pytest.mark.parametrize('tie_word_embeddings', [True, False])
def test_forward_with_padding(
    attention_impl: str,
    pos_emb_config: dict,
    tie_word_embeddings: bool,
):
    # Test that different placement of padding does not affect the output.
    alibi = pos_emb_config['alibi']
    if alibi and not check_alibi_support(attention_impl):
        pytest.skip(f'flash attention below v2.4.2 does not support alibi.')

    rope = pos_emb_config['rope']
    if rope and pos_emb_config['rope_impl'
                              ] == 'dail' and not is_flash_v2_installed():
        pytest.skip(
            f'dail implementation of rope requires gpu and flash attention 2.',
        )

    composer_device = get_device(None)

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
            **pos_emb_config,
        },
        init_config={
            'name': 'baseline_',
            'init_std': 0.02,
        },
        tie_word_embeddings=tie_word_embeddings,
    )
    mpt = MPTForCausalLM(hf_config)
    mpt.eval()
    mpt = composer_device.module_to_device(mpt)

    with get_precision_context(
        'amp_bf16' if composer_device.name == 'gpu' else 'fp32',
    ):
        # padding on the right side of the input
        right_padding_input_ids = torch.tensor([[
            11274,
            16390,
            11,
            50256,
            50256,
            50256,
        ], [11274, 16390, 11, 50256, 50256, 50256]])
        right_padding_input_ids = composer_device.tensor_to_device(
            right_padding_input_ids,
        )
        right_padding_attention_mask = torch.tensor([[1, 1, 1, 0, 0, 0],
                                                     [1, 1, 1, 0, 0, 0]]).bool()
        right_padding_attention_mask = composer_device.tensor_to_device(
            right_padding_attention_mask,
        )

        # padding in the middle of the input
        middle_padding_input_ids = torch.tensor([[
            11274,
            16390,
            50256,
            50256,
            50256,
            11,
        ], [11274, 16390, 50256, 50256, 50256, 11]])
        middle_padding_input_ids = composer_device.tensor_to_device(
            middle_padding_input_ids,
        )
        middle_padding_attention_mask = torch.tensor([[1, 1, 0, 0, 0, 1],
                                                      [1, 1, 0, 0, 0,
                                                       1]]).bool()
        middle_padding_attention_mask = composer_device.tensor_to_device(
            middle_padding_attention_mask,
        )

        # padding on the left side of the input
        left_padding_input_ids = torch.tensor([[
            50256,
            50256,
            50256,
            11274,
            16390,
            11,
        ], [50256, 50256, 50256, 11274, 16390, 11]])
        left_padding_input_ids = composer_device.tensor_to_device(
            left_padding_input_ids,
        )
        left_padding_attention_mask = torch.tensor([[0, 0, 0, 1, 1, 1],
                                                    [0, 0, 0, 1, 1, 1]]).bool()
        left_padding_attention_mask = composer_device.tensor_to_device(
            left_padding_attention_mask,
        )

        # a single batch with padding in different places
        batched_input_ids = torch.tensor([
            [11274, 16390, 11, 50256, 50256, 50256],  # right padding
            [11274, 16390, 50256, 50256, 50256, 11],
        ])  # middle padding
        batched_input_ids = composer_device.tensor_to_device(batched_input_ids)
        batched_attention_mask = torch.tensor([[1, 1, 1, 0, 0, 0],
                                               [1, 1, 0, 0, 0, 1]]).bool()
        batched_attention_mask = composer_device.tensor_to_device(
            batched_attention_mask,
        )

        right_padding_output = mpt(
            right_padding_input_ids,
            attention_mask=right_padding_attention_mask,
        ).logits
        middle_padding_output = mpt(
            middle_padding_input_ids,
            attention_mask=middle_padding_attention_mask,
        ).logits
        left_padding_output = mpt(
            left_padding_input_ids,
            attention_mask=left_padding_attention_mask,
        ).logits
        batched_output = mpt(
            batched_input_ids,
            attention_mask=batched_attention_mask,
        ).logits

        # check that right padding and left padding produce the same output
        right_pad_v_left_pad_rtol = 1e-5
        right_pad_v_left_pad_atol = 1e-6 if attention_impl == 'torch' else 1e-8
        if rope and pos_emb_config['rope_impl'] == 'dail':
            # dail implementation of rope uses bf16 precision and hence the rotations have small numerical errors. This causes some differences between the outputs of padded and unpadded inputs.
            right_pad_v_left_pad_rtol = 2e-2
            right_pad_v_left_pad_atol = 2e-2
        assert torch.allclose(
            right_padding_output[0, :3],
            left_padding_output[0, 3:],
            rtol=right_pad_v_left_pad_rtol,
            atol=right_pad_v_left_pad_atol,
        )

        if not (alibi or (rope and pos_emb_config['rope_impl'] == 'dail')):
            # check that right padding and middle padding produce the same output
            # Note: alibi not implemented for middle padding.
            # Note: dail implementation of rope does not support middle padding.
            assert torch.allclose(
                right_padding_output[0, :3],
                middle_padding_output[0, [0, 1, 5]],
                atol=1e-6 if attention_impl == 'torch' else 1e-8,
            )

        # check that right padding and right padding in a batch produce the same output
        assert torch.allclose(
            right_padding_output[0, :3],
            batched_output[0, :3],
            atol=1e-6 if attention_impl == 'torch' else 1e-8,
        )

        if not (alibi or (rope and pos_emb_config['rope_impl'] == 'dail')):
            # check that middle padding and middle padding in a batch produce the same output
            # Note: alibi not implemented for middle padding.
            # Note: dail implementation of rope does not support middle padding.
            assert torch.allclose(
                middle_padding_output[0],
                batched_output[1, :],
                atol=1e-6 if attention_impl == 'torch' else 1e-8,
            )

        try:
            from flash_attn.bert_padding import unpad_input, pad_input  # type: ignore # yapf: disable # isort: skip
        except:
            unpad_input, pad_input = None, None

        if unpad_input is not None and pad_input is not None:
            # Checking numerical precision with pad_token ffn
            for block in mpt.transformer.blocks:
                # Flip the padding usage in the model
                block.use_pad_tok_in_ffn = not block.use_pad_tok_in_ffn

            right_padding_output_pad_flipped = mpt(
                right_padding_input_ids,
                attention_mask=right_padding_attention_mask,
            ).logits
            middle_padding_output_pad_flipped = mpt(
                middle_padding_input_ids,
                attention_mask=middle_padding_attention_mask,
            ).logits
            left_padding_output_pad_flipped = mpt(
                left_padding_input_ids,
                attention_mask=left_padding_attention_mask,
            ).logits

            pad_vs_unpad_rtol = 1e-5
            pad_vs_unpad_atol = 1e-6
            assert torch.allclose(
                right_padding_output[0, :3],
                right_padding_output_pad_flipped[0, :3],
                rtol=pad_vs_unpad_rtol,
                atol=pad_vs_unpad_atol,
            )

            assert torch.allclose(
                middle_padding_output[0, [0, 1, 5]],
                middle_padding_output_pad_flipped[0, [0, 1, 5]],
                rtol=pad_vs_unpad_rtol,
                atol=pad_vs_unpad_atol,
            )

            assert torch.allclose(
                left_padding_output[0, 3:],
                left_padding_output_pad_flipped[0, 3:],
                rtol=pad_vs_unpad_rtol,
                atol=pad_vs_unpad_atol,
            )


@pytest.mark.parametrize(
    'attention_impl,precision',
    [
        ('torch', 'fp32'),
        pytest.param('flash', 'amp_bf16', marks=pytest.mark.gpu),
        pytest.param('torch', 'amp_bf16', marks=pytest.mark.gpu),
        pytest.param('torch', 'fp32', marks=pytest.mark.gpu),
    ],
)
@pytest.mark.parametrize(
    'pos_emb_config',
    [{
        'alibi': False,
        'rope': False,
    }, {
        'alibi': True,
        'rope': False,
    }, {
        'alibi': False,
        'rope': True,
        'rope_theta': 10000,
        'rope_impl': 'dail',
        'rope_dail_config': {
            'type': 'original',
            'pos_idx_in_fp32': True,
            'xpos_scale_base': 512,
        },
    }, {
        'alibi': False,
        'rope': True,
        'rope_theta': 10000,
        'rope_impl': 'hf',
        'rope_hf_config': {
            'type': 'no_scaling',
            'factor': 1.0,
        },
    }],
)
@pytest.mark.parametrize('tie_word_embeddings', [True, False])
def test_generate(
    attention_impl: str,
    precision: str,
    pos_emb_config: dict,
    tie_word_embeddings: bool,
):
    # Test that generate works, and produces the same output with or without
    # padding in the input.
    if pos_emb_config['alibi'] and not check_alibi_support(attention_impl):
        pytest.skip(f'flash attention below v2.4.2 does not support alibi.')

    if pos_emb_config['rope'] and pos_emb_config[
        'rope_impl'] == 'dail' and not is_flash_v2_installed():
        pytest.skip(
            f'dail implementation of rope requires gpu and flash attention 2.',
        )
    if attention_impl == 'torch' and precision == 'amp_bf16' and tie_word_embeddings == False:
        pytest.skip(f'This test configuration has precision / sampling issues.')

    composer_device = get_device(None)

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
            **pos_emb_config,
        },
        tie_word_embeddings=tie_word_embeddings,
    )
    mpt = MPTForCausalLM(hf_config)
    mpt = composer_device.module_to_device(mpt)
    mpt.eval()

    # padding on the left of the input
    left_padding_input_ids = torch.tensor([[
        50256,
        50256,
        50256,
        11274,
        16390,
        11,
    ], [50256, 50256, 50256, 11274, 16390, 11]])
    left_padding_input_ids = composer_device.tensor_to_device(
        left_padding_input_ids,
    )
    left_padding_attention_mask = torch.tensor([[0, 0, 0, 1, 1, 1],
                                                [0, 0, 0, 1, 1, 1]])
    left_padding_attention_mask = composer_device.tensor_to_device(
        left_padding_attention_mask,
    )

    # no padding in the input
    no_padding_input_ids = torch.tensor([[11274, 16390, 11], [11274, 16390,
                                                              11]])
    no_padding_input_ids = composer_device.tensor_to_device(
        no_padding_input_ids,
    )
    no_padding_attention_mask = torch.tensor([[1, 1, 1], [1, 1, 1]])
    no_padding_attention_mask = composer_device.tensor_to_device(
        no_padding_attention_mask,
    )

    # inputs_embeds
    inputs_embeds = composer_device.tensor_to_device(torch.randn(2, 3, 128))

    # a single batch with different amounts of left padding in the input
    batched_input_ids = torch.tensor([[50256, 50256, 50256, 11274, 16390, 11],
                                      [50256, 50256, 16, 11274, 16390, 11]])
    batched_input_ids = composer_device.tensor_to_device(batched_input_ids)
    batched_attention_mask = torch.tensor([[0, 0, 0, 1, 1, 1],
                                           [0, 0, 1, 1, 1, 1]]).bool()
    batched_attention_mask = composer_device.tensor_to_device(
        batched_attention_mask,
    )

    with get_precision_context(precision):
        # check that a batch with different amounts of padding doesn't crash
        # and produces the right output shape
        batched_generation = mpt.generate(
            input_ids=batched_input_ids,
            attention_mask=batched_attention_mask,
            max_new_tokens=5,
            use_cache=False,
        )
        assert batched_generation.shape == (2, 6 + 5)

        generation_with_left_padding = mpt.generate(
            input_ids=left_padding_input_ids,
            attention_mask=left_padding_attention_mask,
            max_new_tokens=5,
            use_cache=False,
        )
        assert generation_with_left_padding.shape == (2, 6 + 5)
        generation_with_no_padding = mpt.generate(
            input_ids=no_padding_input_ids,
            attention_mask=no_padding_attention_mask,
            max_new_tokens=5,
            use_cache=False,
        )
        assert generation_with_no_padding.shape == (2, 3 + 5)

        # check that left padding and no padding produce the same output
        assert generation_with_no_padding[:, 3:].equal(
            generation_with_left_padding[:, 6:],
        )

        # check that both/neither ids and embeds do not error
        # note that we need to set the BOS token ID for generating from neither
        _ = mpt.generate(
            input_ids=no_padding_input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=no_padding_attention_mask,
            max_new_tokens=5,
            use_cache=False,
        )
        _ = mpt.generate(
            input_ids=no_padding_input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=no_padding_attention_mask,
            max_new_tokens=5,
            use_cache=True,
        )
        _ = mpt.generate(
            input_ids=None,
            max_new_tokens=5,
            use_cache=False,
            bos_token_id=50256,
        )
        _ = mpt.generate(
            input_ids=None,
            max_new_tokens=5,
            use_cache=True,
            bos_token_id=50256,
        )


@pytest.mark.gpu
@pytest.mark.parametrize('world_size', [1, 2])
@pytest.mark.parametrize('tie_word_embeddings', [True, False])
def test_generate_with_device_map(
    tmp_path: pathlib.Path,
    world_size: int,
    tie_word_embeddings: bool,
):
    if not torch.cuda.device_count() >= world_size:
        pytest.skip(f'This test requires {world_size} GPUs.')

    save_path = tmp_path / 'test-device-map'
    hf_config = MPTConfig(
        init_device='cpu',
        d_model=64,
        n_heads=4,
        n_layers=2,
        expansion_ratio=2,
        max_seq_len=4,
        emb_pdrop=0.1,
        resid_pdrop=0.2,
        attn_config={
            'attn_impl': 'torch',
        },
        use_cache=True,
        tie_word_embeddings=tie_word_embeddings,
    )
    mpt = MPTForCausalLM(hf_config)
    mpt.save_pretrained(save_path)

    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    CONFIG_MAPPING._extra_content['mpt'] = MPTConfig
    AutoModelForCausalLM.register(MPTConfig, MPTForCausalLM)
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')

    device_map = {
        'transformer.wte': 0,
        'transformer.wpe': 0,
        'transformer.embd_drop': 0,
        'transformer.blocks.0': 0,
        'transformer.blocks.1': 1 if world_size == 2 else 0,
        'transformer.norm_f': 1 if world_size == 2 else 0,
        'lm_head': 1 if world_size == 2 else 0,
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
        _ = pipe(
            'The fox',
            max_new_tokens=2,
            do_sample=True,
        )


def check_hf_model_equivalence(
    model1: PreTrainedModel,
    model2: PreTrainedModel,
):
    # Checks that two huggingface models are equivalent (config and
    # parameters)
    expected_model_config_dict = model1.config.to_dict()
    new_model_config_dict = model2.config.to_dict()

    # this key just says the folder it was loaded from, which is a tmp dir during pytest
    del expected_model_config_dict['_name_or_path']
    del new_model_config_dict['_name_or_path']

    assert expected_model_config_dict == new_model_config_dict
    assert sum(p.numel() for p in model1.parameters()
              ) == sum(p.numel() for p in model2.parameters())
    assert all(
        type(module1) == type(module2)
        for module1, module2 in zip(model1.modules(), model2.modules())
    )

    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        torch.testing.assert_close(p1, p2)


def test_save_from_pretrained(tmp_path: pathlib.Path):
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


@pytest.mark.parametrize(
    'attn_impl',
    [
        'torch',
        pytest.param('flash', marks=pytest.mark.gpu),
    ],
)
@pytest.mark.parametrize(
    'pos_emb_config',
    [{
        'alibi': False,
        'rope': False,
    }, {
        'alibi': True,
        'rope': False,
    }, {
        'alibi': False,
        'rope': True,
        'rope_theta': 10000,
        'rope_impl': 'dail',
        'rope_dail_config': {
            'type': 'original',
            'pos_idx_in_fp32': True,
            'xpos_scale_base': 512,
        },
    }, {
        'alibi': False,
        'rope': True,
        'rope_theta': 10000,
        'rope_impl': 'hf',
        'rope_hf_config': {
            'type': 'no_scaling',
            'factor': 1.0,
        },
    }],
)
def test_forward_with_cache_and_padding(attn_impl: str, pos_emb_config: dict):
    # Tests that the result is the same with or without padding when using kv caching
    if pos_emb_config['alibi'] and not check_alibi_support(attn_impl):
        pytest.skip(f'flash attention below v2.4.2 does not support alibi.')
    if pos_emb_config['rope'] and pos_emb_config[
        'rope_impl'] == 'dail' and not is_flash_v2_installed():
        pytest.skip(
            f'dail implementation of rope requires gpu and flash attention 2.',
        )

    composer_device = get_device(None)

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
            **pos_emb_config,
        },
        use_cache=True,
        init_config={
            'name': 'baseline_',
            'init_std': 0.02,
        },
        tie_word_embeddings=True,
    )

    mpt = MPTForCausalLM(hf_config)
    mpt = composer_device.module_to_device(mpt)
    mpt.eval()
    with get_precision_context(
        'amp_bf16' if composer_device.name == 'gpu' else 'fp32',
    ):
        first_input_ids_no_padding = torch.tensor([[11274, 16390, 11]])
        first_input_ids_no_padding = composer_device.tensor_to_device(
            first_input_ids_no_padding,
        )
        first_attention_mask_no_padding = torch.tensor([[1, 1, 1]]).bool()
        first_attention_mask_no_padding = composer_device.tensor_to_device(
            first_attention_mask_no_padding,
        )

        # start with passing the first three tokens through (no padding)
        first_output_no_padding = mpt(
            first_input_ids_no_padding,
            attention_mask=first_attention_mask_no_padding,
        )

        second_input_ids_no_padding = torch.tensor([[11274, 16390, 11, 11274]])
        second_input_ids_no_padding = composer_device.tensor_to_device(
            second_input_ids_no_padding,
        )
        second_attention_mask_no_padding = torch.tensor([[1, 1, 1, 1]]).bool()
        second_attention_mask_no_padding = composer_device.tensor_to_device(
            second_attention_mask_no_padding,
        )

        # pass through the fourth token by itself, using the key-value cache (no padding)
        second_output_no_padding = mpt(
            second_input_ids_no_padding[:, -1].unsqueeze(-1),
            attention_mask=second_attention_mask_no_padding,
            past_key_values=first_output_no_padding.past_key_values,
        )

        first_input_ids_padding = torch.tensor([[50256, 11274, 16390, 11]])
        first_input_ids_padding = composer_device.tensor_to_device(
            first_input_ids_padding,
        )
        first_attention_mask_padding = torch.tensor([[0, 1, 1, 1]]).bool()
        first_attention_mask_padding = composer_device.tensor_to_device(
            first_attention_mask_padding,
        )

        # start with passing the first three tokens through (with left padding)
        first_output_padding = mpt(
            first_input_ids_padding,
            attention_mask=first_attention_mask_padding,
        )

        second_input_ids_padding = torch.tensor([[
            50256,
            11274,
            16390,
            11,
            11274,
        ]])
        second_input_ids_padding = composer_device.tensor_to_device(
            second_input_ids_padding,
        )
        second_attention_mask_padding = torch.tensor([[0, 1, 1, 1, 1]]).bool()
        second_attention_mask_padding = composer_device.tensor_to_device(
            second_attention_mask_padding,
        )

        # pass through the fourth token by itself, using the key-value cache (with left padding)
        second_output_padding = mpt(
            second_input_ids_padding[:, -1].unsqueeze(-1),
            attention_mask=second_attention_mask_padding,
            past_key_values=first_output_padding.past_key_values,
        )

        # check that the outputs are the same with or without padding
        if pos_emb_config['rope'] and pos_emb_config[
            'rope_impl'
        ] == 'dail':  # dail implementation of rope uses bf16 precision and hence the rotations have small numerical errors. This causes some differences between the outputs of padded and unpadded inputs.
            torch.testing.assert_close(
                second_output_no_padding.logits,
                second_output_padding.logits[:, -1, :].unsqueeze(1),
                atol=1e-2,
                rtol=1e-6,
            )
        else:
            torch.testing.assert_close(
                second_output_no_padding.logits,
                second_output_padding.logits[:, -1, :].unsqueeze(1),
                atol=1e-6,
                rtol=1e-6,
            )


@pytest.mark.parametrize(
    'attn_impl',
    [
        'torch',
        pytest.param('flash', marks=pytest.mark.gpu),
    ],
)
@pytest.mark.parametrize(
    'pos_emb_config',
    [{
        'alibi': False,
        'rope': False,
    }, {
        'alibi': True,
        'rope': False,
    }, {
        'alibi': False,
        'rope': True,
        'rope_theta': 10000,
        'rope_impl': 'dail',
        'rope_dail_config': {
            'type': 'original',
            'pos_idx_in_fp32': True,
            'xpos_scale_base': 512,
        },
    }, {
        'alibi': False,
        'rope': True,
        'rope_theta': 10000,
        'rope_impl': 'hf',
        'rope_hf_config': {
            'type': 'no_scaling',
            'factor': 1.0,
        },
    }],
)
@pytest.mark.parametrize('tie_word_embeddings', [True, False])
def test_forward_with_cache(
    attn_impl: str,
    pos_emb_config: dict,
    tie_word_embeddings: bool,
):
    # Test that model forward with and without the key-value cache produces the
    # same output.
    if pos_emb_config['alibi'] and not check_alibi_support(attn_impl):
        pytest.skip(f'flash attention below v2.4.2 does not support alibi.')

    if pos_emb_config['rope'] and pos_emb_config[
        'rope_impl'] == 'dail' and not is_flash_v2_installed():
        pytest.skip(
            f'dail implementation of rope requires gpu and flash attention 2.',
        )

    composer_device = get_device(None)

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
            **pos_emb_config,
        },
        use_cache=True,
        init_config={
            'name': 'baseline_',
            'init_std': 0.02,
        },
        tie_word_embeddings=tie_word_embeddings,
    )
    mpt = MPTForCausalLM(hf_config)
    mpt = composer_device.module_to_device(mpt)
    mpt.eval()

    with get_precision_context(
        'amp_bf16' if composer_device.name == 'gpu' else 'fp32',
    ):
        first_input_ids = torch.tensor([[11274, 16390, 11]])
        first_input_ids = composer_device.tensor_to_device(first_input_ids)
        first_attention_mask = torch.tensor([[1, 1, 1]]).bool()
        first_attention_mask = composer_device.tensor_to_device(
            first_attention_mask,
        )

        # start with passing the first three tokens through
        first_output = mpt(first_input_ids, attention_mask=first_attention_mask)

        assert first_output.logits.shape == (1, 3, hf_config.vocab_size)
        assert len(first_output.past_key_values) == hf_config.n_layers
        assert all(
            len(past_key_value) == 2
            for past_key_value in first_output.past_key_values
        )
        if attn_impl == 'torch':
            assert all(
                past_key_value[0].shape == (1, 4, 32, 3)
                for past_key_value in first_output.past_key_values
            )
            assert all(
                past_key_value[1].shape == (1, 4, 3, 32)
                for past_key_value in first_output.past_key_values
            )
        else:
            assert all(
                past_key_value[0].shape == (1, 3, 128)
                for past_key_value in first_output.past_key_values
            )
            assert all(
                past_key_value[1].shape == (1, 3, 128)
                for past_key_value in first_output.past_key_values
            )

        second_input_ids = torch.tensor([[11274, 16390, 11, 11274]])
        second_input_ids = composer_device.tensor_to_device(second_input_ids)
        second_attention_mask = torch.tensor([[1, 1, 1, 1]]).bool()
        second_attention_mask = composer_device.tensor_to_device(
            second_attention_mask,
        )

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
            for past_key_value in second_output.past_key_values
        )
        if attn_impl == 'torch':
            assert all(
                past_key_value[0].shape == (1, 4, 32, 4)
                for past_key_value in second_output.past_key_values
            )
            assert all(
                past_key_value[1].shape == (1, 4, 4, 32)
                for past_key_value in second_output.past_key_values
            )
        else:
            assert all(
                past_key_value[0].shape == (1, 4, 128)
                for past_key_value in second_output.past_key_values
            )
            assert all(
                past_key_value[1].shape == (1, 4, 128)
                for past_key_value in second_output.past_key_values
            )

        # pass through the first four tokens without the key-value cache
        full_output = mpt(
            second_input_ids,
            attention_mask=second_attention_mask,
        )

        # check that the output is the same whether using the key-value cache or not
        torch.testing.assert_close(
            second_output.logits,
            full_output.logits[:, -1, :].unsqueeze(1),
            atol=1.1e-2,
            rtol=1e-2,
        )


@pytest.mark.parametrize(
    'attn_impl',
    [
        'torch',
        pytest.param('flash', marks=pytest.mark.gpu),
    ],
)
@pytest.mark.parametrize(
    'pos_emb_config',
    [{
        'alibi': False,
        'rope': False,
    }, {
        'alibi': True,
        'rope': False,
    }, {
        'alibi': False,
        'rope': True,
        'rope_theta': 10000,
        'rope_impl': 'dail',
        'rope_dail_config': {
            'type': 'original',
            'pos_idx_in_fp32': True,
            'xpos_scale_base': 512,
        },
    }, {
        'alibi': False,
        'rope': True,
        'rope_theta': 10000,
        'rope_impl': 'hf',
        'rope_hf_config': {
            'type': 'no_scaling',
            'factor': 1.0,
        },
    }],
)
@pytest.mark.parametrize('tie_word_embeddings', [True, False])
def test_generate_with_past_kv(
    attn_impl: str,
    pos_emb_config: dict,
    tie_word_embeddings: bool,
):
    if pos_emb_config['alibi'] and not check_alibi_support(attn_impl):
        pytest.skip(f'flash attention below v2.4.2 does not support alibi.')
    if pos_emb_config['rope'] and pos_emb_config[
        'rope_impl'] == 'dail' and not is_flash_v2_installed():
        pytest.skip(
            f'dail implementation of rope requires gpu and flash attention 2.',
        )

    composer_device = get_device(None)

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
            **pos_emb_config,
        },
        use_cache=True,
        init_config={
            'name': 'baseline_',
            'init_std': 0.02,
        },
        tie_word_embeddings=tie_word_embeddings,
    )
    mpt = MPTForCausalLM(hf_config)
    mpt = composer_device.module_to_device(mpt)
    mpt.eval()

    # no padding in the input
    no_padding_input_ids = torch.tensor([[11274, 16390, 11]])
    no_padding_input_ids = composer_device.tensor_to_device(
        no_padding_input_ids,
    )
    no_padding_attention_mask = torch.tensor([[1, 1, 1]])
    no_padding_attention_mask = composer_device.tensor_to_device(
        no_padding_attention_mask,
    )

    with get_precision_context(
        'amp_bf16' if composer_device.name == 'gpu' else 'fp32',
    ):
        with mock.patch.object(
            MPTForCausalLM,
            'forward',
            autospec=True,
        ) as forward_mocked:
            forward_mocked.return_value = CausalLMOutputWithPast(
                logits=composer_device.tensor_to_device(
                    torch.randn((1, 3, hf_config.vocab_size)),
                ),
                past_key_values=[(
                    torch.randn(1, 3, hf_config.d_model),
                    torch.randn(1, 3, hf_config.d_model),
                ) for _ in range(hf_config.n_layers)],
            )
            _ = mpt.generate(
                input_ids=no_padding_input_ids,
                attention_mask=no_padding_attention_mask,
                max_new_tokens=2,
            )

            assert forward_mocked.call_count == 2
            _, _, kwargs = forward_mocked.mock_calls[0]
            assert kwargs['past_key_values'] is None
            _, _, kwargs = forward_mocked.mock_calls[1]
            assert kwargs['past_key_values'] is not None
            assert len(kwargs['past_key_values']) == hf_config.n_layers
            assert kwargs['past_key_values'][0][0].shape == (
                1,
                3,
                hf_config.d_model,
            )


@pytest.mark.parametrize(
    'attn_impl',
    [
        'torch',
        pytest.param('flash', marks=pytest.mark.gpu),
    ],
)
@pytest.mark.parametrize(
    'generation_kwargs',
    [{
        'max_new_tokens': 2,
        'num_beams': 4,
        'top_k': 5,
        'penalty_alpha': 0.4,
    }],
)
@pytest.mark.parametrize(
    'pos_emb_config',
    [{
        'alibi': False,
        'rope': False,
    }, {
        'alibi': True,
        'rope': False,
    }, {
        'alibi': False,
        'rope': True,
        'rope_theta': 10000,
        'rope_impl': 'dail',
        'rope_dail_config': {
            'type': 'original',
            'pos_idx_in_fp32': True,
            'xpos_scale_base': 512,
        },
    }, {
        'alibi': False,
        'rope': True,
        'rope_theta': 10000,
        'rope_impl': 'hf',
        'rope_hf_config': {
            'type': 'no_scaling',
            'factor': 1.0,
        },
    }],
)
@pytest.mark.parametrize('tie_word_embeddings', [True, False])
def test_generation_kwargs_dont_crash(
    attn_impl: str,
    generation_kwargs: Dict[str, Any],
    pos_emb_config: dict,
    tie_word_embeddings: bool,
):
    if pos_emb_config['alibi'] and not check_alibi_support(attn_impl):
        pytest.skip(f'flash attention below v2.4.2 does not support alibi.')

    if pos_emb_config['rope'] and pos_emb_config[
        'rope_impl'] == 'dail' and not is_flash_v2_installed():
        pytest.skip(
            f'dail implementation of rope requires gpu and flash attention 2.',
        )
    composer_device = get_device(None)

    if composer_device.name == 'gpu':
        torch.use_deterministic_algorithms(False)

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
            **pos_emb_config,
        },
        use_cache=True,
        tie_word_embeddings=tie_word_embeddings,
    )
    mpt = MPTForCausalLM(hf_config)
    mpt = composer_device.module_to_device(mpt)
    mpt.eval()

    with get_precision_context(
        'amp_bf16' if composer_device.name == 'gpu' else 'fp32',
    ):
        no_padding_input_ids = torch.tensor([[11274, 16390, 11]])
        no_padding_input_ids = composer_device.tensor_to_device(
            no_padding_input_ids,
        )
        no_padding_attention_mask = torch.tensor([[1, 1, 1]])
        no_padding_attention_mask = composer_device.tensor_to_device(
            no_padding_attention_mask,
        )

        _ = mpt.generate(
            input_ids=no_padding_input_ids,
            attention_mask=no_padding_attention_mask,
            **generation_kwargs,
        )

    if composer_device.name == 'gpu':
        reproducibility.configure_deterministic_mode()


@pytest.mark.gpu
@pytest.mark.parametrize(
    'pos_emb_config',
    [{
        'alibi': False,
        'rope': False,
    }, {
        'alibi': True,
        'rope': False,
    }, {
        'alibi': False,
        'rope': True,
        'rope_theta': 10000,
        'rope_impl': 'dail',
        'rope_dail_config': {
            'type': 'original',
            'pos_idx_in_fp32': True,
            'xpos_scale_base': 512,
        },
    }, {
        'alibi': False,
        'rope': True,
        'rope_theta': 10000,
        'rope_impl': 'hf',
        'rope_hf_config': {
            'type': 'no_scaling',
            'factor': 1.0,
        },
    }],
)
@pytest.mark.parametrize('tie_word_embeddings', [True, False])
def test_model_to(pos_emb_config: dict, tie_word_embeddings: bool):
    # test that moving the model to diff devices and dtypes in diff ways does not break the model
    if pos_emb_config['rope'] and pos_emb_config[
        'rope_impl'] == 'dail' and not is_flash_v2_installed():
        pytest.skip(f'dail implementation of rope requires flash attention 2.')

    hf_config = MPTConfig(
        init_device='cpu',
        d_model=64,
        n_heads=4,
        n_layers=2,
        expansion_ratio=2,
        max_seq_len=4,
        emb_pdrop=0.1,
        resid_pdrop=0.2,
        attn_config={
            'attn_impl': 'torch',
            **pos_emb_config,
        },
        init_config={
            'name': 'baseline_',
            'init_std': 0.02,
        },
        tie_word_embeddings=tie_word_embeddings,
    )
    mpt = MPTForCausalLM(hf_config)
    mpt = mpt.bfloat16()
    mpt = mpt.to('cuda')
    mpt.eval()

    # gen input data
    input_ids = torch.tensor([[11274, 16390, 11]]).to('cuda')
    attention_mask = torch.tensor([[1, 1, 1]]).bool().to('cuda')

    _ = mpt(input_ids, attention_mask=attention_mask)

    # move the model around using different methods
    mpt = mpt.to('cpu')

    # verify the model still works
    if not (pos_emb_config['rope'] and pos_emb_config['rope_impl'] == 'dail'):
        with torch.autocast('cpu', dtype=torch.bfloat16, enabled=True):
            _ = mpt(
                input_ids.to('cpu'),
                attention_mask=attention_mask.to('cpu'),
            )

    mpt = mpt.float()

    # verify the model still works
    if not (pos_emb_config['rope'] and pos_emb_config['rope_impl'] == 'dail'):
        _ = mpt(input_ids.to('cpu'), attention_mask=attention_mask.to('cpu'))

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
                torch.ones(seq_len)[None, ...],
                n_heads,
                torch.float32,
            )
            alibi_bias_hf = alibi_bias_hf - alibi_bias_hf.max(
                dim=2,
                keepdim=True,
            ).values

            # mosaicml alibi bais
            alibi_bias_m = build_alibi_bias(
                n_heads,
                seq_len,
                dtype=torch.float32,
            )
            alibi_bias_m = alibi_bias_m[0]

            torch.testing.assert_close(alibi_bias_hf, alibi_bias_m)


@pytest.mark.parametrize(
    'attn_impl',
    [
        'torch',
        pytest.param('flash', marks=pytest.mark.gpu),
        pytest.param('torch', marks=pytest.mark.gpu),
    ],
)
@pytest.mark.parametrize(
    'pos_emb_config',
    [{
        'alibi': False,
        'rope': False,
    }, {
        'alibi': True,
        'rope': False,
    }, {
        'alibi': False,
        'rope': True,
        'rope_theta': 10000,
        'rope_impl': 'dail',
        'rope_dail_config': {
            'type': 'original',
            'pos_idx_in_fp32': True,
            'xpos_scale_base': 512,
        },
    }, {
        'alibi': False,
        'rope': True,
        'rope_theta': 10000,
        'rope_impl': 'hf',
        'rope_hf_config': {
            'type': 'no_scaling',
            'factor': 1.0,
        },
    }],
)
def test_forward_with_output_attentions_and_output_hidden_states(
    attn_impl: str,
    pos_emb_config: dict,
):
    if pos_emb_config['alibi'] and not check_alibi_support(attn_impl):
        pytest.skip(f'flash attention below v2.4.2 does not support alibi.')
    if attn_impl == 'flash':
        pytest.skip(f'output_attentions only implemented with torch attention.')
    if pos_emb_config['rope'] and pos_emb_config[
        'rope_impl'] == 'dail' and not is_flash_v2_installed():
        pytest.skip(
            f'dail implementation of rope requires gpu and flash attention 2.',
        )

    composer_device = get_device(None)

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
            **pos_emb_config,
        },
        use_cache=True,
        init_config={
            'name': 'baseline_',
            'init_std': 0.02,
        },
        tie_word_embeddings=True,
    )
    mpt = MPTForCausalLM(hf_config)
    mpt = composer_device.module_to_device(mpt)
    mpt.eval()

    with get_precision_context(
        'amp_bf16' if composer_device.name == 'gpu' else 'fp32',
    ):
        input_ids = torch.tensor([[11274, 16390, 11]])
        input_ids = composer_device.tensor_to_device(input_ids)
        attention_mask = torch.tensor([[1, 1, 1]]).bool()
        attention_mask = composer_device.tensor_to_device(attention_mask)

        outputs = mpt(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True,
        )

        assert len(outputs.attentions) == n_layers
        assert all(attn.shape == (1, 4, 3, 3) for attn in outputs.attentions)
        assert len(outputs.hidden_states) == n_layers + 1


@pytest.mark.gpu
@pytest.mark.parametrize('init_device', ['cpu', 'meta', 'mixed'])
@pytest.mark.parametrize('world_size', [2])
def test_hf_init(
    tmp_path: pathlib.Path,
    init_device: str,
    world_size: int,
    batch_size: int = 1,
):
    if not torch.cuda.device_count() >= world_size:
        pytest.skip(f'This test requires {world_size} GPUs.')

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
        init_device='cpu',
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

    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    CONFIG_MAPPING._extra_content['mpt'] = MPTConfig
    AutoModelForCausalLM.register(MPTConfig, MPTForCausalLM)

    context = contextlib.nullcontext()
    if init_device == 'meta':
        context = init_empty_weights(include_buffers=False)

    # Load in a pretrained model with a given context
    with context:
        model = AutoModelForCausalLM.from_pretrained(
            save_path,
            trust_remote_code=True,
        )

    tokenizer_cfg: Dict[str, Any] = _load_tokenizer_cfg(test_cfg.tokenizer)
    tokenizer = build_tokenizer(
        test_cfg.tokenizer.name,
        tokenizer_cfg.get('kwargs', {}),
    )

    optimizer = DecoupledAdamW(
        model.parameters(),
        lr=1e-5,
        betas=(0.9, 0.99),
    )

    prepare_fsdp_module(
        model,
        optimizer,
        FSDPConfig(**fsdp_config),
        precision,
        device,
        False,
    )

    model = HuggingFaceModelWithFSDP(model, tokenizer)

    batch = gen_random_batch(batch_size, test_cfg)

    original_params = next(model.parameters()).clone().data

    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
        outputs = model(batch)
    loss = model.loss(outputs, batch)
    assert isinstance(loss, torch.Tensor)
    loss.backward()
    optimizer.step()

    updated_params = next(model.parameters()).clone().data

    assert not torch.equal(original_params, updated_params)


@pytest.mark.gpu
def test_head_dim_8_flash_mqa_attn(batch_size: int = 2):
    test_cfg = get_config(conf_path='scripts/train/yamls/pretrain/testing.yaml')
    test_cfg.device = torch.cuda.current_device()

    test_cfg.batch_size = batch_size

    hf_config = MPTConfig(
        init_device='cpu',
        d_model=128,
        n_heads=16,
        n_layers=1,
        expansion_ratio=2,
        max_seq_len=128,
        emb_pdrop=0.1,
        resid_pdrop=0.2,
        attn_config={
            'attn_impl': 'flash',
            'attn_type': 'multiquery_attention',
        },
    )
    test_cfg.device = torch.cuda.current_device()

    tokenizer_cfg: Dict[str, Any] = _load_tokenizer_cfg(test_cfg.tokenizer)
    tokenizer = build_tokenizer(
        test_cfg.tokenizer.name,
        tokenizer_cfg.get('kwargs', {}),
    )

    mpt = MPTForCausalLM(hf_config)

    model = HuggingFaceModelWithFSDP(mpt, tokenizer, shift_labels=True)

    model = model.to(test_cfg.device)
    batch = gen_random_batch(batch_size, test_cfg)

    assert batch['input_ids'].shape == torch.Size([
        batch_size,
        test_cfg.max_seq_len,
    ])

    model.train()

    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        output = model(batch)

    assert not torch.isnan(output.logits).any()


def test_construct_blocks():
    n_layers = 13

    config = MPTConfig(
        d_model=32,
        n_heads=16,
        n_layers=n_layers,
        expansion_ratio=2,
        max_seq_len=64,
        attn_config={
            'attn_impl': 'flash',
            'attn_type': 'grouped_query_attention',
            'kv_n_heads': 4,
        },
    )

    # override architecture taken from https://research.character.ai/optimizing-inference/
    config.block_overrides = {}
    config.block_overrides['overrides'] = {
        'reuse_kv_layer': {
            'attn_config': {
                'reuse_kv_layer_idx': -6,
            },
        },
        'sliding_window_layer': {
            'attn_config': {
                'sliding_window_size': 1024,
            },
        },
        'sliding_window_layer_reuse': {
            'attn_config': {
                'sliding_window_size': 1024,
                'reuse_kv_layer_idx': -1,
            },
        },
    }
    config.block_overrides['order'] = [
        {
            'name': 'default',
        },
        {
            'order': [
                {
                    'name': 'sliding_window_layer',
                },
                {
                    'name': 'sliding_window_layer_reuse',
                },
                {
                    'name': 'sliding_window_layer',
                },
                {
                    'name': 'sliding_window_layer_reuse',
                    'repeat': 2,
                },
                {
                    'name': 'reuse_kv_layer',
                },
            ],
            'repeat': 2,
        },
    ]

    block_list = MPTModel(config).construct_blocks(config)

    assert len(block_list) == n_layers
    assert block_list[0].attn.sliding_window_size == -1
    assert block_list[0].attn.reuse_kv_layer_idx is None

    for layer_offset in [1, 7]:
        assert block_list[layer_offset].attn.sliding_window_size == 1024
        assert block_list[layer_offset].attn.reuse_kv_layer_idx is None
        assert block_list[layer_offset + 1].attn.sliding_window_size == 1024
        assert block_list[layer_offset +
                          1].attn.reuse_kv_layer_idx == layer_offset

        assert block_list[layer_offset + 2].attn.sliding_window_size == 1024
        assert block_list[layer_offset + 2].attn.reuse_kv_layer_idx is None
        assert block_list[layer_offset + 3].attn.sliding_window_size == 1024
        assert block_list[layer_offset +
                          3].attn.reuse_kv_layer_idx == layer_offset + 2
        assert block_list[layer_offset + 4].attn.sliding_window_size == 1024
        assert block_list[layer_offset +
                          4].attn.reuse_kv_layer_idx == layer_offset + 2

        assert block_list[layer_offset + 5].attn.sliding_window_size == -1
        assert block_list[layer_offset + 5].attn.reuse_kv_layer_idx == 0


@pytest.mark.gpu
def test_reuse_prev_layer_kv_cache(
    request: pytest.FixtureRequest,
    batch_size: int = 2,
):
    conf_path = 'scripts/train/yamls/pretrain/testing.yaml'
    model_config_overrides = {
        'block_overrides': {
            'order': [
                {
                    'name': 'default',
                },
                {
                    'name': 'kv_reuse_layer',
                },
            ],
            'overrides': {
                'kv_reuse_layer': {
                    'attn_config': {
                        'reuse_kv_layer_idx': -1,
                    },
                },
            },
        },
        'use_cache': True,
    }
    test_cfg, model, _ = _get_objs(
        request=request,
        conf_path=conf_path,
        model_config_overrides=model_config_overrides,
        attn_impl='flash',
    )

    batch = gen_random_batch(batch_size, test_cfg)

    assert batch['input_ids'].shape == torch.Size([
        batch_size,
        test_cfg.max_seq_len,
    ])
    model.train()

    prev_layer_key_value_dict = {}

    def mock_forward(b_forward, b_idx, *args, **kwargs):  # type: ignore
        if 'prev_layer_key_value' in kwargs:
            prev_layer_key_value_dict[b_idx] = kwargs['prev_layer_key_value']
        return b_forward(*args, **kwargs)

    for b_idx, block in enumerate(model.model.transformer.blocks):
        block.forward = partial(mock_forward, block.forward, b_idx)

    with get_precision_context(test_cfg.precision):
        outputs = model(batch)
        assert len(outputs.past_key_values) == 2
        assert torch.all(
            outputs.past_key_values[0][0] == outputs.past_key_values[1][0],
        )
        assert torch.all(
            outputs.past_key_values[0][1] == outputs.past_key_values[1][1],
        )
        assert 0 not in prev_layer_key_value_dict
        assert torch.all(
            prev_layer_key_value_dict[1][0] == outputs.past_key_values[0][0],
        )
        assert torch.all(
            prev_layer_key_value_dict[1][1] == outputs.past_key_values[0][1],
        )


def test_override_block_args():
    block_args = {'a': 1, 'b': {'c': 3}, 'd': 4}
    override_config = {'a': 2, 'b': {'c': 5}, 'e': 6}
    allowed_block_overrides = {'a': None, 'b': {'c': None}, 'e': None}
    new_config = MPTModel._override_block_args(
        block_args,
        override_config,
        allowed_block_overrides,
    )
    assert new_config['a'] == 2
    assert new_config['d'] == 4
    assert new_config['e'] == 6
    assert new_config['b']['c'] == 5


def test_get_modules_order_expanded():
    order = [
        {
            'name': 'default',
        },
        {
            'name': 'layer_a',
            'repeat': 2,
        },
        {
            'order': [{
                'name': 'layer_b',
            },],
            'repeat': 3,
        },
        {
            'name': 'layer_c',
            'repeat': 2,
        },
        {
            'name': 'default',
        },
    ]
    expected_list = [
        'default',
        'layer_a',
        'layer_a',
        'layer_b',
        'layer_b',
        'layer_b',
        'layer_c',
        'layer_c',
        'default',
    ]
    assert expected_list == MPTModel._get_modules_order_expanded(order)


@pytest.mark.parametrize('reuse_kv_layer_idx', [-2, -1, 0])
def test_resolve_reuse_kv_layer_idx(reuse_kv_layer_idx: int):
    layer_a_override = {
        'key_1': 'value_a',
        'attn_config': {
            'key_2': 'value_b',
        },
    }
    layer_b_override = {
        'key_1': 'value_c',
        'attn_config': {
            'key_2': 'value_d',
        },
    }
    layer_c_override = {
        'key_1': 'value_c' if reuse_kv_layer_idx == -1 else 'value_a',
        'attn_config': {
            'key_2': 'value_d' if reuse_kv_layer_idx == -1 else 'value_b',
            'reuse_kv_layer_idx': reuse_kv_layer_idx,
        },
    }
    block_overrides = {
        'overrides': {
            'layer_a': layer_a_override,
            'layer_b': layer_b_override,
            'layer_c': layer_c_override,
        },
    }
    model_modules_order_expanded = ['layer_a', 'layer_b', 'layer_c']
    if reuse_kv_layer_idx == -1:
        model_modules_order_expanded = [
            'layer_a',
            'layer_b',
            'layer_c',
            'layer_c',
            'layer_c',
            'layer_a',
            'layer_c',
        ]
    reuse_kv_layer_idx_dict = {}

    def _validate_helper(b_idx: int) -> int:
        return MPTModel._resolve_reuse_kv_layer_idx(
            overrides_definition=block_overrides['overrides'],
            model_modules_order_expanded=model_modules_order_expanded,
            b_idx=b_idx,
            override_config=copy.deepcopy(
                block_overrides['overrides'][model_modules_order_expanded[b_idx]
                                            ],
            ),
            reuse_kv_layer_idx_dict=reuse_kv_layer_idx_dict,
        )

    if reuse_kv_layer_idx == -1:
        assert _validate_helper(b_idx=2) == 1
        assert _validate_helper(b_idx=3) == 1
        assert _validate_helper(b_idx=4) == 1
        with pytest.raises(
            expected_exception=ValueError,
            match=
            'For reusing the kv cache of a previous layer, the previous layer should match the block config as the current layer\.',  # type: ignore
        ):
            _validate_helper(b_idx=6)

    elif reuse_kv_layer_idx == -2:
        assert _validate_helper(b_idx=2) == 0
    else:
        with pytest.raises(
            expected_exception=ValueError,
            match=
            'The relative index of kv layer to reuse, override_attn_config\[\"reuse_kv_layer_idx\"\]=0, should be negative\.',  # type: ignore
        ):
            _validate_helper(b_idx=2)


def test_hf_rotary_child_class_builds():
    rope_head_dim = 32
    max_seq_len = 128
    rope_theta = 10000
    HFRotaryEmbeddingMP(rope_head_dim, max_seq_len, rope_theta, device='cpu')
