# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
import math
from collections import OrderedDict
from collections.abc import Sequence
from functools import partial
from typing import Optional, Union

import pytest
import torch
from omegaconf import DictConfig, ListConfig
from omegaconf import OmegaConf as om
from torch import nn
from torch.distributed._tensor import (
    DeviceMesh,
    DTensor,
    Shard,
    distribute_tensor,
)

from llmfoundry.layers_registry import param_init_fns
from llmfoundry.models.utils import generic_param_init_fn_
from llmfoundry.models.utils.param_init_fns import fc_init, fused_param_init_helper, stacked_param_init_helper, embedding_init, multihead_attention_init


class MLP(nn.Module):

    def __init__(self, cfg: Union[ListConfig, DictConfig]):
        super().__init__()
        self.fc1 = nn.Linear(cfg.in_features, cfg.out_features, bias=True)
        self.ln_1 = nn.LayerNorm(cfg.out_features)
        self.fc2 = nn.Linear(cfg.out_features, cfg.out_features, bias=True)
        self.fc2._is_residual = True

    def forward(self, x: torch.Tensor):
        y = self.ln_1(self.fc1(x))
        res = y
        y = self.fc2(y)
        y = y + res
        return y


@pytest.mark.parametrize('is_residual', [True, False])
def test_div_is_residual(is_residual: bool):
    in_features, out_features = 8, 32
    cfg = om.create({
        'in_features': in_features,
        'out_features': out_features,
        'n_layers': 2,
    })
    cfg.init_div_is_residual = is_residual
    model = MLP(cfg)

    model.apply(partial(generic_param_init_fn_, init_fn_=nn.init.ones_, **cfg))

    # verify layer norm is init to bias=0 and weight=1
    assert (model.ln_1.weight == 1).all()
    if model.ln_1.bias is not None:
        assert (model.ln_1.bias == 0).all()

    # verify _is_residual works
    expected_value = 1 / math.sqrt(2 * cfg.n_layers) if is_residual else 1
    for n, p in model.named_parameters():
        if n == 'bias':
            assert (p == 0).all()
        elif n == 'weight':
            assert (p == expected_value).all()


@pytest.mark.parametrize('fused', [True, False])
def test_fused_init_helper(fused: bool):
    in_features, out_features = 8, 32
    cfg = om.create({
        'in_features': in_features,
        'out_features': out_features,
        'n_layers': 2,
    })

    fc = nn.Linear(cfg.in_features, cfg.out_features, bias=True)
    fc.train()
    if fused:
        fc._fused = (0, (cfg.out_features // 2,))

    def init_fn_(weight: torch.Tensor):
        # dummy init based on layer width
        with torch.no_grad():
            out_features, _ = weight.shape[:2]
            weight.fill_(1 / out_features)

    fc.apply(partial(generic_param_init_fn_, init_fn_=init_fn_, **cfg))

    expected_value = 1 / cfg.out_features
    if fused:
        expected_value *= 2
    for n, p in fc.named_parameters():
        if n == 'bias':
            assert (p == 0).all()
        elif n == 'weight':
            assert (p == expected_value).all()


@pytest.mark.parametrize(
    'module',
    [
        nn.Linear(8, 16),
        nn.Embedding(8, 16),
        pytest.param(
            nn.LayerNorm(8),
            marks=pytest.mark.xfail(
                reason='LayerNorm is skipped by init_fn_',
                strict=True,
            ),
        ),
        pytest.param(
            nn.Conv2d(8, 16, 3),
            marks=pytest.mark.xfail(
                reason='generic_param_init_fn_ does not init Conv layers',
                strict=True,
            ),
        ),
    ],
)
def test_all_params_init(module: torch.nn.Module):
    fill_val = torch.finfo(torch.float16).max

    def max_fill_init_(weight: torch.Tensor):
        # init param with max value
        with torch.no_grad():
            weight.fill_(fill_val)

    cfg = om.create({
        'n_layers': 2,
    })
    module.apply(
        partial(generic_param_init_fn_, init_fn_=max_fill_init_, **cfg),
    )
    for n, p in module.named_parameters():
        if n == 'bias':
            assert (p == 0).all()
        elif n == 'weight':
            assert (p == fill_val).all()


@pytest.mark.parametrize(
    'emb_init_cfg',
    [
        None,
        ('emb_init_std', 5),
        ('emb_init_std', 0),
        ('emb_init_uniform_lim', 2),
        ('emb_init_uniform_lim', [-1, 4]),
        ('emb_init_uniform_lim', 0),
        ('emb_init_uniform_lim', [1, 1]),
    ],
)
def test_emb_init(emb_init_cfg: Optional[tuple[str, Union[int, list[int]]]]):
    cfg: dict[str, Union[int, list[int]]] = {
        'vocab_size': 64,
        'in_features': 16,
        'out_features': 32,
        'n_layers': 2,
    }
    if emb_init_cfg is not None:
        cfg[emb_init_cfg[0]] = emb_init_cfg[1]
    dict_cfg = om.create(cfg)

    model = nn.Sequential(
        OrderedDict([
            ('emb', nn.Embedding(dict_cfg.vocab_size, dict_cfg.in_features)),
            (
                'fc1',
                nn.Linear(
                    dict_cfg.in_features,
                    dict_cfg.out_features,
                    bias=True,
                ),
            ),
            ('ln1', nn.LayerNorm(dict_cfg.out_features)),
            ('act1', nn.ReLU()),
            (
                'fc2',
                nn.Linear(
                    dict_cfg.out_features,
                    dict_cfg.out_features,
                    bias=True,
                ),
            ),
        ]),
    )

    model.apply(partial(param_init_fns.get('kaiming_normal_'), **dict_cfg))

    assert isinstance(model.emb, torch.nn.Embedding)

    if dict_cfg.get('emb_init_std') is not None:
        emb_init_std = dict_cfg.get('emb_init_std')
        if emb_init_std == 0:
            assert (model.emb.weight == 0).all()
    elif dict_cfg.get('emb_init_uniform_lim') is not None:
        emb_init_uniform_lim = dict_cfg.get('emb_init_uniform_lim')
        if emb_init_uniform_lim == 0:
            assert (model.emb.weight == 0).all()
        elif isinstance(emb_init_uniform_lim, Sequence):
            assert len(emb_init_uniform_lim) <= 2
            if len(
                emb_init_uniform_lim,
            ) == 2 and emb_init_uniform_lim[0] == emb_init_uniform_lim[1]:
                assert (model.emb.weight == emb_init_uniform_lim[0]).all()


@pytest.mark.parametrize(
    'padding_idx',
    [0, 2],
)
def test_emb_padding_init(
    padding_idx: int,
):
    cfg: dict[str, Union[int, list[int]]] = {
        'vocab_size': 64,
        'in_features': 16,
        'n_layers': 2,
        'padding_idx': padding_idx,
        'emb_init_std': 5,
    }
    dict_cfg = om.create(cfg)

    model = nn.Embedding(
        dict_cfg.vocab_size,
        dict_cfg.in_features,
        dict_cfg.padding_idx,
    )

    model.apply(partial(param_init_fns.get('kaiming_normal_'), **dict_cfg))
    assert isinstance(model, torch.nn.Embedding)

    if dict_cfg.get('emb_init_std') is not None:
        assert (model.weight[padding_idx] == 0).all()




def init_arange_(weight: torch.Tensor) -> None:
    with torch.no_grad():
        weight.copy_(
            torch.arange(weight.numel()).reshape(weight.shape).float(),
        )


@pytest.mark.world_size(2)
def test_fused_init_helper_dtensor_vs_tensor():
    #Test that fused_param_init_helper produces the same results for a regular
    # tensor and a DTensor.

    mesh = DeviceMesh('cpu', [0, 1])

    regular_tensor = torch.nn.Parameter(torch.zeros(4, 4))

    dtensor = torch.nn.Parameter(
        distribute_tensor(regular_tensor, mesh, [Shard(0)]),
    )

    # Define fused parameters (dimension 0, split at index 2)
    fused_params = (0, [2])

    # Initialize both tensors using fused_param_init_helper
    fused_param_init_helper(regular_tensor, init_arange_, fused_params)
    fused_param_init_helper(dtensor, init_arange_, fused_params)

    assert isinstance(
        dtensor,
        torch.nn.Parameter,
    ), f'param is not an nn.Parameter anymore: {type(dtensor)}'
    assert isinstance(
        dtensor,
        DTensor,
    ), f'DTensor is not a DTensor: {type(dtensor)}'

    # For comparison, convert DTensor to regular tensor
    dtensor_result = dtensor.full_tensor()

    # Verify results are identical
    assert torch.equal(regular_tensor, dtensor_result), \
        f'fused_param_init_helper produced different results for regular tensor: {regular_tensor} vs DTensor: {dtensor_result}'

    # Check that each partition was separately initialized as expected
    numel_half = dtensor_result.numel() // 2
    expected_result = torch.cat([
        torch.arange(numel_half),
        torch.arange(numel_half),
    ]).reshape(dtensor_result.shape).float()
    assert torch.equal(
        regular_tensor,
        expected_result,
    ), f'Regular tensor was not initialized correctly: {regular_tensor} vs {expected_result}'
    assert torch.equal(
        dtensor_result,
        expected_result,
    ), f'DTensor was not initialized correctly: {dtensor_result} vs {expected_result}'


@pytest.mark.world_size(2)
@pytest.mark.parametrize('is_fused', [False, True])
def test_fc_init_dtensor_vs_tensor(is_fused: bool):
    """Test that fc_init initializes the same way for a regular linear layer
    and a linear layer with DTensor parameters."""
    # Create a simple device mesh for CPU
    mesh = DeviceMesh('cpu', [0, 1])
    
    # Set up minimal config 
    init_div_is_residual = True
    div_is_residual = 2.0
    
    # Create a regular linear layer
    regular_linear = nn.Linear(8, 4)
    
    # Create a linear layer with DTensor parameters
    dtensor_linear = nn.Linear(8, 4)
    
    # Convert the weight and bias to DTensor
    dtensor_linear.weight = torch.nn.Parameter(
        distribute_tensor(dtensor_linear.weight, mesh, [Shard(0)]),
    )
    dtensor_linear.bias = torch.nn.Parameter(
        distribute_tensor(dtensor_linear.bias, mesh, [Shard(0)]),
    )
    
    # Mark one of the layers as residual to test the residual path
    regular_linear._is_residual = True
    dtensor_linear._is_residual = True
    
    # For fused case, add the _fused attribute
    if is_fused:
        # Define fused parameters (dimension 0, split at index 2)
        fused_params = (0, [2])
        regular_linear._fused = fused_params
        dtensor_linear._fused = fused_params
    
    # Initialize both modules using fc_init
    fc_init(regular_linear, init_arange_, init_div_is_residual, div_is_residual)
    fc_init(dtensor_linear, init_arange_, init_div_is_residual, div_is_residual)
    
    # For comparison, convert DTensor to regular tensor
    dtensor_weight = dtensor_linear.weight.full_tensor()
    dtensor_bias = dtensor_linear.bias.full_tensor()
    
    # Verify weight results are identical
    assert torch.equal(regular_linear.weight, dtensor_weight), \
        f'regular tensor: {regular_linear.weight} vs DTensor: {dtensor_weight}'
    
    # Verify bias results are identical (should be all zeros)
    assert torch.equal(regular_linear.bias, dtensor_bias), \
        f'regular tensor: {regular_linear.bias} vs DTensor: {dtensor_bias}'
    
    if is_fused:
        # For fused case, check that each partition was separately initialized as expected
        # Following the same verification logic as in test_fused_init_helper_dtensor_vs_tensor
        numel_half = regular_linear.weight.numel() // 2
        expected_weight = torch.cat([
            torch.arange(numel_half),
            torch.arange(numel_half),
        ]).reshape(regular_linear.weight.shape).float()
        
        # Apply div_is_residual factor for residual path
        expected_weight = expected_weight / div_is_residual
    else:
        # For non-fused case, the initialization is simpler
        expected_weight = torch.arange(regular_linear.weight.numel()).reshape(regular_linear.weight.shape).float() / div_is_residual
    assert torch.equal(
        dtensor_weight,
        expected_weight,
    ), f'DTensor weight was not initialized correctly: {dtensor_weight} vs {expected_weight}'
    
    # Bias should be all zeros in both cases
    assert torch.all(dtensor_bias == 0), \
        f'DTensor bias was not initialized correctly: {dtensor_bias}'


@pytest.mark.world_size(2)
def test_stacked_param_init_helper_dtensor_vs_tensor():
    """Test that stacked_param_init_helper produces the same results for a regular
    tensor and a DTensor."""
    # Create a simple device mesh for CPU
    mesh = DeviceMesh('cpu', [0, 1])

    regular_tensor = torch.nn.Parameter(torch.zeros(4, 4))

    dtensor = torch.nn.Parameter(
        distribute_tensor(regular_tensor, mesh, [Shard(0)]),
    )

    # Choose a stack dimension
    stack_dim = 0

    # Initialize both tensors using stacked_param_init_helper
    stacked_param_init_helper(regular_tensor, init_arange_, stack_dim)
    stacked_param_init_helper(dtensor, init_arange_, stack_dim)

    assert isinstance(
        dtensor,
        torch.nn.Parameter,
    ), f'param is not an nn.Parameter anymore: {type(dtensor)}'
    assert isinstance(
        dtensor,
        DTensor,
    ), f'DTensor is not a DTensor: {type(dtensor)}'

    # For comparison, convert DTensor to regular tensor
    dtensor_result = dtensor.full_tensor()

    # Verify results are identical
    assert torch.equal(regular_tensor, dtensor_result), \
        f"stacked_param_init_helper produced different results for regular tensor: {regular_tensor} vs DTensor: {dtensor_result}"

    # Check that each slice along stack_dim was separately initialized as expected
    rows, cols = regular_tensor.shape
    for idx in range(regular_tensor.size(stack_dim)):
        if stack_dim == 0:
            expected_slice = torch.arange(cols).float()
            actual_slice = regular_tensor[idx]
        else:
            expected_slice = torch.arange(rows).float()
            actual_slice = regular_tensor[:, idx]
        
        assert torch.equal(actual_slice, expected_slice), \
            f'Slice {idx} was not initialized correctly: {actual_slice} vs {expected_slice}'


@pytest.mark.world_size(2)
@pytest.mark.parametrize('use_padding_idx', [False, True])
@pytest.mark.parametrize('init_type', ['std', 'uniform', 'default'])
def test_embedding_init_dtensor_vs_tensor(use_padding_idx: bool, init_type: str):
    """Test that embedding_init initializes the same way for a regular embedding layer
    and an embedding layer with DTensor parameters."""
    # Create a simple device mesh for CPU
    mesh = DeviceMesh('cpu', [0, 1])
    
    # Setup parameters
    vocab_size, embed_dim = 10, 4
    padding_idx = 2 if use_padding_idx else None
    
    # Create a regular embedding layer
    regular_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
    
    # Create an embedding layer with DTensor parameters
    dtensor_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
    
    # Convert the weight to DTensor
    dtensor_embedding.weight = torch.nn.Parameter(
        distribute_tensor(dtensor_embedding.weight, mesh, [Shard(0)]),
    )
    
    # Initialize both modules using embedding_init
    embedding_init(regular_embedding, init_arange_, None, None)
    embedding_init(dtensor_embedding, init_arange_, None, None)
    
    # For comparison, convert DTensor to regular tensor
    dtensor_weight = dtensor_embedding.weight.full_tensor()
    
    # Verify weight results are identical
    assert torch.equal(regular_embedding.weight, dtensor_weight), \
        f'regular tensor: {regular_embedding.weight} vs DTensor: {dtensor_weight}'
    
    # Additional test for padding_idx
    if padding_idx is not None:
        assert torch.all(regular_embedding.weight[padding_idx] == 0), \
            f'Regular embedding padding not initialized to zero: {regular_embedding.weight[padding_idx]}'
        assert torch.all(dtensor_weight[padding_idx] == 0), \
            f'DTensor embedding padding not initialized to zero: {dtensor_weight[padding_idx]}'


@pytest.mark.world_size(2)
@pytest.mark.parametrize('qkv_same_dim', [True, False])
@pytest.mark.parametrize('is_residual', [True, False])
def test_multihead_attention_init_dtensor_vs_tensor(qkv_same_dim: bool, is_residual: bool):
    """Test that multihead_attention_init initializes the same way for a regular 
    MultiheadAttention and one with DTensor parameters."""
    # Create a simple device mesh for CPU
    mesh = DeviceMesh('cpu', [0, 1])
    
    # Setup parameters
    d_model = 8
    nhead = 2
    
    # Set up minimal config 
    init_div_is_residual = True
    div_is_residual = 2.0
    
    # Create regular MultiheadAttention
    regular_mha = nn.MultiheadAttention(
        embed_dim=d_model,
        num_heads=nhead,
        kdim=d_model if qkv_same_dim else d_model*2,
        vdim=d_model if qkv_same_dim else d_model*2,
        batch_first=True,
    )
    
    # Create MultiheadAttention with DTensor parameters
    dtensor_mha = nn.MultiheadAttention(
        embed_dim=d_model,
        num_heads=nhead,
        kdim=d_model if qkv_same_dim else d_model*2,
        vdim=d_model if qkv_same_dim else d_model*2,
        batch_first=True,
    )
    
    # Mark out_proj as residual if needed
    if is_residual:
        regular_mha.out_proj._is_residual = True
        dtensor_mha.out_proj._is_residual = True
    
    # Convert parameters to DTensor
    if qkv_same_dim:
        # In case of same dimensions, in_proj_weight is used
        dtensor_mha.in_proj_weight = torch.nn.Parameter(
            distribute_tensor(dtensor_mha.in_proj_weight, mesh, [Shard(0)]),
        )
        dtensor_mha.in_proj_bias = torch.nn.Parameter(
            distribute_tensor(dtensor_mha.in_proj_bias, mesh, [Shard(0)]),
        )
    else:
        # In case of different dimensions, q/k/v_proj_weight are used
        dtensor_mha.q_proj_weight = torch.nn.Parameter(
            distribute_tensor(dtensor_mha.q_proj_weight, mesh, [Shard(0)]),
        )
        dtensor_mha.k_proj_weight = torch.nn.Parameter(
            distribute_tensor(dtensor_mha.k_proj_weight, mesh, [Shard(0)]),
        )
        dtensor_mha.v_proj_weight = torch.nn.Parameter(
            distribute_tensor(dtensor_mha.v_proj_weight, mesh, [Shard(0)]),
        )
        dtensor_mha.in_proj_bias = torch.nn.Parameter(
            distribute_tensor(dtensor_mha.in_proj_bias, mesh, [Shard(0)]),
        )
    
    # Convert out_proj parameters to DTensor
    dtensor_mha.out_proj.weight = torch.nn.Parameter(
        distribute_tensor(dtensor_mha.out_proj.weight, mesh, [Shard(0)]),
    )
    dtensor_mha.out_proj.bias = torch.nn.Parameter(
        distribute_tensor(dtensor_mha.out_proj.bias, mesh, [Shard(0)]),
    )
    
    # Initialize both modules using multihead_attention_init
    multihead_attention_init(regular_mha, init_arange_, d_model, init_div_is_residual, div_is_residual)
    multihead_attention_init(dtensor_mha, init_arange_, d_model, init_div_is_residual, div_is_residual)
    
    # Convert DTensor parameters to regular tensors for comparison
    if qkv_same_dim:
        # Compare in_proj_weight
        dtensor_in_proj_weight = dtensor_mha.in_proj_weight.full_tensor()
        assert torch.equal(regular_mha.in_proj_weight, dtensor_in_proj_weight), \
            f'regular in_proj_weight: {regular_mha.in_proj_weight} vs DTensor: {dtensor_in_proj_weight}'
    else:
        # Compare q/k/v_proj_weight
        dtensor_q_proj_weight = dtensor_mha.q_proj_weight.full_tensor()
        dtensor_k_proj_weight = dtensor_mha.k_proj_weight.full_tensor()
        dtensor_v_proj_weight = dtensor_mha.v_proj_weight.full_tensor()
        
        assert torch.equal(regular_mha.q_proj_weight, dtensor_q_proj_weight), \
            f'regular q_proj_weight: {regular_mha.q_proj_weight} vs DTensor: {dtensor_q_proj_weight}'
        assert torch.equal(regular_mha.k_proj_weight, dtensor_k_proj_weight), \
            f'regular k_proj_weight: {regular_mha.k_proj_weight} vs DTensor: {dtensor_k_proj_weight}'
        assert torch.equal(regular_mha.v_proj_weight, dtensor_v_proj_weight), \
            f'regular v_proj_weight: {regular_mha.v_proj_weight} vs DTensor: {dtensor_v_proj_weight}'
    
    # Compare in_proj_bias
    dtensor_in_proj_bias = dtensor_mha.in_proj_bias.full_tensor()
    assert torch.equal(regular_mha.in_proj_bias, dtensor_in_proj_bias), \
        f'regular in_proj_bias: {regular_mha.in_proj_bias} vs DTensor: {dtensor_in_proj_bias}'
    
    # Compare out_proj parameters
    dtensor_out_proj_weight = dtensor_mha.out_proj.weight.full_tensor()
    dtensor_out_proj_bias = dtensor_mha.out_proj.bias.full_tensor()
    
    assert torch.equal(regular_mha.out_proj.weight, dtensor_out_proj_weight), \
        f'regular out_proj.weight: {regular_mha.out_proj.weight} vs DTensor: {dtensor_out_proj_weight}'
    assert torch.equal(regular_mha.out_proj.bias, dtensor_out_proj_bias), \
        f'regular out_proj.bias: {regular_mha.out_proj.bias} vs DTensor: {dtensor_out_proj_bias}'
    
    # Verify biases are zeros
    assert torch.all(dtensor_in_proj_bias == 0), \
        f'DTensor in_proj_bias not initialized to zero: {dtensor_in_proj_bias}'
    assert torch.all(dtensor_out_proj_bias == 0), \
        f'DTensor out_proj.bias not initialized to zero: {dtensor_out_proj_bias}'
    
    # For residual case, verify scaling was applied correctly
    if is_residual:
        # The out_proj weight should be scaled by div_is_residual
        expected_out_proj_weight = torch.arange(regular_mha.out_proj.weight.numel()).reshape(
            regular_mha.out_proj.weight.shape).float() / div_is_residual
        
        assert torch.equal(dtensor_out_proj_weight, expected_out_proj_weight), \
            f'DTensor out_proj.weight was not scaled correctly: {dtensor_out_proj_weight} vs {expected_out_proj_weight}'
