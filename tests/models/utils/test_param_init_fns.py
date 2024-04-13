# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
import math
from collections import OrderedDict
from collections.abc import Sequence
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import pytest
import torch
from omegaconf import DictConfig, ListConfig
from omegaconf import OmegaConf as om
from torch import nn

from llmfoundry.layers_registry import param_init_fns
from llmfoundry.models.utils import generic_param_init_fn_


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


@pytest.mark.parametrize('module', [
    nn.Linear(8, 16),
    nn.Embedding(8, 16),
    pytest.param(nn.LayerNorm(8),
                 marks=pytest.mark.xfail(
                     reason='LayerNorm is skipped by init_fn_', strict=True)),
    pytest.param(nn.Conv2d(8, 16, 3),
                 marks=pytest.mark.xfail(
                     reason='generic_param_init_fn_ does not init Conv layers',
                     strict=True)),
])
def test_all_params_init(module: torch.nn.Module):
    fill_val = torch.finfo(torch.float16).max

    def max_fill_init_(weight: torch.Tensor):
        # init param with max value
        with torch.no_grad():
            weight.fill_(fill_val)

    cfg = om.create({
        'n_layers': 2,
    })
    module.apply(partial(generic_param_init_fn_, init_fn_=max_fill_init_,
                         **cfg))
    for n, p in module.named_parameters():
        if n == 'bias':
            assert (p == 0).all()
        elif n == 'weight':
            assert (p == fill_val).all()


@pytest.mark.parametrize('emb_init_cfg', [
    None, ('emb_init_std', 5), ('emb_init_std', 0), ('emb_init_uniform_lim', 2),
    ('emb_init_uniform_lim', [-1, 4]), ('emb_init_uniform_lim', 0),
    ('emb_init_uniform_lim', [1, 1])
])
def test_emb_init(emb_init_cfg: Optional[Tuple[str, Union[int, List[int]]]]):
    cfg: Dict[str, Union[int, List[int]]] = {
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
            ('fc1',
             nn.Linear(dict_cfg.in_features, dict_cfg.out_features, bias=True)),
            ('ln1', nn.LayerNorm(dict_cfg.out_features)),
            ('act1', nn.ReLU()),
            ('fc2',
             nn.Linear(dict_cfg.out_features, dict_cfg.out_features,
                       bias=True)),
        ]))

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
            if len(emb_init_uniform_lim
                  ) == 2 and emb_init_uniform_lim[0] == emb_init_uniform_lim[1]:
                assert (model.emb.weight == emb_init_uniform_lim[0]).all()
