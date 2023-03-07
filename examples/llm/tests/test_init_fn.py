# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import math
from functools import partial

import pytest
import torch
from composer.utils import reproducibility
from omegaconf import OmegaConf as om
from torch import nn

from examples.llm.src.models.param_init_fns import generic_param_init_fn_


class MLP(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.in_features, cfg.out_features, bias=True)
        self.ln_1 = nn.LayerNorm(cfg.out_features)
        self.fc2 = nn.Linear(cfg.out_features, cfg.out_features, bias=True)
        self.fc2._is_residual = True  # type: ignore

    def forward(self, x):
        y = self.ln_1(self.fc1(x))
        res = y
        y = self.fc2(y)
        y = y + res
        return y


@pytest.mark.parametrize('is_residual', [True, False])
def test_div_is_residual(is_residual: bool):
    reproducibility.seed_all(7)

    in_features, out_features = 8, 32
    cfg = om.create({
        'in_features': in_features,
        'out_features': out_features,
        'n_layers': 2,
    })
    cfg.init_div_is_residual = is_residual
    model = MLP(cfg)

    model.apply(partial(generic_param_init_fn_, cfg=cfg,
                        init_fn_=nn.init.ones_))

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
def test_fused_init_helper(fused):
    reproducibility.seed_all(7)

    in_features, out_features = 8, 32
    cfg = om.create({
        'in_features': in_features,
        'out_features': out_features,
    })

    fc = nn.Linear(cfg.in_features, cfg.out_features, bias=True)
    fc.train()
    if fused:
        fc._fused = (0, (cfg.out_features // 2,))  # type: ignore

    def init_fn_(weight):
        # dummy init based on layer width
        with torch.no_grad():
            out_features, _ = weight.shape[:2]
            weight.fill_(1 / out_features)

    fc.apply(partial(generic_param_init_fn_, cfg=cfg, init_fn_=init_fn_))

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
def test_all_params_init(module):
    fill_val = torch.finfo(torch.float16).max

    def max_fill_init_(weight):
        # init param with max value
        with torch.no_grad():
            weight.fill_(fill_val)

    module.apply(
        partial(generic_param_init_fn_,
                cfg=om.create({}),
                init_fn_=max_fill_init_))
    for n, p in module.named_parameters():
        if n == 'bias':
            assert (p == 0).all()
        elif n == 'weight':
            assert (p == fill_val).all()
