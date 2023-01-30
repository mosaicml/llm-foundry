# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import warnings

import pytest
import torch
from composer.utils import reproducibility
from omegaconf import OmegaConf as om

from examples.llm.src.model_registry import COMPOSER_MODEL_REGISTRY


@pytest.mark.gpu
@pytest.mark.parametrize(
    'attn_impl,dropout,strict,alibi,mask_val',
    [
        ('flash', 0.0, True, False, 1),
        ('flash', 0.1, True, False, 1),
        ('torch', 0.0, False, False,
         1),  # requires strict=False to skip loading model.attn_mask
        ('triton', 0.0, False, False,
         1),  # requires strict=False to skip loading model.attn_mask
        ('triton', 0.1, False, False,
         1),  # requires strict=False to skip loading model.attn_mask
        pytest.param('torch',
                     0.0,
                     False,
                     True,
                     1,
                     marks=pytest.mark.xfail(
                         reason='hf model is not implemented with alibi')),
        pytest.param('triton',
                     0.1,
                     False,
                     True,
                     1,
                     marks=pytest.mark.xfail(
                         reason='hf model is not implemented with alibi')),
        ('torch', 0.0, False, False, 0
        ),  # requires strict=False to skip loading model.attn_mask, testing case where key_pad_mask is 0
        ('triton', 0.0, False, False, 0
        ),  # requires strict=False to skip loading model.attn_mask, testing case where key_pad_mask is 0
        ('triton', 0.1, False, False, 0
        ),  # requires strict=False to skip loading model.attn_mask, testing case where key_pad_mask is 0
    ])
def test_compare_hf_v_mosaic_gpt(attn_impl, dropout, strict, alibi, mask_val):
    warnings.filterwarnings(
        action='ignore',
        message='Torchmetrics v0.9 introduced a new argument class property')
    conf_path = 'yamls/mosaic_gpt/125m.yaml'  # set cfg path
    batch_size = 2  # set batch size
    device = 'cuda'  # set decive

    # ensure reproducibility
    seed = 17
    reproducibility.seed_all(seed)  # set seed

    # get hf gpt2 cfg
    hf_cfg = om.create({
        'name': 'hf_causal_lm',
        'hf_config_name_or_path': 'gpt2'
    })

    # get hf gpt2 model
    print(hf_cfg)
    hf_model = COMPOSER_MODEL_REGISTRY[hf_cfg.name](hf_cfg).to(device)
    hf_n_params = sum(p.numel() for p in hf_model.parameters())

    hf_model.model.config.embd_pdrop = dropout
    hf_model.model.transformer.drop.p = dropout

    hf_model.model.config.resid_pdrop = dropout
    for b in hf_model.model.transformer.h:
        b.mlp.dropout.p = dropout
    for b in hf_model.model.transformer.h:
        b.attn.resid_dropout.p = dropout

    # in mosaic gpt, attn_dropout is integrated into the FlashMHA kernel
    # and will therefore generate different drop idx when compared to nn.Dropout
    # reguradless of if rng is seeded
    # attn_dropout must be set to 0 for numerical comparisons.
    hf_model.model.config.attn_pdrop = 0.0
    for b in hf_model.model.transformer.h:
        b.attn.attn_dropout.p = 0.0

    # get mosaic 125m config
    with open(conf_path) as f:
        cfg = om.load(f)

    # extract model cfg
    cfg = cfg.model
    # use triton attn implementation
    cfg.attn_impl = attn_impl
    cfg.alibi = alibi
    # modify cfg for HF GPT2 compatibility
    cfg.max_seq_len = hf_model.model.config.n_ctx
    cfg.device = device
    # set dropout prob
    cfg.resid_pdrop = hf_model.model.config.resid_pdrop
    cfg.emb_pdrop = hf_model.model.config.embd_pdrop
    # attn_dropout is integrated into the FlashMHA kernel
    # given this, it will generate different drop idx when compared to nn.Dropout
    # reguradless of if rng is seeded.
    cfg.attn_pdrop = hf_model.model.config.attn_pdrop

    # Build Model
    print('Initializing model...')

    print(cfg)
    model = COMPOSER_MODEL_REGISTRY[cfg.name](cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    if alibi:
        assert hf_n_params != n_params
    else:
        assert hf_n_params == n_params

    # generate random input branch
    batch = {}
    batch['input_ids'] = torch.randint(low=0,
                                       high=cfg.vocab_size,
                                       size=(batch_size,
                                             cfg.max_seq_len)).to(device)
    batch['labels'] = torch.randint(low=0,
                                    high=cfg.vocab_size,
                                    size=(batch_size,
                                          cfg.max_seq_len)).to(device)
    batch['attention_mask'] = torch.ones(size=(batch_size, cfg.max_seq_len),
                                         dtype=torch.int64).to(device)

    batch['attention_mask'][:, cfg.max_seq_len // 2:] = mask_val

    hf_model.train()
    model.train()

    # UTIL: can be used to verify that models are not the same at init
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        torch.manual_seed(0)
        hf_model_fwd = hf_model(batch)
        torch.manual_seed(0)
        model_fwd = model(batch)
    print(f'{hf_model_fwd.mean().item() = }\n{model_fwd.mean().item() = }')
    if hf_model_fwd.mean().allclose(model_fwd.mean()):
        warn_msg = f'WARNING: model_fwd ({model_fwd}) and hf_model_fwd ({hf_model_fwd}) are very close at init.'
        raise warnings.warn(warn_msg)  # type: ignore

    hf_model_statedict = hf_model.state_dict()

    # convert hf gpt statedict to mosaic gpt statedict
    # HF keys which are ignored
    hf_keys_ignore = ['.attn.masked_bias', '.attn.bias', 'lm_head']
    # HF params which need to be transposed
    _transpose = [
        '.attn.c_attn.', '.attn.c_proj.', '.mlp.c_fc.', '.mlp.c_proj.'
    ]
    # HF keys which need to be replaced by the associated value
    hf_2_mosaic_key_mods = {
        'model.transformer.h.': 'model.transformer.blocks.',
        '.mlp.c_fc.': '.mlp.mlp_up.',
        '.mlp.c_proj.': '.mlp.mlp_down.',
    }
    if attn_impl == 'torch':
        hf_2_mosaic_key_mods[
            '.attn.c_attn.weight'] = '.causal_attn.mhsa.in_proj_weight'
        hf_2_mosaic_key_mods[
            '.attn.c_attn.bias'] = '.causal_attn.mhsa.in_proj_bias'
        hf_2_mosaic_key_mods['.attn.c_proj.'] = '.causal_attn.mhsa.out_proj.'
    else:
        hf_2_mosaic_key_mods['.attn.c_attn.'] = '.causal_attn.mhsa.Wqkv.'
        hf_2_mosaic_key_mods['.attn.c_proj.'] = '.causal_attn.mhsa.out_proj.'

    # convert hf gpt statedict to mosaic gpt statedict using the dict and list above
    _hf_model_statedict = {}
    for k, v in hf_model_statedict.items():
        skip = False
        for _k in hf_keys_ignore:
            if _k in k:
                skip = True
                continue
        for _k in _transpose:
            if _k in k:
                v = v.t()

        for _k, _v in hf_2_mosaic_key_mods.items():
            if _k in k:
                k = k.replace(_k, _v)
        if not skip:
            _hf_model_statedict[k] = v

    # load hf model weights into mosaic gpt model
    model.load_state_dict(_hf_model_statedict, strict=strict)

    with torch.autocast(device_type=device, dtype=torch.float16):
        torch.manual_seed(seed)
        hf_model_fwd = hf_model(batch)
        torch.manual_seed(seed)
        model_fwd = model(batch)

    print(f'{hf_model_fwd.mean().item() = }\n{model_fwd.mean().item() = }')
    print(f'{hf_model_fwd = }\n{model_fwd = }')

    # given dropout seeded the same way, the mean of the outputs is extremely similar
    assert hf_model_fwd.mean().allclose(model_fwd.mean())
    assert hf_model_fwd.allclose(model_fwd, rtol=1e-02, atol=1e-02)
