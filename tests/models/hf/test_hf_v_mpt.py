# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
import warnings
from typing import Optional

import pytest
import torch
from composer.utils import reproducibility
from omegaconf import OmegaConf as om

from llmfoundry.utils.builders import build_composer_model, build_tokenizer
from llmfoundry.utils.config_utils import to_dict_recursive


@pytest.mark.gpu
@pytest.mark.parametrize('attn_impl,dropout,alibi,mask_val,no_attn_mask', [
    ('flash', 0.0, False, 1, False),
    ('flash', 0.1, False, 1, False),
    ('torch', 0.0, False, 1, False),
    ('torch', 0.0, False, 0, False),
    ('flash', 0.0, False, None, True),
    ('torch', 0.0, False, None, True),
])
def test_compare_hf_v_mpt(attn_impl: str, dropout: float, alibi: bool,
                          mask_val: Optional[int], no_attn_mask: bool):
    warnings.filterwarnings(
        action='ignore',
        message='Torchmetrics v0.9 introduced a new argument class property')
    warnings.filterwarnings(action='ignore',
                            message='Using Fused Cross Entropy Loss.')

    conf_path = 'scripts/train/yamls/pretrain/mpt-125m.yaml'  # set cfg path
    batch_size = 2  # set batch size
    device = 'cuda'  # set device

    # get hf gpt2 cfg
    hf_cfg = om.create({
        'model': {
            'name': 'hf_causal_lm',
            'pretrained_model_name_or_path': 'gpt2',
            'device': 'cpu',
            'pretrained': False,
            'config_overrides': {
                'n_layer': 2,
                'n_embd': 64,
                'n_head': 8,
            }
        },
        'tokenizer': {
            'name': 'gpt2'
        },
    })

    # get hf gpt2 model
    print(hf_cfg)
    tokenizer_name = hf_cfg.tokenizer['name']
    tokenizer_kwargs = hf_cfg.tokenizer.get('kwargs', {})
    tokenizer = build_tokenizer(
        tokenizer_name=tokenizer_name,
        tokenizer_kwargs=tokenizer_kwargs,
    )
    name = hf_cfg.model.pop('name')
    hf_cfg.model.pop('device')
    hf_model = build_composer_model(
        name=name,
        cfg=to_dict_recursive(hf_cfg.model),
        tokenizer=tokenizer,
    ).to(device)
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
    # regardless of if rng is seeded
    # attn_dropout must be set to 0 for numerical comparisons.
    hf_model.model.config.attn_pdrop = 0.0
    for b in hf_model.model.transformer.h:
        b.attn.attn_dropout.p = 0.0

    # get mosaic 125m config
    with open(conf_path) as f:
        cfg = om.load(f)

    # extract model cfg
    model_cfg = cfg.model
    # use given attn implementation
    model_cfg.attn_impl = attn_impl
    model_cfg.alibi = alibi
    # modify cfg for HF GPT2 compatibility
    model_cfg.max_seq_len = hf_model.model.config.n_ctx
    model_cfg.init_device = device
    model_cfg.vocab_size = hf_model.model.config.vocab_size
    # set dropout prob
    model_cfg.resid_pdrop = hf_model.model.config.resid_pdrop
    model_cfg.emb_pdrop = hf_model.model.config.embd_pdrop
    # attn_dropout is integrated into the FlashMHA kernel
    # given this, it will generate different drop idx when compared to nn.Dropout
    # regardless of if rng is seeded.
    model_cfg.attn_pdrop = hf_model.model.config.attn_pdrop
    model_cfg.n_layers = hf_model.model.config.n_layer
    model_cfg.d_model = hf_model.model.config.n_embd
    model_cfg.n_heads = hf_model.model.config.n_head

    # Build Model
    print('Initializing model...')

    print(model_cfg)
    if 'name' in model_cfg:
        name = model_cfg.pop('name')
    if 'device' in model_cfg:
        model_cfg.pop('device')
    model = build_composer_model(
        name=name,
        cfg=to_dict_recursive(model_cfg),
        tokenizer=tokenizer,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    if alibi:
        assert hf_n_params != n_params
    else:
        assert hf_n_params == n_params

    # generate random input branch
    batch = {}
    batch['input_ids'] = torch.randint(low=0,
                                       high=model_cfg.vocab_size,
                                       size=(batch_size,
                                             model_cfg.max_seq_len)).to(device)
    batch['labels'] = torch.randint(low=0,
                                    high=model_cfg.vocab_size,
                                    size=(batch_size,
                                          model_cfg.max_seq_len)).to(device)
    kpm = None
    if no_attn_mask:
        if 'attention_mask' in batch.keys():
            _ = batch.pop('attention_mask')
    else:
        batch['attention_mask'] = torch.ones(size=(batch_size,
                                                   model_cfg.max_seq_len),
                                             dtype=torch.int64).to(device)
        # mask out some tokens
        assert mask_val is not None
        batch['attention_mask'][:, model_cfg.max_seq_len // 2:] = mask_val
        kpm = batch['attention_mask'].view(*batch['attention_mask'].shape, 1)

    hf_model.train()
    model.train()

    # UTIL: can be used to verify that models are not the same at init
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        hf_model_fwd = hf_model(batch)['logits']
        if kpm is not None:
            hf_model_fwd *= kpm
        model_fwd = model(batch).logits
        if kpm is not None:
            model_fwd *= kpm
    print(f'{hf_model_fwd.mean().item() = }\n{model_fwd.mean().item() = }')
    if hf_model_fwd.mean().allclose(model_fwd.mean()):
        warn_msg = f'WARNING: model_fwd ({model_fwd}) and hf_model_fwd ({hf_model_fwd}) are very close at init.'
        raise RuntimeError(warn_msg)

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
        '.mlp.c_fc.': '.ffn.up_proj.',
        '.mlp.c_proj.': '.ffn.down_proj.',
        '.attn.c_attn.': '.attn.Wqkv.',
        '.attn.c_proj.': '.attn.out_proj.',
        '.ln_': '.norm_',
    }

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
    model.load_state_dict(_hf_model_statedict)

    with torch.autocast(device_type=device, dtype=torch.float16):
        reproducibility.seed_all(17)
        hf_model_fwd = hf_model(batch)['logits']
        if kpm is not None:
            hf_model_fwd *= kpm
        reproducibility.seed_all(17)
        model_fwd = model(batch).logits
        if kpm is not None:
            model_fwd *= kpm

    print(f'{hf_model_fwd.mean().item() = }\n{model_fwd.mean().item() = }')
    print(f'{hf_model_fwd = }\n{model_fwd = }')

    # given dropout seeded the same way, the mean of the outputs is extremely similar
    assert hf_model_fwd.mean().allclose(model_fwd.mean(),
                                        rtol=1e-04,
                                        atol=1e-06)
    assert hf_model_fwd.allclose(model_fwd, rtol=1e-02, atol=1e-02)
