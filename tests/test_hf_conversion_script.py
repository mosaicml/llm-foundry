# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib
import sys

from composer import Trainer

from llmfoundry.models.mpt.modeling_mpt import ComposerMPTCausalLM

# Add repo root to path so we can import scripts and test it
repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(repo_dir)
import shutil
from argparse import Namespace
from typing import cast

import pytest
import torch
import transformers
from omegaconf import DictConfig
from omegaconf import OmegaConf as om

from llmfoundry import COMPOSER_MODEL_REGISTRY
from scripts.inference.convert_composer_to_hf import convert_composer_to_hf


def delete_transformers_cache():
    hf_cache_home = os.path.expanduser(
        os.getenv(
            'HF_HOME',
            os.path.join(os.getenv('XDG_CACHE_HOME', '~/.cache'),
                         'huggingface')))
    HF_MODULES_CACHE = os.getenv('HF_MODULES_CACHE',
                                 os.path.join(hf_cache_home, 'modules'))
    if os.path.exists(HF_MODULES_CACHE) and os.path.isdir(HF_MODULES_CACHE):
        shutil.rmtree(HF_MODULES_CACHE)


def get_config(
        conf_path: str = 'scripts/train/yamls/pretrain/testing.yaml'
) -> DictConfig:
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    with open(conf_path) as f:
        test_cfg = om.load(f)

    return cast(DictConfig, test_cfg)


@pytest.mark.parametrize('model', ['mpt', 'neo', 'llama2'])
def test_convert_and_generate(model: str, tmp_path: pathlib.Path):
    delete_transformers_cache()

    om_cfg = None
    if model == 'mpt':
        om_cfg = get_config(
            conf_path='scripts/train/yamls/pretrain/testing.yaml')
    elif model == 'neo':
        om_cfg = get_config(
            conf_path='scripts/train/yamls/pretrain/gpt-neo-125m.yaml')
        om_cfg['model']['config_overrides']['hidden_size'] = 36
    elif model == 'llama2':
        if 'HUGGING_FACE_HUB_TOKEN' not in os.environ:
            pytest.skip(
                'The CI cluster does not have access to the Llama models, so skip this test.'
            )
        om_cfg = get_config(
            conf_path='scripts/train/yamls/pretrain/gpt-neo-125m.yaml')
        om_cfg['model'][
            'pretrained_model_name_or_path'] = 'meta-llama/Llama-2-7b-hf'
        om_cfg['model']['config_overrides']['num_hidden_layers'] = 2
        om_cfg['model']['use_auth_token'] = True
        om_cfg['tokenizer']['name'] = 'meta-llama/Llama-2-7b-hf'
    else:
        raise ValueError(f'Unknown model {model}')
    assert om_cfg is not None

    om_cfg['model']['init_device'] = 'cpu'
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        om_cfg.tokenizer.name, use_auth_token=model == 'llama2')
    original_model = COMPOSER_MODEL_REGISTRY[om_cfg['model'].name](
        om_cfg['model'], tokenizer)
    trainer = Trainer(model=original_model, device='cpu')
    trainer.save_checkpoint(os.path.join(tmp_path, 'checkpoint.pt'))

    args = Namespace(composer_path=os.path.join(tmp_path, 'checkpoint.pt'),
                     hf_output_path=os.path.join(tmp_path, 'hf-output-folder'),
                     output_precision='fp32',
                     local_checkpoint_save_location=None,
                     hf_repo_for_upload=None,
                     test_uploaded_model=False)
    convert_composer_to_hf(args)

    loaded_config = transformers.AutoConfig.from_pretrained(
        os.path.join(tmp_path, 'hf-output-folder'), trust_remote_code=True)
    loaded_model = transformers.AutoModelForCausalLM.from_pretrained(
        os.path.join(tmp_path, 'hf-output-folder'),
        config=loaded_config,
        trust_remote_code=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        os.path.join(tmp_path, 'hf-output-folder'), trust_remote_code=True)

    output = loaded_model.generate(tokenizer('hello',
                                             return_tensors='pt')['input_ids'],
                                   max_new_tokens=1)
    assert output.shape == (1, 2 + (1 if model == 'llama2' else 0))

    assert sum(p.numel() for p in original_model.model.parameters()) == sum(
        p.numel() for p in loaded_model.parameters())
    assert all(
        str(type(module1)).split('.')[-1] == str(type(module2)).split('.')[-1]
        for module1, module2 in zip(original_model.model.modules(),
                                    loaded_model.modules()))
    for p1, p2 in zip(original_model.model.parameters(),
                      loaded_model.parameters()):
        assert torch.allclose(p1, p2)

    delete_transformers_cache()


@pytest.mark.gpu
def test_convert_and_generate_triton(tmp_path: pathlib.Path):
    delete_transformers_cache()

    cfg = get_config()
    cfg['model']['init_device'] = 'cpu'
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        'EleutherAI/gpt-neox-20b')
    model = ComposerMPTCausalLM(cfg['model'], tokenizer)
    trainer = Trainer(model=model)
    trainer.save_checkpoint(os.path.join(tmp_path, 'checkpoint.pt'))

    args = Namespace(composer_path=os.path.join(tmp_path, 'checkpoint.pt'),
                     hf_output_path=os.path.join(tmp_path, 'hf-output-folder'),
                     output_precision='fp32',
                     local_checkpoint_save_location=None,
                     hf_repo_for_upload=None,
                     test_uploaded_model=False)
    convert_composer_to_hf(args)

    config = transformers.AutoConfig.from_pretrained(os.path.join(
        tmp_path, 'hf-output-folder'),
                                                     trust_remote_code=True)
    config.attn_config['attn_impl'] = 'triton'
    model = transformers.AutoModelForCausalLM.from_pretrained(
        os.path.join(tmp_path, 'hf-output-folder'),
        config=config,
        trust_remote_code=True)
    model.to(device='cuda', dtype=torch.bfloat16)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        os.path.join(tmp_path, 'hf-output-folder'), trust_remote_code=True)

    output = model.generate(tokenizer(
        'hello', return_tensors='pt')['input_ids'].to(device='cuda'),
                            max_new_tokens=1)
    assert output.shape == (1, 2)

    delete_transformers_cache()


def test_convert_and_generate_meta(tmp_path: pathlib.Path):
    delete_transformers_cache()

    from composer.utils import dist
    gathered_paths = dist.all_gather_object(tmp_path)
    tmp_path_gathered = gathered_paths[0]

    om_cfg = get_config(conf_path='scripts/train/yamls/pretrain/testing.yaml')

    om_cfg['model']['init_device'] = 'cpu'
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        om_cfg.tokenizer.name)
    original_model = COMPOSER_MODEL_REGISTRY[om_cfg['model'].name](
        om_cfg['model'], tokenizer)
    trainer = Trainer(model=original_model, device='cpu')
    trainer.save_checkpoint(os.path.join(tmp_path_gathered, 'checkpoint.pt'))

    # patch in the meta device for testing
    sd = torch.load(os.path.join(tmp_path_gathered, 'checkpoint.pt'),
                    map_location='cpu')
    sd['state']['integrations']['huggingface']['model']['config']['content'][
        'init_device'] = 'meta'
    torch.save(sd, os.path.join(tmp_path_gathered, 'checkpoint.pt'))

    args = Namespace(composer_path=os.path.join(tmp_path_gathered,
                                                'checkpoint.pt'),
                     hf_output_path=os.path.join(tmp_path_gathered,
                                                 'hf-output-folder'),
                     output_precision='fp32',
                     local_checkpoint_save_location=None,
                     hf_repo_for_upload=None,
                     test_uploaded_model=False)
    convert_composer_to_hf(args)

    loaded_config = transformers.AutoConfig.from_pretrained(
        os.path.join(tmp_path_gathered, 'hf-output-folder'),
        trust_remote_code=True)
    loaded_model = transformers.AutoModelForCausalLM.from_pretrained(
        os.path.join(tmp_path_gathered, 'hf-output-folder'),
        config=loaded_config,
        trust_remote_code=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        os.path.join(tmp_path_gathered, 'hf-output-folder'),
        trust_remote_code=True)

    output = loaded_model.generate(tokenizer('hello',
                                             return_tensors='pt')['input_ids'],
                                   max_new_tokens=1)
    assert output.shape == (1, 2)

    assert sum(p.numel() for p in original_model.model.parameters()) == sum(
        p.numel() for p in loaded_model.parameters())
    assert all(
        str(type(module1)).split('.')[-1] == str(type(module2)).split('.')[-1]
        for module1, module2 in zip(original_model.model.modules(),
                                    loaded_model.modules()))
    for p1, p2 in zip(original_model.model.parameters(),
                      loaded_model.parameters()):
        assert torch.allclose(p1, p2)

    delete_transformers_cache()
