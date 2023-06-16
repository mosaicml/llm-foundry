# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import os
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
        conf_path='scripts/train/yamls/pretrain/testing.yaml') -> DictConfig:
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    with open(conf_path) as f:
        test_cfg = om.load(f)
    return cast(DictConfig, test_cfg)


def test_convert_and_generate_torch(tmp_path):
    delete_transformers_cache()

    cfg = get_config()
    cfg['model']['init_device'] = 'cpu'
    cfg['model']['attn_config']['attn_impl'] = 'torch'
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
    config.attn_config['attn_impl'] = 'torch'
    model = transformers.AutoModelForCausalLM.from_pretrained(
        os.path.join(tmp_path, 'hf-output-folder'),
        config=config,
        trust_remote_code=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        os.path.join(tmp_path, 'hf-output-folder'), trust_remote_code=True)

    output = model.generate(tokenizer('hello',
                                      return_tensors='pt')['input_ids'],
                            max_new_tokens=1)
    assert output.shape == (1, 2)

    delete_transformers_cache()


@pytest.mark.gpu
def test_convert_and_generate_triton(tmp_path):
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
