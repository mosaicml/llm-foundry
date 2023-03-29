# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0
import sys
import warnings

import torch
from composer.core import get_precision_context
from composer.utils import get_device
from omegaconf import OmegaConf as om

from examples.llm.src import COMPOSER_MODEL_REGISTRY


def build_composer_model(model_cfg, tokenizer_cfg):
    warnings.filterwarnings(
        action='ignore',
        message='Torchmetrics v0.9 introduced a new argument class property')
    try:
        return COMPOSER_MODEL_REGISTRY[model_cfg.name](model_cfg, tokenizer_cfg)
    except:
        raise ValueError(
            f'Not sure how to build model with name={model_cfg.name}')


def get_mosaicgpt_inference_model(checkpoint_yaml_path: str):
    with open(checkpoint_yaml_path) as f:
        cfg = om.load(f)
    # set init_device to cpu for checkpoint loading
    # ToDo: Directly load a checkpoint into 'meta' model
    cfg.model.init_device = 'cpu'
    model = build_composer_model(cfg.model, cfg.tokenizer)

    ckpt_load_path = cfg.get('load_path', None)  # type: ignore
    if ckpt_load_path is None:
        raise ValueError('Checkpoint load_path is required for exporting.')

    checkpoint = torch.load(ckpt_load_path, map_location='cpu')

    if 'state' in checkpoint.keys():
        # it's a full training checkpoint
        model.load_state_dict(checkpoint['state']['model'], strict=True)
    else:
        # it's a weights-only checkpoint
        model.load_state_dict(checkpoint, strict=True)

    if model.tokenizer.pad_token_id is None:
        warnings.warn(
            'pad_token_id is not set for the tokenizer. Using eos_token_id as pad_token_id.'
        )
        model.tokenizer.pad_token_id = model.tokenizer.eos_token_id
    model.tokenizer.padding_side = 'left'

    return model


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('please provide a configuration yaml')
        sys.exit(-1)
    yaml_path = sys.argv[1]
    with open(yaml_path) as f:
        cfg = om.load(f)
    model = get_mosaicgpt_inference_model(yaml_path)
    model.eval()

    generate_kwargs = {
        'max_new_tokens': 100,
        'use_cache': True,
        'do_sample': True,
        'top_p': 0.95,
        'eos_token_id': model.tokenizer.eos_token_id,
    }
    prompts = [
        'My name is',
        'This is an explanation of deep learning to a five year old. Deep learning is',
    ]
    device = get_device(None)
    device.module_to_device(model)
    encoded_inp = model.tokenizer(prompts, return_tensors='pt', padding=True)
    for key, value in encoded_inp.items():
        encoded_inp[key] = device.tensor_to_device(value)

    with torch.no_grad():
        with get_precision_context(
                cfg.get(
                    'precision',  # type: ignore
                    'amp_bf16')):
            generation = model.model.generate(
                input_ids=encoded_inp['input_ids'],
                attention_mask=encoded_inp['attention_mask'],
                **generate_kwargs,
            )

    decoded_out = model.tokenizer.batch_decode(generation,
                                               skip_special_tokens=True)
    print('\n###\n'.join(decoded_out))
