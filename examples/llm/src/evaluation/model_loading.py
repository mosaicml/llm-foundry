# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Dict, Union
from urllib.parse import urlparse

import torch
from composer.utils import reproducibility
from composer.utils.file_helpers import get_file
from composer.utils.object_store import S3ObjectStore
from omegaconf import OmegaConf as om
from transformers import AutoModelForCausalLM, AutoTokenizer

from examples.llm.src.model_registry import COMPOSER_MODEL_REGISTRY
from examples.llm.src.tokenizer import TOKENIZER_REGISTRY, LLMTokenizer

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device(
    'cpu')
CHECKPOINT_DIR = f'{os.path.dirname(__file__)}/model_checkpoints'


def _get_checkpoint_name_from_path(path: str) -> str:
    """To go from checkpoint name to path, replace '|' with '/'."""
    return path.lstrip('/').replace('/', '|')


def download_starting_checkpoints(path_to_download: str) -> str:
    """Downloads the pretrained checkpoints to start from.

    Args:
        path_to_download (str): an http:// or s3:// path. Local files not
            yet supported.
    """
    load_object_store = None
    parsed_first_checkpoint = urlparse(path_to_download)
    if parsed_first_checkpoint.scheme == 's3':
        load_object_store = S3ObjectStore(bucket=parsed_first_checkpoint.netloc)

    parsed_path = urlparse(path_to_download)
    download_path = (parsed_path.path if parsed_path.scheme == 's3' else
                     parsed_first_checkpoint.path).lstrip('/')
    checkpoint_name = _get_checkpoint_name_from_path(parsed_path.path)
    local_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
    if not os.path.exists(local_path):
        get_file(destination=local_path,
                 path=download_path,
                 object_store=load_object_store,
                 progress_bar=True)

    return checkpoint_name


def init_huggingface_causal_lm(
        checkpoint: str
) -> Dict[str, Union[AutoModelForCausalLM, AutoTokenizer]]:
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    return {'model': model, 'tokenizer': tokenizer}


def init_composer_ckpt_from_yaml(
        checkpoint: str,
        config: str) -> Dict[str, Union[torch.nn.Module, LLMTokenizer, str]]:
    """Load a MosaicGPT model and LLMTokenizer from a checkpoint.

    Constructs the MosaicGPT model and LLMTokenizer from the yaml config and
    attempts to initialize its weights from the state dict in `checkpoint`.
    If there is an error during state dict loading, returns a randomly
    initialized model.

    Args:
        checkpoint (str): Pytorch .pt checkpoint path. Must be located in
            `CHECKPOINT_DIR` or on s3
        config (str): YAML config. Must be an absolute path

    Returns:
        model, tok, precision (Dict[str, Union[MosaicGPT, LLMTokenizer, str]]):
            Model and tokenizer to be used to build the lm_eval.base.LM wrapper
            as well as precision context.
    """
    with open(config) as f:
        cfg = om.load(f)
    print('Building MosaicGPT w/ config: ')
    print(om.to_yaml(cfg))
    seed = cfg.seed if hasattr(cfg, 'seed') else 17
    reproducibility.seed_all(seed)

    # Build Model
    print('Initializing model...')
    cfg.model.device = str(DEVICE)
    model = COMPOSER_MODEL_REGISTRY[cfg.model.name](cfg.model)
    pre = next(model.parameters()).clone().data

    if checkpoint.startswith('s3://'):
        checkpoint = download_starting_checkpoints(checkpoint)
        print(f'Downloaded from s3 to local path {checkpoint}')

    model.load_state_dict(
        torch.load(f'{CHECKPOINT_DIR}/{checkpoint}',
                   map_location=DEVICE)['state']['model'])
    post = next(model.parameters()).clone().data
    assert not torch.equal(pre, post)
    print('Successfully loaded model weights')

    n_params = sum(p.numel() for p in model.parameters())
    print(f'{n_params=:.2e}')

    tokenizer = TOKENIZER_REGISTRY[cfg.tokenizer.type](**cfg.tokenizer.args)
    return {
        'model': model.model,
        'tokenizer': tokenizer,
        'precision': cfg.precision
    }


MODEL_LOADERS = {
    'composer_checkpoint': init_composer_ckpt_from_yaml,
    'pretrained_hf': init_huggingface_causal_lm,
}
