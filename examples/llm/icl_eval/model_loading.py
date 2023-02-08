# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

from typing import Union

import transformers
from composer.metrics.nlp import HFCrossEntropy, Perplexity
from composer.models.huggingface import HuggingFaceModel
from omegaconf import DictConfig
from omegaconf import OmegaConf as om

from examples.llm.src.mosaic_gpt import ComposerMosaicGPT


def init_huggingface_causal_lm(
        pretrained_model_name_or_path: str) -> HuggingFaceModel:
    """Construct a HuggingFaceModel from a pretrained HF Causal LM.

    Args:
        pretrained_model_name_or_path (str): The name or path to the pretrained HF model

    Returns:
        model HuggingFaceModel:
            a HuggingFaceModel wrapper around a HF Causal LM with the pretrained weights loaded.
    """
    print(f'Building HF Causal LM w/ name: {pretrained_model_name_or_path}')
    model = transformers.AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path)
    return HuggingFaceModel(model=model,
                            tokenizer=None,
                            metrics=[HFCrossEntropy(),
                                     Perplexity()])


def init_mosaic_gpt(config_path: str, allow_meta: bool) -> ComposerMosaicGPT:
    """Construct a MosaicGPT model from a config file.

    Constructs the MosaicGPT model from the yaml config and extracts the FSDP config
    to be passed to Trainer.

    Args:
        config_path (str): path to MosaicGPT YAML config. Must be an absolute path

    Returns:
        model ComposerMosaicGPT:
            a ComposerMosaicGPT model.
    """
    with open(config_path) as f:
        cfg = om.load(f)

    if not allow_meta:
        cfg.model.device = 'cpu'

    print('Building MosaicGPT w/ config: ')
    print(om.to_yaml(cfg.model))

    return ComposerMosaicGPT(cfg.model)


def load_composer_model(
        cfg: DictConfig) -> Union[HuggingFaceModel, ComposerMosaicGPT]:
    if cfg.model.model_type == 'pretrained_hf':
        return init_huggingface_causal_lm(
            cfg.model.pretrained_model_name_or_path)
    elif cfg.model.model_type == 'mosaic_gpt':
        return init_mosaic_gpt(cfg.model.config_path,
                               (cfg.fsdp_config is not None))
    else:
        raise ValueError(f'Unrecogized model type: {cfg.model.model_type}')
