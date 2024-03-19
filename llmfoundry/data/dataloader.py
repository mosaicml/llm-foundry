# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Dataloader builder utilities."""

from composer import DataSpec
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from transformers import PreTrainedTokenizerBase

from llmfoundry.utils.registry_utils import builder, registry


def build_dataloader(cfg: DictConfig, tokenizer: PreTrainedTokenizerBase,
                     device_batch_size: int) -> DataSpec:
    """Builds a dataloader from a config.

    Args:
        cfg (DictConfig): An omegaconf dictionary used to configure the loader.
        tokenizer (PreTrainedTokenizerBase): The tokenizer that the model will use.
        device_batch_size (int): The size of the batches (number of examples)
            that the dataloader will produce.
    """
    kwargs = om.to_container(cfg)
    assert isinstance(kwargs, dict)

    name = kwargs.pop('name')
    extra_kwargs = {
        'tokenizer': tokenizer,
        'device_batch_size': device_batch_size
    }

    return builder(
        name=name,
        registry=registry.dataloaders,
        partial_function=True,
        pre_validation_function=None,
        post_validation_function=None,
        kwargs=kwargs | extra_kwargs,
    )
