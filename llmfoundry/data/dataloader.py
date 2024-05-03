# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Dataloader builder utilities."""

from typing import Any, Dict

from composer import DataSpec
from transformers import PreTrainedTokenizerBase

from llmfoundry import registry
from llmfoundry.utils.registry_utils import construct_from_registry

__all__ = [
    'build_dataloader',
]


def build_dataloader(
    cfg: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    device_batch_size: int,
) -> DataSpec:
    """Builds a dataloader from a config.

    Args:
        cfg (DictConfig): An omegaconf dictionary used to configure the loader.
        tokenizer (PreTrainedTokenizerBase): The tokenizer that the model will use.
        device_batch_size (int): The size of the batches (number of examples)
            that the dataloader will produce.
    """
    name = cfg.pop('name')
    kwargs: Dict[str, Any] = {
        **cfg,
        'tokenizer': tokenizer,
        'device_batch_size': device_batch_size,
    }

    return construct_from_registry(
        name=name,
        registry=registry.dataloaders,
        partial_function=False,
        pre_validation_function=None,
        post_validation_function=None,
        kwargs=kwargs,
    )
