# Copyright 2022-2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Dataloader builder utilities."""

from composer import DataSpec
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase

from llmfoundry.data.denoising import build_text_denoising_dataloader
from llmfoundry.data.finetuning.dataloader import build_finetuning_dataloader
from llmfoundry.data.text_data import build_text_dataloader

LOADER_NAME_TO_FUNCTION = {
    'text': build_text_dataloader,
    'text_denoising': build_text_denoising_dataloader,
    'finetuning': build_finetuning_dataloader,
}


def build_dataloader(cfg: DictConfig, tokenizer: PreTrainedTokenizerBase,
                     device_batch_size: int) -> DataSpec:
    """Builds a dataloader from a config.

    Args:
        cfg (DictConfig): An omegaconf dictionary used to configure the loader.
        tokenizer (PreTrainedTokenizerBase): The tokenizer that the model will use.
        device_batch_size (int): The size of the batches (number of examples)
            that the dataloader will produce.
    """
    if cfg.name not in LOADER_NAME_TO_FUNCTION:
        allowed = ', '.join(LOADER_NAME_TO_FUNCTION.keys())
        raise ValueError(f'Expected dataloader name to be one of {allowed}' +
                         f' but found name "{cfg.name}" in config: {cfg}')

    return LOADER_NAME_TO_FUNCTION[cfg.name](cfg, tokenizer, device_batch_size)
