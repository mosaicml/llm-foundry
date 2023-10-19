# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Dataloader builder utilities."""

from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase

from llmfoundry.data.text_data import build_text_dataloader

from llmfoundry.data.denoising import build_text_denoising_dataloader

from llmfoundry.data.finetuning.dataloader import build_finetuning_dataloader


def build_dataloader(cfg: DictConfig, tokenizer: PreTrainedTokenizerBase,
                     device_batch_size: int):
    if cfg.name == 'text':
        return build_text_dataloader(
            cfg,
            tokenizer,
            device_batch_size,
        )
    elif cfg.name == 'text_denoising':
        return build_text_denoising_dataloader(
            cfg,
            tokenizer,
            device_batch_size,
        )
    elif cfg.name == 'finetuning':
        return build_finetuning_dataloader(
            cfg,
            tokenizer,
            device_batch_size,
        )
    else:
        raise ValueError(f'Not sure how to build dataloader with config: {cfg}')
