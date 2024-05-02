# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from llmfoundry.data.data import ConcatTokensDataset, NoConcatDataset
from llmfoundry.data.dataloader import build_dataloader
from llmfoundry.data.finetuning import (
    Seq2SeqFinetuningCollator,
    StreamingFinetuningDataset,
    build_finetuning_dataloader,
)
from llmfoundry.data.packing import (
    BinPackCollator,
    auto_packing_ratio,
    profile_packing,
)
from llmfoundry.data.text_data import (
    ConcatenatedSequenceCollatorWrapper,
    StreamingTextDataset,
    build_text_dataloader,
    get_tokens_per_batch_func,
)
from llmfoundry.registry import dataloaders

dataloaders.register('text', func=build_text_dataloader)
dataloaders.register('finetuning', func=build_finetuning_dataloader)

__all__ = [
    'Seq2SeqFinetuningCollator',
    'build_finetuning_dataloader',
    'StreamingFinetuningDataset',
    'StreamingTextDataset',
    'build_text_dataloader',
    'NoConcatDataset',
    'ConcatTokensDataset',
    'build_dataloader',
    'BinPackCollator',
    'auto_packing_ratio',
    'profile_packing',
    'ConcatenatedSequenceCollatorWrapper',
    'get_tokens_per_batch_func',
]
