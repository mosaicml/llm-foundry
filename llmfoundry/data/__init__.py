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
)
from llmfoundry.data.utils import (
    get_data_spec,
    get_finetuning_collator,
    get_text_collator,
    validate_ds_replication,
)
from llmfoundry.registry import (
    collators,
    data_specs,
    dataloaders,
    dataset_replication_validators
)

dataloaders.register('text', func=build_text_dataloader)
dataloaders.register('finetuning', func=build_finetuning_dataloader)

dataset_replication_validators.register('dataset_replication_validator',
                                        func=validate_ds_replication)
collators.register('finetuning_collator', func=get_finetuning_collator)
collators.register('text_collator', func=get_text_collator)
data_specs.register('data_spec', func=get_data_spec)

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
]
