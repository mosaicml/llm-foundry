# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from llmfoundry.data.data import ConcatTokensDataset, NoConcatDataset
from llmfoundry.data.dataloader import build_dataloader
from llmfoundry.data.finetuning import (Seq2SeqFinetuningCollator,
                                        StreamingFinetuningDataset,
                                        build_finetuning_dataloader)
from llmfoundry.data.packing import (BinPackCollator, auto_packing_ratio,
                                     profile_packing)
from llmfoundry.data.text_data import (ConcatenatedSequenceCollatorWrapper,
                                       StreamingTextDataset,
                                       build_text_dataloader)
from llmfoundry.registry import (collators, data_specs, dataloaders,
                                 dataset_replication_validators)

from llmfoundry.data.utils import get_data_spec as utils_get_data_spec
from llmfoundry.data.utils import get_finetuning_collator as utils_get_finetuning_collator
from llmfoundry.data.utils import get_text_collator as utils_get_text_collator
from llmfoundry.data.utils import validate_ds_replication as utils_validate_ds_replication

dataloaders.register('text', func=build_text_dataloader)
dataloaders.register('finetuning', func=build_finetuning_dataloader)

dataset_replication_validators.register(
    'dataset_replication_validator', func=utils_validate_ds_replication
)
collators.register(
    'finetuning_collator', func=utils_get_finetuning_collator
)
collators.register(
    'text_collator', func=utils_get_text_collator
)
data_specs.register(
    'data_spec', func=utils_get_data_spec
)

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
