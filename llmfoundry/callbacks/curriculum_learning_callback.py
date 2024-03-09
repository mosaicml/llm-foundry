# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Enable curriculum learning by resuming with a different dataset.

This callback is currently experimental. The API may change without warning in
the future.
"""

import logging
from typing import Any, Dict

from composer.core import State
from composer.loggers import Logger
from streaming import StreamingDataset
from torch.utils.data import DataLoader

from llmfoundry.interfaces import CallbackWithConfig

log = logging.getLogger(__name__)


class CurriculumLearning(CallbackWithConfig):
    """Starts an epoch with a different dataset when resuming from a checkpoint.

    This callback is currently experimental. The API may change without warning in the future.

    Args:
        dataset_index (int): The index of the dataset currently being used.
        current_dataset_config (Dict): The configuration of the dataset currently
            being used.
    """

    def __init__(self, dataset_index: int, train_config: Dict):
        self.dataset_index = dataset_index
        self.saved_dataset_index = 0
        self.all_dataset_configs = []
        self.current_dataset_state = {}
        # The current dataset config is resolved and passed in train.py
        self.current_dataset_config = train_config['dataloader']

    def before_load(self, state: State, logger: Logger):
        del logger

        # Save the current dataset state so we can restore it correctly
        # if we are resuming with a new dataset.
        train_loader = state.train_dataloader
        # Check if we are using a DataLoader and StreamingDataset
        if not isinstance(train_loader, DataLoader):
            raise ValueError(
                f'CurriculumLearning callback can only be used with a train ',
                f'dataloader of type DataLoader, but got {type(train_loader)}.')
        dataset = train_loader.dataset
        if not isinstance(dataset, StreamingDataset):
            raise ValueError(
                f'CurriculumLearning callback only supports StreamingDataset ',
                f'because it requires loading and saving dataset state. ',
                f'Instead, got a dataset of type {type(dataset)}')
        assert isinstance(dataset, StreamingDataset)
        # Save the current dataset state so we can restore it if needed.
        self.current_dataset_state = dataset.state_dict(  # type: ignore
            num_samples=0, from_beginning=False)

    def after_load(self, state: State, logger: Logger):
        del logger

        # As saved_dataset_index is loaded from state_dict, this only runs when
        # a user explicitly increments the dataset_index and not on any other
        # resumption, including autoresume.
        train_loader = state._train_dataloader
        assert isinstance(
            train_loader,
            DataLoader), 'CurriculumLearning callback requires a DataLoader.'
        dataset = train_loader.dataset
        assert isinstance(
            dataset, StreamingDataset
        ), 'CurriculumLearning callback requires a StreamingDataset.'
        if self.saved_dataset_index < self.dataset_index:
            # Ignore the dataset state that was read in from the checkpoint, and
            # replace with the new dataset state. This preserves resumption info.
            if self.current_dataset_state['epoch'] < 0:
                # Make sure the epoch in the loaded state dict is not negative.
                # Since `__iter__` has not yet been called on the dataset, the
                # epoch index in the dataset will still be -1. We need to ensure
                # that we set the epoch correctly to 0 in this case.
                self.current_dataset_state['epoch'] = 0
            dataset.load_state_dict(  # type: ignore
                self.current_dataset_state)
            # Start a new epoch since we are using a new dataset.
            # This will also reset the sample_in_epoch written to checkpoint,
            # making sure that subsequent resumptions proceed correctly.
            state.timestamp = state.timestamp.to_next_epoch()
            # Append the new dataset config to the list of all dataset configs.
            self.all_dataset_configs.append(self.current_dataset_config)
        elif self.dataset_index == 0 and len(self.all_dataset_configs) == 0:
            # Make sure to track our current dataset config if we are just starting training.
            self.all_dataset_configs.append(self.current_dataset_config)

    def state_dict(self):
        return {
            'dataset_index': self.dataset_index,
            'all_dataset_configs': self.all_dataset_configs
        }

    def load_state_dict(self, state: Dict[str, Any]):
        self.saved_dataset_index = state.get('dataset_index', 0)
        self.all_dataset_configs = state.get('all_dataset_configs', [])
