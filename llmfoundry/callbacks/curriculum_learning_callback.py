# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Enable curriculum learning by resuming with a different dataset.

This callback is currently experimental. The API may change in the future.
"""

import logging
from typing import Any, Dict, Iterable, Union

from composer.core import Callback, State
from composer.loggers import Logger
from streaming import StreamingDataset
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


class CurriculumLearning(Callback):
    """Starts an epoch with a different dataset when resuming from a checkpoint.

    This callback is currently experimental. The API may change in the future.

    Args:
        dataset_index (int): The index of the dataset currently being used.
        current_dataset_config (Dict): The configuration of the dataset currently
            being used.
        dataloader (Union[DataLoader, Iterable]): The training dataloader currently being used.
    """

    def __init__(self, dataset_index: int, dataloader: Union[DataLoader,
                                                             Iterable],
                 current_dataset_config: Dict):
        self.dataset_index = dataset_index
        self.saved_dataset_index = 0
        self.all_dataset_configs = []
        # The current dataset config is resolved and passed in train.py
        self.current_dataset_config = current_dataset_config
        self.new_dataset_setup = False
        self.dataset_config_appended = False

        # Must pass in dataset directly since it is not actually accessible at Event.INIT in
        # Composer. We need to get the new dataset state to override checkpoint dataset state.
        # Check if we are using a DataLoader and StreamingDataset
        if not isinstance(dataloader, DataLoader):
            raise ValueError(
                f'CurriculumLearning callback can only be used with a train ',
                f'dataloader of type DataLoader, but got {type(dataloader)}.')
        dataset = dataloader.dataset
        if not isinstance(dataset, StreamingDataset):
            raise ValueError(
                f'CurriculumLearning callback only supports StreamingDataset ',
                f'because it requires loading and saving dataset state. ',
                f'Instead, got a dataset of type {type(dataset)}')
        # Save the current dataset state so we can restore it if needed.
        self.current_dataset_state = dataset.state_dict(0, False)

    def after_load(self, state: State, logger: Logger):
        del logger

        # As saved_dataset_index is loaded from state_dict, this only run when
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
                # Otherwise, the state will be considered stale, and will be ignored.
                self.current_dataset_state['epoch'] = 0
            dataset.load_state_dict(self.current_dataset_state)
            # Start a new epoch since we are using a new dataset.
            # This will also reset the sample_in_epoch written to checkpoint,
            # making sure that subsequent resumptions proceed correctly.
            state.timestamp = state.timestamp.to_next_epoch()
            self.new_dataset_setup = True
        elif self.dataset_index == 0 and len(self.all_dataset_configs) == 0:
            # Make sure to save our current dataset config if we are just starting training.
            self.new_dataset_setup = True

    def state_dict(self):
        if self.new_dataset_setup and not self.dataset_config_appended:
            # Append the new dataset config to the list of all dataset configs.
            self.all_dataset_configs.append(self.current_dataset_config)
            # Only append the dataset config once, not on every single checkpoint save.
            self.dataset_config_appended = True
        return {
            'dataset_index': self.dataset_index,
            'all_dataset_configs': self.all_dataset_configs
        }

    def load_state_dict(self, state: Dict[str, Any]):
        self.saved_dataset_index = state.get('dataset_index', 0)
        self.all_dataset_configs = state.get('all_dataset_configs', [])
