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
from llmfoundry.utils.exceptions import (
    BaseContextualError,
    TrainDataLoaderLocation,
)

log = logging.getLogger(__name__)

__all__ = ['CurriculumLearning']


@experimental_class('CurriculumLearning callback')
class CurriculumLearning(CallbackWithConfig):
    """Starts an epoch with a different dataset when resuming from a checkpoint.

    Args:
        train_config (Dict): The configuration of the dataset currently
            being used. Note that this is the full train config and must
            contain the 'train_loader' key.
        dataset_index (int): The index of the dataset currently being used.
    """

    def __init__(self, train_config: Dict, dataset_index: int):
        self.dataset_index = dataset_index
        self.saved_dataset_index = 0
        self.all_dataset_configs = []
        self.current_dataset_state = {}
        # The current dataset config is resolved and passed in train.py
        self.current_dataset_config = train_config['train_loader']

    def before_load(self, state: State, logger: Logger):
        del logger

        # Ensure all duration units are the same as max_duration
        datamix_units = [datamix['duration'].unit for datamix in self._schedule]
        assert state.max_duration is not None, 'max_duration should have beeen set.'
        if any(state.max_duration.unit != unit for unit in datamix_units):
            raise ValueError((
                f'All durations in the schedule must have the same units as '
                f'the max_duration. Expected {state.max_duration.unit}, but '
                f'got {datamix_units}.'
            ))

        # Ensure schedule duration is equal to max_duration
        schedule_duration = Time(0, state.max_duration.unit)
        for datamix in self._schedule:
            assert isinstance(datamix['duration'], Time)
            schedule_duration += datamix['duration']
        if schedule_duration != state.max_duration:
            raise ValueError((
                'The sum of all durations in the schedule must be equal to the '
                'max_duration.'
            ))

        self._validate_dataloader(state.train_dataloader)

    def after_load(self, state: State, logger: Logger):
        del logger  # unused

        self._validate_dataloader(state.train_dataloader)

        # If checkpoint was saved before iteration was incremented, we need to increment it now
        if ((
            self._schedule[self._schedule_index]['duration'].unit
            == TimeUnit.TOKEN and state.timestamp.token_in_iteration
            >= self._schedule[self._schedule_index]['duration'].value
        ) or (
            self._schedule[self._schedule_index]['duration'].unit
            == TimeUnit.EPOCH and state.timestamp.epoch_in_iteration
            >= self._schedule[self._schedule_index]['duration'].value
        )):
            log.warning((
                'The CurriculumLearning callback has detected that the previous run did not correctly '
                'increment the iteration.'
            ))
            self._schedule_index += 1
            state.timestamp = state.timestamp.to_next_iteration()

    def iteration_start(self, state: State, logger: Logger):
        # Swap the dataset if starting a new iteration that's not the original datamix
        if self._schedule_index > 0:
            # TODO: trainer._train_data_spec should be updated whenever the dataloader is updated
            # Dataloaders with the same prefix access the same shared memory
            # which is stale
            clean_stale_shared_memory()
            datamix = copy.deepcopy(self._schedule[self._schedule_index])
            data_spec = self._build_train_loader(
                train_loader_config=datamix['train_loader'],
                logger=logger,
            )
            state.set_dataloader(
                dataloader=data_spec.dataloader,
                dataloader_label='train',
            )
            state.train_dataloader = state.dataloader
            self._validate_dataloader(state.train_dataloader)

        # Set the length of the new iteration
        state._iteration_length = self._schedule[self._schedule_index
                                                ]['duration']

    def iteration_end(self, state: State, logger: Logger):
        del state, logger  # unused

        self._schedule_index += 1

    def state_dict(self):
        return {
            'schedule': self._schedule,
            'schedule_index': self._schedule_index,
        }

    def load_state_dict(self, state: dict[str, Any]):
        self._schedule_index = state['schedule_index']

        # Ensure that the schedule has not changed on previously trained datamixes
        for idx in range(state['schedule_index']):
            if self._schedule[idx] != state['schedule'][idx]:
                raise ValueError((
                    f'Previous datamixes must stay the same across ',
                    f'resumptions. Expected {state["schedule"][idx]} but got ',
                    f'{self._schedule[idx]}',
                ))

        # Ensure that the datamix has not changed on the current datamix
        current_loader = self._schedule[self._schedule_index]['train_loader']
        saved_loader = state['schedule'][self._schedule_index]['train_loader']
        if current_loader != saved_loader:
            raise ValueError((
                f'The current datamix must stay the same across resumptions. ',
                f'Expected {saved_loader} but got {current_loader}',
            ))

        # Ensure that the current datamix duration is greater than timestamp
        duration = self._schedule[self._schedule_index]['duration']
        if duration.unit != TimeUnit.TOKEN and duration.unit != TimeUnit.EPOCH:
            raise ValueError((
                f'Duration must be in terms of tokens or epochs, but got ',
                f'{duration.unit}.',
            ))
        if ((
            duration.unit == TimeUnit.TOKEN and
            duration > state['timestamp'].token_in_iteration
        ) or (
            duration.unit == TimeUnit.EPOCH and
            duration > state['timestamp'].epoch_in_iteration
        )):
            raise ValueError((
                'The duration of the current datamix must be less or equal to '
                'than the saved timestamp.'
            ))

    def _build_train_loader(
        self,
        train_loader_config: dict[str, Any],
        logger: Logger,
    ) -> DataSpec:
        from llmfoundry.data.dataloader import build_dataloader

        # Copied from scripts/train/train.py
        log.info(
            f'Building train loader in CurriculumLearning callback for dataset {self._schedule_index}',
        )
        assert self.tokenizer is not None
        try:
            return build_dataloader(
                train_loader_config,
                self.tokenizer,
                self.device_train_batch_size,
            )
        except BaseContextualError as e:
            e.location = TrainDataLoaderLocation
            raise e

    def _validate_dataloader(self, train_loader: Any):
        # Check if we are using a DataLoader and StreamingDataset
        if not isinstance(train_loader, DataLoader):
            raise ValueError(
                f'CurriculumLearning callback can only be used with a train ',
                f'dataloader of type DataLoader, but got {type(train_loader)}.',
            )
        dataset = train_loader.dataset
        if not isinstance(dataset, StreamingDataset):
            raise ValueError(
                f'CurriculumLearning callback only supports StreamingDataset ',
                f'because it requires loading and saving dataset state. ',
                f'Instead, got a dataset of type {type(dataset)}',
            )
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
            DataLoader,
        ), 'CurriculumLearning callback requires a DataLoader.'
        dataset = train_loader.dataset
        assert isinstance(
            dataset,
            StreamingDataset,
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
            'all_dataset_configs': self.all_dataset_configs,
        }

    def load_state_dict(self, state: Dict[str, Any]):
        self.saved_dataset_index = state.get('dataset_index', 0)
        self.all_dataset_configs = state.get('all_dataset_configs', [])
