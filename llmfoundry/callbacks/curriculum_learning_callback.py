# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Enable curriculum learning by specifying a schedule of datasets to train on.

This module provides a CurriculumLearning callback that allows for dynamic
dataset switching during training based on a predefined schedule.
"""

import copy
import logging
from dataclasses import dataclass
from typing import Any, Union

from composer import DataSpec
from composer.core import State, Time, TimeUnit, ensure_time
from composer.loggers import Logger
from streaming import StreamingDataset
from streaming.base.util import clean_stale_shared_memory
from torch.utils.data import DataLoader

from llmfoundry.interfaces import CallbackWithConfig
from llmfoundry.utils.exceptions import (
    BaseContextualError,
    TrainDataLoaderLocation,
)

log = logging.getLogger(__name__)

__all__ = ['CurriculumLearning']


@dataclass
class CurriculumLearningState:
    schedule: list[dict[str, Any]]
    schedule_index: int


class CurriculumLearning(CallbackWithConfig):
    """Starts an epoch with a different dataset when resuming from a checkpoint.

    Example duration:
    <number>tok
    Example schedule:
    [
        {
            'duration': <number>tok,
            'dataset': <dataset parameters>,
        },
        {
            'duration': <number>tok,
            'dataset': <dataset parameters>,
        },
        {
            'duration': <number>tok,
            'dataset': <dataset parameters>,
        ],
    ]

    Args:
        train_config (dict): The configuration of the dataset currently
            being used. Note that this is the full train config and must
            contain the 'train_loader', 'device_train_batch_size', and
            'tokenizer' keys.
        duration (Union[Time, str, int]): The duration of the first datamix
            (which corresponds to the train_loader).
        schedule (list[dict[str, Any]]): The list of datamixes to use and their
            durations. Duration units must match max_duration and be in terms of
            a TimeUnit that is supported by Iteration. The duration values must
            be positive. There must be at least one datamix in the schedule. The
            first datamix during training is not included in the schedule. On
            resumption, previously trained on datamixes and durations cannot be
            changed. The duration of the current datamix must be greater than
            the saved timestamp. The dataset must be a StreamingDataset.
    """

    def __init__(
        self,
        train_config: dict[str, Any],
        schedule: list[dict[str, Any]],
        duration: Union[Time, str, int],
    ):
        # Ensure all duration units are in epochs or tokens and values are positive
        self._schedule = schedule
        if len(self._schedule) == 0:
            raise ValueError('The schedule must have at least one datamix.')
        first_datamix = {
            'duration': duration,
            'dataset': train_config['train_loader']['dataset'],
        }
        self._schedule.insert(0, first_datamix)
        for datamix in self._schedule:
            self._validate_datamix(datamix)

        self._schedule_index = 0
        self._train_loader_config: dict[str, Any] = train_config['train_loader']
        self._device_train_batch_size = train_config['device_train_batch_size']
        self._tokenizer = None

    def init(self, state: State, logger: Logger):
        del logger  # unused

        if not hasattr(state.model, 'tokenizer'):
            raise ValueError('state.model must have a tokenizer attribute.')
        self._tokenizer = state.model.tokenizer

    def before_load(self, state: State, logger: Logger):
        del logger  # unused

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
        duration = self._schedule[self._schedule_index]['duration']
        if ((
            duration.unit == TimeUnit.TOKEN and
            state.timestamp.token_in_iteration >= duration.value
        ) or (
            duration.unit == TimeUnit.EPOCH and
            state.timestamp.epoch_in_iteration >= duration.value
        )):
            log.warning((
                'The CurriculumLearning callback has detected that the '
                'previous run did not correctly increment the iteration.'
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
            train_loader_config = copy.deepcopy(self._train_loader_config)
            train_loader_config['dataset'].update(datamix['dataset'])
            data_spec = self._build_train_loader(
                train_loader_config=train_loader_config,
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
            'state':
                CurriculumLearningState(
                    schedule=self._schedule,
                    schedule_index=self._schedule_index,
                ),
        }

    def load_state_dict(self, state: dict[str, Any]):
        schedule = state['state'].schedule
        self._schedule_index = state['state'].schedule_index

        # Ensure that the schedule has not changed on previously trained datamixes
        for idx in range(self._schedule_index):
            if self._schedule[idx] != schedule[idx]:
                raise ValueError((
                    f'Previous datamixes must stay the same across ',
                    f'resumptions. Expected {schedule[idx]} but got ',
                    f'{self._schedule[idx]}',
                ))

        # Ensure that the datamix has not changed on the current datamix
        current_dataset = self._schedule[self._schedule_index]['dataset']
        saved_dataset = schedule[self._schedule_index]['dataset']
        if current_dataset != saved_dataset:
            raise ValueError((
                f'The current datamix must stay the same across resumptions. ',
                f'Expected {saved_dataset} but got {current_dataset}',
            ))

        # Ensure that the current datamix duration is in the correct units
        duration = self._schedule[self._schedule_index]['duration']
        if duration.unit != TimeUnit.TOKEN and duration.unit != TimeUnit.EPOCH:
            raise ValueError((
                f'Duration must be in terms of tokens or epochs, but got ',
                f'{duration.unit}.',
            ))

    def _build_train_loader(
        self,
        train_loader_config: dict[str, Any],
        logger: Logger,
    ) -> DataSpec:
        del logger  # unused

        from llmfoundry.data.dataloader import build_dataloader

        # Copied from scripts/train/train.py
        log.info(
            f'Building train loader in CurriculumLearning callback for dataset {self._schedule_index}',
        )
        assert self._tokenizer is not None
        try:
            return build_dataloader(
                train_loader_config,
                self._tokenizer,
                self._device_train_batch_size,
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

    def _validate_datamix(self, datamix: dict[str, Any]):
        if 'duration' not in datamix:
            raise ValueError('Each datamix must have a duration.')
        datamix['duration'] = ensure_time(
            datamix['duration'],
            TimeUnit.EPOCH,
        )
        if datamix['duration'].value <= 0:
            raise ValueError('The duration must be positive.')
        if (
            datamix['duration'].unit != TimeUnit.EPOCH and
            datamix['duration'].unit != TimeUnit.TOKEN
        ):
            raise ValueError(
                'Schedules can only be defined in terms of epochs or tokens.',
            )

        if 'dataset' not in datamix:
            raise ValueError('Each datamix must have a dataset.')
