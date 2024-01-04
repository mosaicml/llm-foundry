# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Run the eval loop asynchronously as part of a MosaicML platform run.

This callback is currently experimental. The API may change in the future.
"""

import logging
import os
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from composer.callbacks import CheckpointSaver
from composer.core import Callback, Event, State, Time, TimeUnit
from composer.loggers import Logger
from composer.loggers.mosaicml_logger import (MOSAICML_PLATFORM_ENV_VAR,
                                              RUN_NAME_ENV_VAR)
from composer.utils import dist
from composer.utils.file_helpers import list_remote_objects
from composer.utils.misc import create_interval_scheduler

from mcli import ComputeConfig, Run, RunConfig, create_run, get_run

log = logging.getLogger(__name__)

REQUIRED_PARAMS_FOR_EVAL = {
    'device_eval_batch_size',
    'icl_tasks',  # only required for eval, may not be specified in pure training
    'max_seq_len',
    'model',  # converted into models
    'tokenizer',  # converted into models
}
OPTIONAL_PARAMS_FOR_EVAL = {
    'dist_timeout',
    'eval_gauntlet',
    'fsdp_config',
    'icl_subset_num_batches',
    'loggers',
    'precision',
    'python_log_level',
    'seed',
}

RUN_NAME_PREFIX = 'eval'
MAX_RUN_NAME_BASE_LENGTH = 55


def get_run_name(training_run_name: str, current_interval: str) -> str:
    """Get the new eval run name.

    Args:
        training_run_name: The name of the current training run
        current_interval: The current interval string of the training run

    Returns:
        The new run name
    """
    name_without_uuid_suffix = training_run_name.rsplit('-', 1)[0]

    max_length = MAX_RUN_NAME_BASE_LENGTH - len(RUN_NAME_PREFIX) - len(
        current_interval) - 2

    # A run name that is too long will fail a createRun call
    if len(name_without_uuid_suffix) > max_length:
        new_name = name_without_uuid_suffix[:max_length]
        log.warning(
            f'Training run name {name_without_uuid_suffix} may be too long,' +
            f' truncating to {new_name}')
        name_without_uuid_suffix = new_name

    return f'{RUN_NAME_PREFIX}-{current_interval}-{name_without_uuid_suffix}'


SUPPORTED_UNITS = {TimeUnit.EPOCH, TimeUnit.BATCH}


def get_interval_from_checkpoint(checkpoint: str, unit: TimeUnit) -> Time:
    """Get the interval from a checkpoint name.

    Args:
        checkpoint: The name of the checkpoint
        unit: The unit of the interval

    Returns:
        The interval time
    """
    if unit == TimeUnit.EPOCH:
        val = checkpoint.split('-')[0].replace('ep', '')
    elif unit == TimeUnit.BATCH:
        val = checkpoint.split('-')[1].replace('ba', '')
    else:
        raise ValueError(
            f'Unsupported unit {unit}. Must be in {SUPPORTED_UNITS}')

    return Time(int(val), unit)


def get_eval_parameters(
    parameters: Dict[str, Any],
    checkpoint: str,
    training_run_name: str,
) -> Dict[str, Any]:
    """Get the parameters needed for the eval run.

    Args:
        parameters: The parameters from the training run
        checkpoint: The path to the latest checkpoint
        training_run_name: The name of the training run

    Returns:
        The parameters needed for the eval run as a dict
    """
    looking_for = REQUIRED_PARAMS_FOR_EVAL.copy()

    # Go through all parameters and pull out the ones needed for eval
    subset_keys = {}
    for key in parameters:
        if key in OPTIONAL_PARAMS_FOR_EVAL:
            subset_keys[key] = parameters[key]
        elif key in REQUIRED_PARAMS_FOR_EVAL:
            subset_keys[key] = parameters[key]
            looking_for.remove(key)

    if looking_for:
        raise Exception(
            f'Missing the following required parameters for async eval: {looking_for}'
        )

    for logger, config in subset_keys.get('loggers', {}).items():
        if logger == 'wandb':
            config['group'] = config.pop('name', training_run_name)

        # mlflow currently does not support grouping, so this will just launch
        # a new mlflow run

    # Create new eval models list
    model = subset_keys.pop('model')

    model_name = model.get('name', None)
    if not model_name:
        raise Exception(f'Async evaluation requires "name" keys for models')
    new_models = {
        'model_name': model_name,
        'model': model,
        'load_path': checkpoint
    }

    tokenizer = subset_keys.pop('tokenizer', None)
    if tokenizer is not None:
        new_models['tokenizer'] = tokenizer
    subset_keys['models'] = [new_models]
    return subset_keys


def validate_interval(interval: Union[str, int, Time],
                      save_interval: Union[str, int, Time]) -> Time:
    if isinstance(save_interval, str):
        new_save_interval: Time = Time.from_timestring(save_interval)
    elif isinstance(save_interval, int):
        new_save_interval: Time = Time(save_interval, TimeUnit.EPOCH)
    else:
        new_save_interval: Time = save_interval

    if isinstance(interval, str):
        result: Time = Time.from_timestring(interval)
    elif isinstance(interval, int):
        result: Time = Time(interval, TimeUnit.EPOCH)
    else:
        result: Time = interval

    if result.unit not in SUPPORTED_UNITS:
        raise ValueError(
            f'Async eval interval must be in units {SUPPORTED_UNITS}')

    if new_save_interval.unit != result.unit:
        raise ValueError(
            'Save interval and async eval interval must be in the same unit')
    if result < new_save_interval:
        raise ValueError(
            'Async eval interval must be equal or greater (less frequent) than save interval'
        )
    if result.value % new_save_interval.value != 0:
        raise ValueError(
            'Async eval interval must be a multiple of save interval')
    return result


class AsyncEval(Callback):
    """Run the eval loop asynchronously as part of a MosaicML platform run.

    This callback is currently experimental. The API may change in the future.

    Args:
        training_config: Dict[str, Any]: The config from the training run
        interval: Union[str, int, Time]: The interval describing how often eval runs should be
            launched. If an integer, it will be assumed to be in :attr:`.TimeUnit.EPOCH`.
            Otherwise, the unit must be either :attr:`.TimeUnit.EPOCH` or :attr:`.TimeUnit.BATCH`.
        compute: Optional[Union[ComputeConfig, Dict[str, Any]]]: The compute configuration to
            use for the eval run. If not provided, the same cluster as the current run and a
            single, full GPU node will be used.
    """

    def __init__(
        self,
        training_config: Dict[str, Any],
        interval: Union[str, int, Time],
        compute: Optional[Union[ComputeConfig, Dict[str, Any]]] = None,
    ):

        self.compute = compute

        # Run these during init to fail fast in any of the error cases
        for required in ('save_interval', 'save_folder'):
            if required not in training_config:
                raise ValueError(f'{required} required for async eval')
        self.checkpoint_save_folder = training_config['save_folder']
        self.training_config = training_config

        self.current_run = self._get_current_run()
        get_eval_parameters(
            parameters=training_config,
            checkpoint='test',
            training_run_name=self.current_run.name,
        )

        # Validate the interval (how often to launch eval runs)
        self.interval = validate_interval(interval,
                                          self.training_config['save_interval'])

        # Configures how often to check for new checkpoints
        self.check_interval = Time(max(self.interval.value // 5, 1),
                                   self.interval.unit)

        # Keep track of checkpoints by interval that have already been evaled
        # Format: {interval: (checkpoint, run_name)}
        self.checkpoints_evaled: Dict[Time, Tuple[str, str]] = {}

        # Scheduling is based on the check interval, while _get_checkpoints_and_launch_runs
        # will only launch runs at the interval
        self.is_at_check_interval = create_interval_scheduler(
            self.check_interval,
            # There is a custom close to ensure that the final checkpoint
            # (which is the most important) is evaled after it is written
            include_end_of_training=False,
        )

        log.info('Initialized AsyncEval callback. Will generate runs at ' +
                 f'interval {interval}, checking at {self.check_interval}')

    def _get_ready_sharded_checkpoints(
        self,
        checkpointer_checkpoints: List[str],
        remote_files: List[str],
    ):
        """
        Determine which checkpoints from the checkpointer are ready to be evaled,
        based on which shards have been uploaded to the remote checkpoint folder.

        This has special logic for sharded checkpoints to consider checkpoints composed
        of multiple shards (one per gpu) and metadata

        Args:
            checkpointer_checkpoints: The checkpoints from the checkpointer state
            remote_files: List of remote files in the save folder
        
        Returns:
            List of checkpoints that are complete and ready to be evaled
        """

        # Count the number of shards for each checkpoint group
        remote_file_group_counts = Counter()
        for f in remote_files:
            interval_path = Path(f).parts[-2]
            remote_file_group_counts[interval_path] += 1

        # Check if all shards are present for each checkpoint group
        checkpoints_to_eval = []
        for checkpoint in checkpointer_checkpoints:
            # eg {save_folder}/ep0-ba1/.
            interval_path = Path(checkpoint).parts[-2]

            # expecting one shard per gpu + 1 for metadata
            expected_shard_count = dist.get_world_size() + 1
            if remote_file_group_counts[interval_path] != expected_shard_count:
                log.debug(
                    f'Checkpoint {checkpoint} not fully uploaded (missing shards '
                    +
                    f'{remote_file_group_counts[interval_path]}/{expected_shard_count}), skipping'
                )
                continue

            checkpoints_to_eval.append(interval_path)

        return checkpoints_to_eval

    def _get_ready_single_checkpoints(
        self,
        checkpointer_checkpoints: List[str],
        remote_checkpoints: List[str],
    ):
        """
        Determine which checkpoints from the checkpointer are ready to be evaled,
        based on which checkpoints have been uploaded

        This is much simpler than the sharded case, because there is only one file

        Args:
            checkpointer_checkpoints: The checkpoints from the checkpointer state
            remote_checkpoints: List of remote checkpoints in the save folder
        
        Returns:
            List of checkpoints that are complete and ready to be evaled
        """
        unique_remote_checkpoints = set(remote_checkpoints)

        checkpoints_to_eval = []
        for checkpoint in checkpointer_checkpoints:
            # eg {save_folder}/ep0-ba1-rank0.pt
            interval_path = Path(checkpoint).parts[-1]

            if checkpoint not in unique_remote_checkpoints:
                log.debug(
                    f'Checkpoint {checkpoint} not fully uploaded, skipping')
                continue

            checkpoints_to_eval.append(interval_path)
        return checkpoints_to_eval

    def _get_checkpoints_and_launch_runs(self, state: State):
        """Get the latest checkpoint from the training run.

        Args:
            state: The current state of the training run

        Returns:
            Returns checkpoints that have not been evaled
        """
        checkpointer = None
        for callback in state.callbacks:
            if isinstance(callback, CheckpointSaver):
                checkpointer = callback
                break

        if not checkpointer:
            warnings.warn('No checkpoint saver callback found. Skipping eval')
            return

        if not checkpointer.saved_checkpoints:
            log.debug(
                'No saved checkpoints found on the checkpointer. Skipping eval')
            return

        remote_checkpoints = list_remote_objects(self.checkpoint_save_folder)

        if not remote_checkpoints:
            log.debug('No saved checkpoints found yet on remote. Skipping eval')
            return

        if state.fsdp_elastic_sharded_enabled:
            checkpoints_to_eval = self._get_ready_sharded_checkpoints(
                checkpointer.saved_checkpoints, remote_checkpoints)
        else:
            checkpoints_to_eval = self._get_ready_single_checkpoints(
                checkpointer.saved_checkpoints, remote_checkpoints)

        for checkpoint_interval_path in checkpoints_to_eval:
            interval = get_interval_from_checkpoint(checkpoint_interval_path,
                                                    self.interval.unit)
            if interval.value % self.interval.value != 0:
                log.debug(
                    f'Checkpoint {checkpoint_interval_path} ({interval}) is not at an eval interval ({self.interval}), skipping'
                )
                continue
            if interval in self.checkpoints_evaled:
                continue  # Skip checkpoints that have already been evaled

            full_checkpoint_path = f'{self.checkpoint_save_folder}/{checkpoint_interval_path}'
            eval_run = self.launch_run(full_checkpoint_path, interval)
            self.checkpoints_evaled[interval] = (full_checkpoint_path,
                                                 eval_run.name)

    def run_event(self, event: Event, state: State, logger: Logger) -> None:
        del logger

        should_launch_run = all([
            state.get_elapsed_duration() is not None,
            # could also skip check intervals before the first async eval interval,
            # but this may make the scheduler more complicated
            self.is_at_check_interval(state, event),
            dist.get_global_rank() == 0,
        ])

        if should_launch_run:
            self._get_checkpoints_and_launch_runs(state)

    def close(self, state: State, logger: Logger) -> None:
        del logger

        if dist.get_global_rank() != 0:
            return

        # Eval any remaining checkpoints
        self._get_checkpoints_and_launch_runs(state)

        # Eval the latest checkpoint
        latest_timestamp = state.timestamp.get(self.interval.unit)
        if latest_timestamp not in self.checkpoints_evaled:
            save_latest_filename = self.training_config.get(
                'save_latest_filename', None)

            if not save_latest_filename:
                rank = dist.get_global_rank()
                save_latest_filename = f'latest-rank{rank}.pt'

            checkpoint = f'{self.checkpoint_save_folder}/{save_latest_filename}'

            eval_run = self.launch_run(checkpoint, latest_timestamp)
            self.checkpoints_evaled[latest_timestamp] = (checkpoint,
                                                         eval_run.name)

        log.info(
            f'AsyncEval callback finished. Launched {len(self.checkpoints_evaled)} eval runs:'
        )
        for interval, (checkpoint, run_name) in self.checkpoints_evaled.items():
            log.info(f'  {interval}: {checkpoint}, {run_name}')

    def _get_current_run(self) -> Run:
        if os.environ.get(MOSAICML_PLATFORM_ENV_VAR,
                          'false').lower() == 'false':
            raise RuntimeError(
                'AsyncEval callback is only supported when running on the MosaicML platform'
            )

        run_name = os.environ.get(RUN_NAME_ENV_VAR, None)
        if not run_name:
            raise RuntimeError(
                'RUN_NAME environment variable must be set to use the AsyncEval callback'
            )

        # Allows the MapiException to be raised if the run doesn't exist
        return get_run(run_name, include_details=True)

    def launch_run(self, checkpoint: str, current_interval: Time) -> Run:
        log.info(f'Launching eval run for {checkpoint} at {current_interval}')

        cfg = self.current_run.submitted_config
        default_compute = {
            'gpus': 8,
            'cluster': self.current_run.cluster,
        }

        run_name = get_run_name(self.current_run.name, str(current_interval))

        params = get_eval_parameters(
            parameters=self.training_config,
            checkpoint=checkpoint,
            training_run_name=self.current_run.name,
        )
        params['run_name'] = run_name

        integrations = cfg.integrations
        found_llm_foundry, installation_path = False, 'llm-foundry'
        for i in integrations:
            if i['integration_type'] != 'git_repo':
                continue

            if not i['git_repo'].endswith('llm-foundry'):  # detects forks
                continue

            found_llm_foundry = True
            if i.get('path'):
                installation_path = i['path']

        if not found_llm_foundry:
            from llmfoundry import __version__ as latest_foundry_version

            # If github integration is not found, foundry is likely installed
            # through the run command. In this case, we'll add the integration
            # so the eval run will still work. However, it could cause unexpected
            # behaviors because its not using custom repos or branches specified
            # in the training run. For this reason, we'll log a warning
            version = f'v{latest_foundry_version}'
            log.warning(
                'No github integration found for llm-foundry. Adding installation '
                + f'to eval run for latest foundry release ({version}). ' +
                'To use a fork, custom branch, or custom version, configure ' +
                'llm-foundry installation through a github integration')
            integrations.append({
                'integration_type': 'git_repo',
                'git_repo': 'mosaicml/llm-foundry',
                'git_branch': version,
                'pip_install': '-e .[gpu]',
                'ssh_clone': False,
            })

        # This will record the timestamp and make it available for grouping
        # and plotting in wandb
        metadata = cfg.metadata
        metadata['eval_timestamp'] = current_interval.value
        metadata['eval_timestamp_unit'] = current_interval.unit.value

        # TODO: This just runs an eval run, but we also want to attach the
        # deployment, which would require a hf conversion and parametrizing the
        # dependent_deployment in the run config
        command = f'cd {installation_path}/scripts \n composer eval/eval.py $PARAMETERS'
        run_config = RunConfig(
            name=run_name,
            image=self.current_run.image,
            compute=self.compute or default_compute,
            command=command,
            integrations=integrations,
            env_variables=cfg.env_variables,
            metadata=cfg.metadata,
            parameters=params,
        )

        log.info(f'Creating new run with config: \n{run_config}')
        new_run = create_run(run_config)
        log.info(f'Launched new run {new_run.name} inside eval loop')
        return new_run
