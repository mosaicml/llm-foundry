# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Run the eval loop asynchronously as part of a MosaicML platform run.

This callback is currently experimental. The API may change in the future.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union, cast

from composer.callbacks import CheckpointSaver
from composer.core import Callback, Event, State, Time, TimeUnit
from composer.loggers import Logger
from composer.loggers.mosaicml_logger import (MOSAICML_PLATFORM_ENV_VAR,
                                              RUN_NAME_ENV_VAR)
from composer.utils import dist
from composer.utils.misc import create_interval_scheduler
from omegaconf import DictConfig
from omegaconf import OmegaConf as om

from mcli import Run, RunConfig, create_run, get_run

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
    'eval_loader',
    'fsdp_config',
    'eval_subset_num_batches',  # converted to subset_num_batches
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


def get_latest_checkpoint(event: Event, state: State) -> Optional[str]:
    """Get the latest checkpoint from the training run.

    Args:
        event: The current run event
        state: The current state of the training run

    Returns:
        The path to the latest checkpoint, or None if there is not a latest checkpoint
    """
    checkpointer = None
    for callback in state.callbacks:
        if isinstance(callback, CheckpointSaver):
            checkpointer = callback
            break

    if not checkpointer:
        log.warning('No checkpoint saver callback found')
        return None

    if not checkpointer.saved_checkpoints:
        log.warning('No saved checkpoints found on the checkpointer')
        return None

    latest = checkpointer.saved_checkpoints[-1]
    return str(Path(latest).parts[-1])


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

    if 'eval_subset_num_batches' in subset_keys:
        subset_keys['subset_num_batches'] = subset_keys.pop(
            'eval_subset_num_batches')

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


def validate_eval_run_config(
        eval_run_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:

    if isinstance(eval_run_config, DictConfig):
        parsed_run_config = om.to_container(eval_run_config)
        run_config = cast(Dict[str, Any], parsed_run_config)
    elif eval_run_config is None:
        return {}
    else:
        run_config = eval_run_config.copy()

    if not run_config:
        return {}

    supported_keys = {'image', 'command', 'compute', 'scheduling'}
    for key in run_config:
        if key not in supported_keys:
            raise ValueError(
                f'Unsupported eval run config key {key}. Supported keys: {supported_keys}'
            )

    for dict_field in ('scheduling', 'compute'):
        if dict_field in run_config and not isinstance(run_config[dict_field],
                                                       dict):
            raise TypeError(
                f'Eval run {dict_field} must be a dict. ' +
                f'Got {run_config[dict_field]} ({type(run_config[dict_field])})'
            )
        for value in run_config.get(dict_field, {}).values():

            if not (isinstance(value, int) or isinstance(value, str)):
                raise TypeError(
                    f'Eval run {dict_field} values must be integers or ' +
                    f'strings. Got {value} ({type(value)})')

    for str_field in ('image', 'command'):
        if str_field in run_config and not isinstance(run_config[str_field],
                                                      str):
            raise TypeError(
                f'Eval run {str_field} must be a string. Got ' +
                f'{run_config[str_field]} ({type(run_config[str_field])})')

    return run_config


class AsyncEval(Callback):
    """Run the eval loop asynchronously as part of a MosaicML platform run.

    This callback is currently experimental. The API may change in the future.

    Args:
        training_params: Dict[str, Any]: The parameter config from the training run
        interval: Union[str, int, Time]: The interval describing how often eval runs should be
            launched. If an integer, it will be assumed to be in :attr:`.TimeUnit.EPOCH`.
            Otherwise, the unit must be either :attr:`.TimeUnit.EPOCH`, :attr:`.TimeUnit.BATCH`,
            :attr:`.TimeUnit.TOKEN`, or :attr:`.TimeUnit.SAMPLE`.
        eval_run_config: Optional[Dict[str, Any]]: A subset of mcli run config values to use
            for the eval run. If not specified, any fields from run config will be created
            dynamically from the training run config and the interval. The following fields
            are supported:
            - ``image``: Image of the eval run. Default: same as training run
            - ``command``: Command to run for the eval run. Default: calls
                `composer scripts/eval/eval.py $PARAMETERS`. If custom setup is needed,
                the command should include calling the eval script with $PARAMETERS
            - ``compute``: Compute to use for the eval run. Default: same cluster as
                the training run and a single node (8 GPUs)
            - ``scheduling``: Scheduling to use for the eval run. Default: same as training run

            All fields are optional, but if specified, must be valid for a mcli run config. We
            provide this optional config to give you the most flexibility in customizing the eval
            run, but it is recommended to use the default values unless you have a specific use case
    """

    def __init__(
        self,
        training_params: Dict[str, Any],
        interval: Union[str, int, Time],
        eval_run_config: Optional[Dict[str, Any]] = None,
    ):

        for required in ('save_interval', 'save_folder'):
            if required not in training_params:
                raise ValueError(f'{required} required for async eval')

        self.checkpoint_save_folder = training_params['save_folder']
        self.training_params = training_params
        self.eval_run_config = validate_eval_run_config(eval_run_config)
        self.interval = validate_interval(interval,
                                          self.training_params['save_interval'])
        self.check_interval = create_interval_scheduler(
            interval,
            # There is a custom close to ensure that the final checkpoint
            # (which is the most important) is evaled after it is written
            include_end_of_training=False,
        )
        self.last_checkpoint: Optional[str] = None

        # Run these during init to fail fast in any of the error cases
        self.current_run = self._get_current_run()
        get_eval_parameters(
            parameters=training_params,
            checkpoint='test',
            training_run_name=self.current_run.name,
        )
        log.info(
            f'Initialized AsyncEval callback. Will generate runs at interval {interval}'
        )

        # TODO: potentially support eval_first

    def run_event(self, event: Event, state: State, logger: Logger) -> None:
        del logger

        should_launch_run = all([
            state.get_elapsed_duration() is not None,
            self.check_interval(state, event),
            dist.get_global_rank() == 0,
        ])

        if should_launch_run:
            current_interval = state.timestamp.get(self.interval.unit)
            checkpoint = get_latest_checkpoint(event, state)
            if not checkpoint:
                return  # warnings logged in get_latest_checkpoint

            # TODO: ensure the checkpoint is fully written before launching the eval run
            full_checkpoint = f'{self.checkpoint_save_folder}/{checkpoint}'
            if full_checkpoint == self.last_checkpoint:
                # Do not eval a checkpoint that has already been evaluated.
                log.info(
                    'Skipping async eval because the checkpoint has not changed'
                )
                return

            self.launch_run(full_checkpoint, current_interval)
            self.last_checkpoint = full_checkpoint

    def close(self, state: State, logger: Logger) -> None:
        del logger

        if dist.get_global_rank() != 0:
            return

        save_latest_filename = self.training_params.get('save_latest_filename',
                                                        None)

        if not save_latest_filename:
            rank = dist.get_global_rank()
            save_latest_filename = f'latest-rank{rank}.pt'

        checkpoint = f'{self.checkpoint_save_folder}/{save_latest_filename}'
        self.launch_run(checkpoint, state.timestamp.get(self.interval.unit))

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
            parameters=self.training_params,
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
        default_command = f'cd {installation_path}/scripts \n composer eval/eval.py $PARAMETERS'
        run_config = RunConfig(
            name=run_name,
            image=self.eval_run_config.get('image', self.current_run.image),
            command=self.eval_run_config.get('command', default_command),
            compute=self.eval_run_config.get('compute', default_compute),
            scheduling=self.eval_run_config.get(
                'scheduling',
                self.current_run.submitted_config.scheduling,
            ),
            integrations=integrations,
            env_variables=cfg.env_variables,
            metadata=cfg.metadata,
            parameters=params,
        )

        log.info(f'Creating new run with config: \n{run_config}')
        new_run = create_run(run_config)
        log.info(f'Launched new run {new_run.name} inside eval loop')
        return new_run
