# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from typing import Any, Dict, List, Optional, Union

from composer.core import Callback, Event, State, Time
from composer.loggers import Logger
from composer.loggers.mosaicml_logger import (MOSAICML_PLATFORM_ENV_VAR,
                                              RUN_NAME_ENV_VAR)
from composer.utils import dist
from composer.utils.misc import create_interval_scheduler

from mcli import ComputeConfig, Run, RunConfig, create_run, get_run

log = logging.getLogger(__name__)

MAX_RUN_NAME_LENGTH = 40

REQUIRED_PARAMS_FOR_EVAL = {
    'device_eval_batch_size',
    'icl_tasks',  # only required for eval, may not be specified in pure training
    'max_seq_len',
    'model',  # converted into models
    'tokenizer',  # converted into models
    'save_folder',  # required, but used as load_path
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


def get_run_name(previous_run_name: str, count: int) -> str:
    *name_without_uuid_suffix, _ = previous_run_name.split('-')
    name_suffix = ('-'.join(name_without_uuid_suffix))

    if len(name_suffix) > MAX_RUN_NAME_LENGTH:
        log.warning(
            f'Training run name {name_suffix} may be too long, truncating to {MAX_RUN_NAME_LENGTH} characters'
        )
        name_suffix = name_suffix[:MAX_RUN_NAME_LENGTH]

    return f'eval{count}-{name_suffix}'


def get_load_path(save_folder: str,
                  save_latest_filename: Optional[str] = None) -> str:
    # TODO: check that the prefix is remote and not a local file (not supported of course)

    if save_latest_filename is None:
        rank = dist.get_global_rank()
        save_latest_filename = f'latest-rank{rank}.pt'

    return f'{save_folder}/{save_latest_filename}'


def get_eval_models_dict(
    model: Dict[str, Any],
    tokenizer: Dict[str, Any],
) -> List[Dict[str, Any]]:
    name = model.get('name')

    cfg_overrides = model.pop('cfg_overrides', {})
    for key in cfg_overrides:
        model[key] = cfg_overrides[key]

    new_model = {'model_name': name, 'model': model}

    if tokenizer:
        new_model['tokenizer'] = tokenizer

    return [new_model]


class AsyncEval(Callback):
    """Run the eval loop asynchronously as part of a MosaicML platform run.

    Args:
        interval: Union[str, int, Time]: The interval describing how often eval runs should be
            launched. If an integer, it will be assumed to be in :attr:`.TimeUnit.EPOCH`.
            Otherwise, the unit must be either :attr:`.TimeUnit.EPOCH`, :attr:`.TimeUnit.BATCH`,
            :attr:`.TimeUnit.TOKEN`, or :attr:`.TimeUnit.SAMPLE`.
        compute: Optional[Union[ComputeConfig, Dict[str, Any]]]: The compute configuration to
            use for the eval run. If not provided, the same cluster as the current run and a
            single, full GPU node will be used.
    """

    def __init__(
        self,
        interval: Union[str, int, Time],
        compute: Optional[Union[ComputeConfig, Dict[str, Any]]] = None,
    ):
        self.check_interval = create_interval_scheduler(interval)
        self.compute = compute
        self.count = 0
        self.last_launch: Optional[Time] = None

        # Run these during init to fail fast in any of the error cases
        self.current_run = self._get_current_run()
        self.get_eval_parameters(
            self.current_run.submitted_config.parameters or {},
            self.current_run.name,
        )
        log.info(
            f'Initialized AsyncEval callback. Will generate runs at interval {interval}'
        )

    def run_event(self, event: Event, state: State, logger: Logger) -> None:
        del logger
        if all([
                state.get_elapsed_duration() is not None,
                self.check_interval(state, event),
                self.last_launch != state.timestamp.batch,
                dist.get_global_rank() == 0
        ]):
            self.launch_run()

            self.last_launch = state.timestamp.batch
            self.count += 1

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

    def get_eval_parameters(
        self,
        parameters: Dict[str, Any],
        run_name: str,
    ) -> Dict[str, Any]:
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

        # Convert the save_folder to a load_path
        subset_keys['load_path'] = get_load_path(
            subset_keys.pop('save_folder'),
            parameters.get('save_latest_filename', None))

        # Update the loggers to use the training run name
        for logger, config in subset_keys.get('loggers', {}).items():
            if logger == 'wandb':
                config['name'] = config.get('name', run_name)
            elif logger == 'mlflow':
                config['run_name'] = config.get('run_name', run_name)

        # Create new eval models list
        subset_keys['models'] = get_eval_models_dict(
            subset_keys.pop('model'), subset_keys.pop('tokenizer'))

        subset_keys['run_name'] = get_run_name(run_name, 0)
        return subset_keys

    def launch_run(self) -> Run:
        cfg = self.current_run.submitted_config
        default_compute = {
            'gpus': 8,
            'cluster': self.current_run.cluster,
        }
        params = self.get_eval_parameters(cfg.parameters or {},
                                          self.current_run.name)

        # TODO: This just runs an eval run, but we also want to attach the
        # deployment, which would require a hf conversion and parametrizing the
        # dependent_deployment in the run config
        command = 'cd llm-foundry/scripts \n composer eval/eval.py $PARAMETERS'
        run_config = RunConfig(
            name=get_run_name(self.current_run.name, self.count),
            image=self.current_run.image,
            compute=self.compute or default_compute,
            command=command,
            integrations=cfg.integrations,
            env_variables=cfg.env_variables,
            metadata=cfg.metadata,
            parameters=params,
        )

        # Increase default timeout of 10s just in case
        new_run = create_run(run_config, timeout=20)
        log.info(
            f'Launched new run {new_run.name} inside eval loop with config: \n{new_run.submitted_config}'
        )
        return new_run
