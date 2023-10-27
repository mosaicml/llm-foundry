# Copyright 2023 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
import logging
import os
from typing import Any, Dict, Optional, Union

from composer.core import Callback, Event, State, Time
from composer.loggers import Logger
from composer.loggers.mosaicml_logger import (MOSAICML_PLATFORM_ENV_VAR,
                                              RUN_NAME_ENV_VAR)
from composer.utils import create_interval_scheduler, dist
from mcli.api.runs import ComputeConfig  # TODO: should be available in root

from mcli import Run, RunConfig, create_run, get_run

log = logging.getLogger(__name__)

MAX_RUN_NAME_LENGTH = 40

# Note: train parameter names. See comments if they are different from eval
REQUIRED_PARAMS_FOR_EVAL = {
    'device_eval_batch_size',
    'icl_tasks',  # only required for eval
    'max_seq_len',
    'model',  # models
    'save_folder',  # required, but used as load_path
}
OPTIONAL_PARAMS_FOR_EVAL = {
    'dist_timeout',
    'eval_gauntlet',
    'fsdp_config',  # fsdp_dict_cfg
    'icl_subset_num_batches',
    'loggers',
    'precision',
    'python_log_level',
    'seed',
}


def get_run_name(previous_run_name: str, count: int) -> str:
    return f'eval{count}-{previous_run_name[:MAX_RUN_NAME_LENGTH]}'


def get_load_path(save_folder: str,
                  save_latest_filename: Optional[str] = None) -> str:
    # TODO: check that the prefix is remote and not a local file (not supported of course)

    if not save_latest_filename:
        rank = dist.get_global_rank()
        save_latest_filename = f'latest-rank{rank}.pt'

    return f'{save_folder}/{save_latest_filename}'


class AsyncEval(Callback):
    """Run the eval loop asynchronously as part of a MosaicML platform run

    Args:
        interval: Union[str, int, Time]: The interval describing how often eval runs should be
            launched. If an integer, it will be assumed to be in :attr:`.TimeUnit.EPOCH`.
            Otherwise, the unit must be either :attr:`.TimeUnit.EPOCH`, :attr:`.TimeUnit.BATCH`,
            :attr:`.TimeUnit.TOKEN`, or :attr:`.TimeUnit.SAMPLE`.
    """

    def __init__(
        self,
        interval: Union[str, int, Time],
        compute: Optional[ComputeConfig] = None,
    ):
        self.check_interval = create_interval_scheduler(interval)
        self.compute = compute
        self.count = 0

        # Run these during init to fail fast in any of the error cases
        self.current_run = self._get_current_run()
        self._get_eval_parameters()

    def run_event(self, event: Event, state: State, logger: Logger) -> None:
        del logger
        if state.get_elapsed_duration() is not None and self.check_interval(
                state, event):
            new_run = self._launch_run()
            logger.info(f'Launched new run {new_run.name} for eval loop')
            self.count += 1

    def _get_current_run(self) -> Run:
        if os.environ.get(MOSAICML_PLATFORM_ENV_VAR,
                          'false').lower() == 'false':
            raise Exception(
                'AsyncEval callback is only supported when running on the MosaicML platform'
            )

        run_name = os.environ.get(RUN_NAME_ENV_VAR, None)
        if not run_name:
            raise Exception(
                'RUN_NAME environment variable must be set to use the AsyncEval callback'
            )

        # allows the MapiException to be raised if the run doesn't exist
        return get_run(run_name, include_details=True)

    def _get_eval_parameters(self) -> Dict[str, Any]:
        cfg_params = self.current_run.submitted_config.parameters or {}
        looking_for = REQUIRED_PARAMS_FOR_EVAL.copy()

        # Go through all parameters and pull out the ones needed for eval
        subset_keys = {}
        for key in cfg_params:
            if key in OPTIONAL_PARAMS_FOR_EVAL:
                subset_keys[key] = cfg_params[key]
            elif key in REQUIRED_PARAMS_FOR_EVAL:
                subset_keys[key] = cfg_params[key]
                looking_for.remove(key)

        if looking_for:
            raise Exception(
                f'Missing the following required parameters for async eval: {looking_for}'
            )

        # Convert the save_folder to a load_path
        subset_keys['load_path'] = get_load_path(
            subset_keys.pop('save_folder'),
            cfg_params.get('save_latest_filename', None))

        # Rename the keys to match the eval script
        subset_keys['models'] = [cfg_params.pop('model')]
        if 'fsdp_cfg' in subset_keys:
            subset_keys['fsdp_dict_cfg'] = cfg_params.pop('fsdp_cfg')

        cfg_params['run_name'] = get_run_name(self.current_run.name, self.count)
        return cfg_params

    def _launch_run(self) -> Run:
        cfg = self.current_run.submitted_config
        default_compute = {
            'nodes': 1,
            'cluster': self.current_run.cluster,
        }
        params = self._get_eval_parameters()

        # TODO: This just runs an eval run, but we also want to attach the
        # deployment, which would require a hf conversion and parametrizing the
        # dependent_deployment in the run config
        command = 'cd llm-foundry/scripts \n composer eval/eval.py $PARAMETERS'
        c = RunConfig(
            name=get_run_name(self.current_run.name, self.count),
            image=self.current_run.image,
            compute=self.compute or default_compute,
            command=command,
            integrations=cfg.integrations,
            env_variables=cfg.env_variables,
            metadata=cfg.metadata,
            parameters=params,
        )

        return create_run(c)
