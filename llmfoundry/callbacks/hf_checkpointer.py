# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import copy
import logging
import math
import os
import re
import shutil
import tempfile
import time
import warnings
from multiprocessing.context import SpawnProcess
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from composer.core import Callback, Event, Precision, State, Time, TimeUnit
from composer.devices import Device
from composer.loggers import Logger, MLFlowLogger
from composer.models import HuggingFaceModel
from composer.utils import (
    dist,
    format_name_with_dist_and_time,
    maybe_create_remote_uploader_downloader_from_uri,
    parse_uri,
)
from composer.utils.misc import create_interval_scheduler
from mlflow.transformers import _fetch_model_card, _write_license_information
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from llmfoundry.models.mpt import MPTConfig, MPTForCausalLM
from llmfoundry.models.utils import init_empty_weights
from llmfoundry.utils.exceptions import StoragePermissionError
from llmfoundry.utils.huggingface_hub_utils import \
    edit_files_for_hf_compatibility

try:
    import transformer_engine.pytorch as te
    is_te_imported = True
except ModuleNotFoundError:
    is_te_imported = False

log = logging.getLogger(__name__)

__all__ = ['HuggingFaceCheckpointer']

_LICENSE_FILE_PATTERN = re.compile(r'license(\.[a-z]+|$)', re.IGNORECASE)

from contextlib import contextmanager


@contextmanager
def _monitor_process_saver(mlflow_logger: MLFlowLogger):
    # Save the current monitor process
    if hasattr(mlflow_logger, 'monitor_process'):
        original_monitor_process = mlflow_logger.monitor_process  # type: ignore
        mlflow_logger.monitor_process = None  # type: ignore
    else:
        original_monitor_process = None

    try:
        # Yield control back to the calling code
        yield
    finally:
        # Restore the monitor process
        if original_monitor_process is not None:
            mlflow_logger.monitor_process = original_monitor_process  # type: ignore


def _maybe_get_license_filename(
    local_dir: str,
    pretrained_model_name: Optional[str] = None,
) -> Optional[str]:
    """Returns the name of the license file if it exists in the local_dir.

    Note: This is intended to be consistent with the code in MLflow.
    https://github.com/mlflow/mlflow/blob/5d13d6ec620a02de9a5e31201bf1becdb9722ea5/mlflow/transformers/__init__.py#L1152

    Since LLM Foundry supports local model files being used rather than fetching the files from the Hugging Face Hub,
    MLflow's logic to fetch and write the license information on model save is not applicable; it will try to search for
    a Hugging Face repo named after the local path. However, the user can provide the original pretrained model name,
    in which case this function will use that to fetch the correct license information.

    If the license file does not exist, returns None.
    """
    try:
        license_filename = next(
            file for file in os.listdir(local_dir)
            if _LICENSE_FILE_PATTERN.search(file)
        )

        # If a pretrained model name is provided, replace the license file with the correct info from HF Hub.
        if pretrained_model_name is not None:
            log.info(
                f'Overwriting license file {license_filename} with license info for model {pretrained_model_name} from Hugging Face Hub',
            )
            os.remove(os.path.join(local_dir, license_filename))
            model_card = _fetch_model_card(pretrained_model_name)

            local_dir_path = Path(local_dir).absolute()
            _write_license_information(
                pretrained_model_name,
                model_card,
                local_dir_path,
            )
            license_filename = next(
                file for file in os.listdir(local_dir)
                if _LICENSE_FILE_PATTERN.search(file)
            )

        return license_filename

    except StopIteration:
        return None


def _log_model_with_multi_process(
    mlflow_logger: MLFlowLogger,
    python_logging_level: int,
    transformers_model: Union[dict[str, Any], str],
    artifact_path: str,
    pretrained_model_name: str,
    registered_model_name: Optional[str],
    await_registration_for: int,
    mlflow_logging_config: dict[str, Any],
):
    """Call MLFlowLogger.log_model.

    First, patch the mlflow save_model function by removing duplicate tokenizer
    files in the model directory. Then, register the model to mlflow from a
    child process.
    """
    # Setup logging for child process. This ensures that any logs from composer are surfaced.
    if python_logging_level > 0:
        # If logging_level is 0, then the composer logger was unset.
        logging.basicConfig(
            format=
            f'%(asctime)s: rank{dist.get_global_rank()}[%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s',
            force=True,
        )
        logging.getLogger('composer').setLevel(python_logging_level)
        logging.getLogger('llmfoundry').setLevel(python_logging_level)

    import mlflow
    original_save_model = mlflow.transformers.save_model

    def save_model_patch(*args: Any, **kwargs: Any):
        original_save_model(*args, **kwargs)
        tokenizer_files = []
        save_path = kwargs['path']
        tokenizer_path = os.path.join(save_path, 'components', 'tokenizer')
        if os.path.exists(tokenizer_path):
            tokenizer_files = os.listdir(
                os.path.join(save_path, 'components', 'tokenizer'),
            )
        try:
            # Check if there are duplicate tokenizer files in the model directory and remove them.
            for tokenizer_file_name in tokenizer_files:
                dupe_file = os.path.isfile(
                    os.path.join(save_path, 'model', tokenizer_file_name),
                )
                if dupe_file:
                    log.debug(
                        f'Removing duplicate tokenizer file: {tokenizer_file_name}',
                    )
                    os.remove(
                        os.path.join(save_path, 'model', tokenizer_file_name),
                    )
            license_filename = _maybe_get_license_filename(
                save_path,
                pretrained_model_name,
            )
            if license_filename is not None:
                mlflow_logger._mlflow_client.log_artifact(
                    mlflow_logger._run_id,
                    os.path.join(save_path, license_filename),
                )
        except Exception as e:
            log.error(
                f'Exception when removing duplicate tokenizer files in the model directory',
                e,
            )

    mlflow.transformers.save_model = save_model_patch  # type: ignore

    mlflow.set_tracking_uri(mlflow_logger.tracking_uri)
    if mlflow_logger.model_registry_uri is not None:
        mlflow.set_registry_uri(mlflow_logger.model_registry_uri)

    register_model_path = f'{mlflow_logger.model_registry_prefix}.{registered_model_name}' if mlflow_logger.model_registry_prefix and registered_model_name else registered_model_name
    mlflow_logger.log_model(
        transformers_model=transformers_model,
        flavor='transformers',
        artifact_path=artifact_path,
        registered_model_name=register_model_path,
        run_id=mlflow_logger._run_id,
        await_registration_for=await_registration_for,
        **mlflow_logging_config,
    )


class HuggingFaceCheckpointer(Callback):
    """Save a huggingface formatted checkpoint during training.

    Args:
        save_folder (str): Top level folder to save checkpoints to (can be a
            URI). It is likely that this would be the same as your save_folder.
        save_interval: Union[str, int, Time]: The interval describing how often
            checkpoints should be saved. If an integer, it will be assumed to be
            in :attr:`.TimeUnit.EPOCH`. Otherwise, the unit must be either
            :attr:`.TimeUnit.EPOCH`, :attr:`.TimeUnit.BATCH`,
            :attr:`.TimeUnit.TOKEN`, or :attr:`.TimeUnit.SAMPLE`.
        huggingface_folder_name (str): Folder to save each checkpoint under (can
            be a format string). Default is ``ba{batch}``.
        precision: The precision to save the model in. Default is ``float32``.
            Options are ``bfloat16``, ``float16``, or ``float32``.
        overwrite (bool): Whether to overwrite previous checkpoints.
        mlflow_registered_model_name (Optional[str]): The name to register the
            model under in the MLflow model registry. If ``None``, the model
            will not be registered. Default is ``None``.
        mlflow_logging_config (Optional[dict]): A dictionary of config arguments
            that will get passed along to the MLflow ``save_model`` call.
            Expected to contain ``metadata`` and ``task`` keys. If either is
            unspecified, the defaults are ``'text-generation'`` and
            ``{'task': 'llm/v1/completions'}`` respectively. A default input example
            and signature intended for text generation is also included under the
            keys ``input_example`` and ``signature``.
        flatten_imports (Sequence[str]): A sequence of import prefixes that will
            be flattened when editing MPT files.
        final_register_only (bool): If true, only register the model in the MLFlow
            registry on the last batch and do not save the HuggingFace checkpoint. If
            registration fails or mlflow_registered_model_name is not set, then we will
            fallback to saving the HuggingFace checkpoint.
    """

    def __init__(
        self,
        save_folder: str,
        save_interval: Union[str, int, Time],
        huggingface_folder_name: str = 'ba{batch}',
        precision: str = 'float32',
        overwrite: bool = True,
        mlflow_registered_model_name: Optional[str] = None,
        mlflow_logging_config: Optional[dict] = None,
        flatten_imports: Sequence[str] = ('llmfoundry',),
        final_register_only: bool = False,
        register_wait_seconds: int = 7200,
    ):
        _, _, self.save_dir_format_str = parse_uri(save_folder)
        self.overwrite = overwrite
        self.precision = precision
        self.dtype = {
            'float32': torch.float32,
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
        }[precision]
        self.flatten_imports = flatten_imports
        self.using_peft = False

        self.final_register_only = final_register_only
        self.register_wait_seconds = register_wait_seconds

        self.mlflow_registered_model_name = mlflow_registered_model_name
        if self.final_register_only and self.mlflow_registered_model_name is None:
            self.final_register_only = False
            warnings.warn(
                'final_register_only is set to True, but mlflow_registered_model_name is not set. '
                +
                f'Defaulting to final_register_only=False and saving the HuggingFace checkpoint to {save_folder=}.',
            )

        # mlflow config setup
        if mlflow_logging_config is None:
            mlflow_logging_config = {}
        if self.mlflow_registered_model_name is not None:
            # Both the metadata and the task are needed in order for mlflow
            # and databricks optimized model serving to work
            passed_metadata = mlflow_logging_config.get('metadata', {})
            mlflow_logging_config['metadata'] = passed_metadata
            mlflow_logging_config.setdefault('task', 'llm/v1/completions')

            default_input_example = {
                'prompt': np.array(['What is Machine Learning?']),
            }
            is_chat = mlflow_logging_config['task'].endswith('chat') or (
                mlflow_logging_config['metadata'] is not None and
                mlflow_logging_config['metadata'].get('task',
                                                      '').endswith('chat')
            )
            if is_chat:
                default_input_example = {
                    'messages': [{
                        'role': 'user',
                        'content': 'What is Machine Learning?',
                    }],
                }
            mlflow_logging_config.setdefault(
                'input_example',
                default_input_example,
            )

        self.mlflow_logging_config = mlflow_logging_config
        if 'metadata' in self.mlflow_logging_config:
            self.pretrained_model_name = self.mlflow_logging_config[
                'metadata'].get(
                    'pretrained_model_name',
                    None,
                )
        else:
            self.pretrained_model_name = None

        self.huggingface_folder_name_fstr = os.path.join(
            'huggingface',
            huggingface_folder_name,
        )

        self.save_interval: Time = Time.from_input(
            save_interval,
            TimeUnit.EPOCH,
        )
        self.check_interval = create_interval_scheduler(
            self.save_interval,
            include_end_of_training=True,
        )
        self.remote_ud = maybe_create_remote_uploader_downloader_from_uri(
            save_folder,
            loggers=[],
        )
        if self.remote_ud is not None:
            self.remote_ud._num_concurrent_uploads = 4

        self.last_checkpoint_batch: Optional[Time] = None
        self.mlflow_loggers = []

        self.register_processes: list[SpawnProcess] = []
        # Temporary save directory used by child_processes.
        self.temp_save_dir = None

    def run_event(self, event: Event, state: State, logger: Logger) -> None:
        # The interval scheduler handles only returning True for the appropriate events
        if state.get_elapsed_duration() is not None and self.check_interval(
            state,
            event,
        ) and self.last_checkpoint_batch != state.timestamp.batch:
            is_last_batch = self._is_last_batch(state)
            self._save_checkpoint(
                state,
                logger,
                register_to_mlflow=(
                    self.mlflow_registered_model_name is not None and
                    is_last_batch
                ),
                upload_to_save_folder=not self.final_register_only or
                not is_last_batch,
            )
        elif event == Event.INIT:
            if not isinstance(state.model, HuggingFaceModel):
                raise ValueError(
                    f'`HuggingFaceCheckpointer` is only compatible with `HuggingFaceModel`s. '
                    + f'Got {type(state.model)} instead.',
                )
            if self.remote_ud is not None:
                try:
                    self.remote_ud.init(state, logger)
                except PermissionError as e:
                    if 'Client Error' in str(
                        e,
                    ):  # thrown from composer.utils._wrap_mlflow_exceptions
                        raise StoragePermissionError(
                            'Error when write to save_folder.',
                        ) from e
                    raise e
                state.callbacks.append(self.remote_ud)

            if self.mlflow_registered_model_name is not None:
                self.mlflow_loggers = [
                    logger_destination
                    for logger_destination in logger.destinations
                    if isinstance(logger_destination, MLFlowLogger)
                ]
                if len(self.mlflow_loggers) == 0:
                    raise ValueError(
                        f'`mlflow_registered_model_name` was set, but no `MLFlowLogger` was found in the `logger.destinations` list. '
                        +
                        'Please add an `MLFlowLogger` or set `mlflow_registered_model_name` to `None`.',
                    )

                import mlflow
                import mlflow.environment_variables
                mlflow.environment_variables.MLFLOW_HUGGINGFACE_MODEL_MAX_SHARD_SIZE.set(
                    '1GB',
                )

            # Check if the model is using PEFT
            if state.is_model_ddp:
                composer_model = state.model.module
            elif isinstance(state.model.model, FSDP):
                composer_model = state.model
            else:
                composer_model = state.model
            self.using_peft = composer_model.using_peft
        elif event == Event.FIT_END:
            # Wait for all child processes spawned by the callback to finish.
            timeout = self.register_wait_seconds
            wait_start = time.time()
            while not self._all_register_processes_done(state.device):
                wait_time = time.time() - wait_start
                if wait_time > timeout:
                    raise TimeoutError(
                        f'Waited {wait_time} seconds for child processes to complete. Exceeded timeout of {timeout} seconds.',
                    )
                time.sleep(2)

            if self._any_register_processes_error(
                state.device,
            ) and self.final_register_only:
                log.error(
                    'An error occurred in one or more registration processes. Fallback to saving the HuggingFace checkpoint.',
                )
                self._save_checkpoint(
                    state,
                    logger,
                    upload_to_save_folder=True,
                    register_to_mlflow=False,
                )

            # Clean up temporary save directory; all processes are done with it.
            if self.temp_save_dir is not None:
                shutil.rmtree(self.temp_save_dir)

    def _is_last_batch(self, state: State):
        elapsed_duration = state.get_elapsed_duration()
        if elapsed_duration is not None and elapsed_duration >= 1.0:
            return True

        assert state.max_duration is not None  # for pyright

        epoch_complete = state.dataloader_len == state.timestamp.batch_in_epoch
        second_to_last_epoch = state.max_duration.unit == TimeUnit.EPOCH and (
            state.timestamp.epoch == state.max_duration.value - 1
        )
        # If the save interval is specified as exactly the same number of batches as the total duration,
        # but the max duration is specified in epochs, we need a special case to identify we are on the last batch
        # and should write the mlflow checkpoint. This should occur on the last batch of the final epoch.
        if self.save_interval.unit == TimeUnit.BATCH and second_to_last_epoch and epoch_complete:
            return True

        # If the save interval is specified as 1dur, and the max duration is in epoch units
        # we need a special case to identify we are on the last batch and should write the mlflow checkpoint
        if self.save_interval.unit == TimeUnit.DURATION and self.save_interval.value == 1 and state.max_duration.unit == TimeUnit.EPOCH:
            assert state.dataloader_len is not None  # for pyright
            return int(state.timestamp.batch) % math.ceil(
                state.max_duration.value * state.dataloader_len,
            ) == 0

        return False

    def _all_register_processes_done(self, device: Device) -> bool:
        not_done = any(
            process.is_alive() for process in self.register_processes
        )
        x = device.tensor_to_device(torch.tensor(1 if not_done else 0))
        dist.all_reduce(x, reduce_operation='MAX')
        return x.item() == 0

    def _any_register_processes_error(self, device: Device) -> bool:
        has_errors = any(
            process.exitcode is not None and process.exitcode != 0
            for process in self.register_processes
        )
        x = device.tensor_to_device(torch.tensor(1 if has_errors else 0))
        dist.all_reduce(x, reduce_operation='MAX')
        return x.item() == 1

    def transform_model_and_tokenizer(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
    ) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """Transform the model and tokenizer before saving.

        This allows a subclass to modify the model and tokenizer before saving. The base class implementation will
        make no modifications.

        Args:
            model (PreTrainedModel): The model to be transformed.
            tokenizer (PreTrainedTokenizerBase): The tokenizer to be transformed.

        Returns:
            Tuple[PreTrainedModel, PreTrainedTokenizerBase]: The transformed model and tokenizer.
        """
        return model, tokenizer

    def transform_config(
        self,
        original_config: PretrainedConfig,
    ) -> PretrainedConfig:
        """Transform the model config before saving.

        Args:
            original_config (Any): The original model config.

        Returns:
            The transformed model config.
        """
        copied_config = copy.deepcopy(original_config)
        if copied_config.model_type == 'mpt':
            copied_config.attn_config['attn_impl'] = 'torch'
            copied_config.init_device = 'cpu'
            if 'moe_world_size' in getattr(copied_config, 'ffn_config', {}):
                copied_config.ffn_config['moe_world_size'] = 1
        return copied_config

    def pre_register_edit(self, local_save_path: str):
        """Edit the model before registering with MLflow.

        This allows a subclass to modify the model before registering with MLflow. The base class implementation will
        make no modifications.

        Args:
            local_save_path (str): The path to the model to be transformed.
        """
        pass

    def transform_model_pre_registration(
        self,
        model: PreTrainedModel,
    ) -> PreTrainedModel:
        """Transform the model before registering with MLflow.

        This allows a subclass to modify the model before registering with MLflow. The base class implementation will
        make no modifications.

        Args:
            model (PreTrainedModel): The model to be transformed.

        Returns:
            PreTrainedModel: The transformed model.
        """
        return model

    def _get_hf_model(self, state: State):
        self.last_checkpoint_batch = state.timestamp.batch

        log.info('Saving HuggingFace formatted checkpoint')

        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        CONFIG_MAPPING._extra_content['mpt'] = MPTConfig
        MPTConfig.register_for_auto_class()
        MPTForCausalLM.register_for_auto_class('AutoModelForCausalLM')

        log.debug('Gathering state dict')

        if state.is_model_ddp:
            original_model: PreTrainedModel = state.model.module.model
            state_dict_model = state.model.module.model
            original_tokenizer = state.model.module.tokenizer
        elif isinstance(state.model.model, FSDP):
            original_model: PreTrainedModel = state.model.model.module
            state_dict_model = state.model.model
            original_tokenizer = state.model.tokenizer
        else:
            original_model: PreTrainedModel = state.model.model
            state_dict_model = state.model.model
            original_tokenizer = state.model.tokenizer

        cpu_offload = True

        # Add hook to move tensors to cpu to avoid CUDA OOM
        def tensor_hook(
            module: nn.Module,
            state_dict: dict[str, Any],
            prefix: str,
            *args: Any,
        ) -> dict[str, Any]:
            dtensor_fqns = []
            for fqn in state_dict.keys():
                tensor = state_dict[fqn]
                if isinstance(tensor, DTensor):
                    dtensor_fqns.append(fqn)
                    tensor = tensor.full_tensor()  # type: ignore
                    if dist.get_global_rank() == 0:
                        # Offload any DTensors to CPU
                        if cpu_offload:
                            tensor = tensor.cpu()
                        state_dict[fqn] = tensor
                    else:
                        state_dict[fqn] = None

                if isinstance(state_dict[fqn], torch.Tensor):
                    state_dict[fqn] = state_dict[fqn].to(dtype=self.dtype)
                del tensor
            if dist.get_global_rank() != 0:
                state_dict = {}
            return state_dict

        hooks = []
        for _, module in state_dict_model.named_modules():
            hooks.append(module._register_state_dict_hook(tensor_hook),)

        state_dict = get_model_state_dict(
            state_dict_model,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=cpu_offload,
            ),
        )
        for hook in hooks:
            hook.remove()

        new_model_instance = None  # Need this for pyright because variable could be unbound

        if dist.get_global_rank() == 0:
            log.debug('Saving Hugging Face checkpoint in global rank 0')

            # Transform HF config before building 2nd model copy
            new_config = self.transform_config(
                original_config=original_model.config,
            )

            log.debug(f'Creating new model instance')

            # First create the model instance on meta device to avoid the
            # initialization cost.
            with init_empty_weights():
                if self.using_peft:
                    active_adapter = original_model.active_adapter
                    base_model = original_model.get_base_model()
                    new_base_model_instance = type(base_model)(new_config)

                    new_model_instance = type(original_model)(
                        new_base_model_instance,
                        original_model.peft_config[active_adapter],
                    )
                    del new_base_model_instance
                else:
                    new_model_instance = type(original_model)(new_config)
                    if new_model_instance.generation_config is not None:
                        new_model_instance.generation_config.update(
                            **original_model.generation_config.to_dict(),
                        )

            # Then load the state dict in with "assign" so that the state dict
            # is loaded properly even though the model is initially on meta device.
            new_model_instance.load_state_dict(state_dict, assign=True)
            del state_dict

            # Transform the model and tokenizer before saving
            new_model_instance, original_tokenizer = self.transform_model_and_tokenizer(
                new_model_instance,
                original_tokenizer,
            )

            # Ensure that the pretrained model name is correctly set on the saved HF checkpoint.
            if self.pretrained_model_name is not None:
                new_model_instance.name_or_path = self.pretrained_model_name
                if self.using_peft:
                    new_model_instance.base_model.name_or_path = self.pretrained_model_name
                    for k in new_model_instance.peft_config.keys():
                        new_model_instance.peft_config[
                            k
                        ].base_model_name_or_path = self.pretrained_model_name

            log.debug('Saving Hugging Face checkpoint to disk')

        return new_model_instance, original_tokenizer

    def _register_hf_model(
        self,
        temp_save_dir: str,
        original_tokenizer: PreTrainedTokenizerBase,
        use_temp_dir: bool,
        new_model_instance: PreTrainedModel,
    ):
        assert new_model_instance is not None
        new_model_instance = self.transform_model_pre_registration(
            new_model_instance,
        )
        register_save_dir = os.path.join(
            temp_save_dir,
            'register_save',
        )
        new_model_instance.save_pretrained(
            register_save_dir,
            max_shard_size='1GB',
        )
        if original_tokenizer:
            original_tokenizer.save_pretrained(register_save_dir)

        self.pre_register_edit(register_save_dir)

        for mlflow_logger in self.mlflow_loggers:
            if self.mlflow_registered_model_name:
                log.debug(
                    f'Registering model to UC at {mlflow_logger.model_registry_prefix}.{self.mlflow_registered_model_name}',
                )

            # Save the monitor process to be restored after registering the model.
            with _monitor_process_saver(mlflow_logger):
                process = SpawnProcess(
                    target=_log_model_with_multi_process,
                    kwargs={
                        'mlflow_logger':
                            mlflow_logger,
                        'python_logging_level':
                            logging.getLogger('llmfoundry').level,
                        'transformers_model': {
                            'model': new_model_instance,
                            'tokenizer': original_tokenizer,
                        } if self.using_peft else register_save_dir,
                        'artifact_path':
                            'final_model_checkpoint',
                        'pretrained_model_name':
                            self.pretrained_model_name,
                        'registered_model_name':
                            self.mlflow_registered_model_name,
                        'await_registration_for':
                            3600,
                        'mlflow_logging_config':
                            self.mlflow_logging_config,
                    },
                )

                process.start()
                self.register_processes.append(process)

        # Save the temporary directory to be cleaned up later.
        if use_temp_dir:
            self.temp_save_dir = temp_save_dir

    def _save_checkpoint(
        self,
        state: State,
        logger: Logger,
        upload_to_save_folder: bool,
        register_to_mlflow: bool,
    ):
        """Save a HuggingFace formatted checkpoint.

        Args:
            state (State): The training state.
            logger (Logger): The logger.
            upload_to_save_folder (bool): Whether to upload the HF checkpoint to the save folder.
            register_to_mlflow (bool): Whether to register the model to MLFlow
        """
        del logger  # unused

        save_dir = format_name_with_dist_and_time(
            str(
                Path(self.save_dir_format_str) /
                self.huggingface_folder_name_fstr,
            ),
            state.run_name,
            state.timestamp,
        )

        # Use a temporary directory if save_dir is remote.
        use_temp_dir = self.remote_ud is not None
        temp_save_dir = tempfile.mkdtemp() if use_temp_dir else save_dir

        new_model_instance, original_tokenizer = self._get_hf_model(state)

        dist.barrier()

        if dist.get_global_rank() == 0:
            assert new_model_instance is not None
            if upload_to_save_folder:
                # This context manager casts the TE extra state in io.BytesIO format to tensor format
                # Needed for proper hf ckpt saving.
                context_manager = te.onnx_export(
                    True,
                ) if is_te_imported and state.precision == Precision.AMP_FP8 else contextlib.nullcontext(
                )
                with context_manager:
                    new_model_instance.save_pretrained(
                        temp_save_dir,
                        max_shard_size='1GB',
                    )
                if original_tokenizer is not None:
                    assert isinstance(
                        original_tokenizer,
                        PreTrainedTokenizerBase,
                    )
                    original_tokenizer.save_pretrained(temp_save_dir)

                # Only need to edit files for MPT because it has custom code
                if new_model_instance.config.model_type == 'mpt':
                    log.debug('Editing MPT files for HuggingFace compatibility')
                    edit_files_for_hf_compatibility(
                        temp_save_dir,
                        self.flatten_imports,
                    )

                if self.remote_ud is not None:
                    for filename in os.listdir(temp_save_dir):
                        remote_file_name = os.path.join(save_dir, filename)
                        remote_file_uri = self.remote_ud.remote_backend.get_uri(
                            remote_file_name,
                        )
                        log.info(
                            f'Uploading HuggingFace formatted checkpoint to {remote_file_uri}',
                        )
                        self.remote_ud.upload_file(
                            state=state,
                            remote_file_name=remote_file_name,
                            file_path=Path(
                                os.path.join(temp_save_dir, filename),
                            ),
                            overwrite=self.overwrite,
                        )

        dist.barrier()

        if dist.get_global_rank() == 0:
            assert new_model_instance is not None
            if self.using_peft:
                model_name = self.mlflow_logging_config.get('metadata', {}).get(
                    'pretrained_model_name',
                    None,
                )
                if model_name is not None:
                    new_model_instance.name_or_path = model_name
                    new_model_instance.model.name_or_path = model_name
                    new_model_instance.base_model.name_or_path = model_name
            if register_to_mlflow:
                self._register_hf_model(
                    temp_save_dir,
                    original_tokenizer,
                    use_temp_dir,
                    new_model_instance,
                )
            else:
                # Clean up the temporary directory if we don't need to register to mlflow.
                if use_temp_dir:
                    shutil.rmtree(temp_save_dir)
        dist.barrier()
