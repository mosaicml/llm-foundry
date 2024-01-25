# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import copy
import logging
import math
import os
import tempfile
from pathlib import Path
from typing import Optional, Sequence, Union

import torch
from composer.core import Callback, Event, State, Time, TimeUnit
from composer.core.state import fsdp_state_dict_type_context
from composer.loggers import Logger, MLFlowLogger
from composer.models import HuggingFaceModel
from composer.models.huggingface import peft_installed
from composer.utils import (dist, format_name_with_dist_and_time,
                            maybe_create_remote_uploader_downloader_from_uri,
                            parse_uri)
from composer.utils.misc import create_interval_scheduler
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from llmfoundry.models.mpt import MPTConfig, MPTForCausalLM
from llmfoundry.utils.huggingface_hub_utils import \
    edit_files_for_hf_compatibility

log = logging.getLogger(__name__)


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
            ``{'task': 'llm/v1/completions'}`` respectively.
        flatten_imports (Sequence[str]): A sequence of import prefixes that will
            be flattened when editing MPT files.
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

        # mlflow config setup
        self.mlflow_registered_model_name = mlflow_registered_model_name
        if mlflow_logging_config is None:
            mlflow_logging_config = {}
        if self.mlflow_registered_model_name is not None:
            # Both the metadata and the task are needed in order for mlflow
            # and databricks optimized model serving to work
            default_metadata = {'task': 'llm/v1/completions'}
            passed_metadata = mlflow_logging_config.get('metadata', {})
            mlflow_logging_config['metadata'] = {
                **default_metadata,
                **passed_metadata
            }
            mlflow_logging_config.setdefault('task', 'text-generation')
        self.mlflow_logging_config = mlflow_logging_config

        self.huggingface_folder_name_fstr = os.path.join(
            'huggingface', huggingface_folder_name)

        self.save_interval: Time = Time.from_input(save_interval,
                                                   TimeUnit.EPOCH)
        self.check_interval = create_interval_scheduler(
            self.save_interval, include_end_of_training=True)
        self.remote_ud = maybe_create_remote_uploader_downloader_from_uri(
            save_folder, loggers=[])
        if self.remote_ud is not None:
            self.remote_ud._num_concurrent_uploads = 4

        self.last_checkpoint_batch: Optional[Time] = None
        self.mlflow_loggers = []

    def run_event(self, event: Event, state: State, logger: Logger) -> None:
        # The interval scheduler handles only returning True for the appropriate events
        if state.get_elapsed_duration() is not None and self.check_interval(
                state,
                event) and self.last_checkpoint_batch != state.timestamp.batch:
            self._save_checkpoint(state, logger)
        elif event == Event.INIT:
            if not isinstance(state.model, HuggingFaceModel):
                raise ValueError(
                    f'`HuggingFaceCheckpointer` is only compatible with `HuggingFaceModel`s. '
                    + f'Got {type(state.model)} instead.')
            if self.remote_ud is not None:
                self.remote_ud.init(state, logger)
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
                        'Please add an `MLFlowLogger` or set `mlflow_registered_model_name` to `None`.'
                    )

                import mlflow
                mlflow.environment_variables.MLFLOW_HUGGINGFACE_MODEL_MAX_SHARD_SIZE.set(
                    '5GB')

    def _is_last_batch(self, state: State):
        elapsed_duration = state.get_elapsed_duration()
        if elapsed_duration is not None and elapsed_duration >= 1.0:
            return True

        assert state.max_duration is not None  # for pyright
        # If the save interval is specified as 1dur, and the max duration is in epoch units
        # we need a special case to identify we are on the last batch and should write the mlflow checkpoint
        if self.save_interval.unit == TimeUnit.DURATION and self.save_interval.value == 1 and state.max_duration.unit == TimeUnit.EPOCH:
            assert state.dataloader_len is not None  # for pyright
            return int(state.timestamp.batch) % math.ceil(
                state.max_duration.value * state.dataloader_len) == 0

        return False

    def _save_checkpoint(self, state: State, logger: Logger):
        del logger  # unused

        self.last_checkpoint_batch = state.timestamp.batch

        log.info('Saving HuggingFace formatted checkpoint')

        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        CONFIG_MAPPING._extra_content['mpt'] = MPTConfig
        MPTConfig.register_for_auto_class()
        MPTForCausalLM.register_for_auto_class('AutoModelForCausalLM')

        save_dir = format_name_with_dist_and_time(
            str(
                Path(self.save_dir_format_str) /
                self.huggingface_folder_name_fstr), state.run_name,
            state.timestamp)
        dir_context_mgr = tempfile.TemporaryDirectory(
        ) if self.remote_ud is not None else contextlib.nullcontext(
            enter_result=save_dir)

        with dir_context_mgr as temp_save_dir:
            assert isinstance(temp_save_dir,
                              str)  # pyright doesn't know about enter_result

            log.debug('Gathering state dict')
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

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

            state_dict_context = fsdp_state_dict_type_context(
                original_model, state_dict_type='full') if (
                    (not state.is_model_ddp) and isinstance(
                        state_dict_model, FSDP)) else contextlib.nullcontext()

            with state_dict_context:
                state_dict = state_dict_model.state_dict()

                # convert the state dict to the requested precision
                for k, v in state_dict.items():
                    if isinstance(v, torch.Tensor):
                        state_dict[k] = v.to(dtype=self.dtype)

            if dist.get_global_rank() == 0:
                log.debug('Saving Hugging Face checkpoint in global rank 0')

                copied_config = copy.deepcopy(original_model.config)
                if copied_config.model_type == 'mpt':
                    copied_config.attn_config['attn_impl'] = 'torch'
                    copied_config.init_device = 'cpu'

                # TODO: after torch 2.1, we can load a state dict into a meta model
                # and skip the extra model init
                log.debug(f'Creating new model instance')

                from peft import PeftModel
                is_peft_model = False
                if peft_installed:
                    is_peft_model = isinstance(original_model, PeftModel)
                if is_peft_model:
                    active_adapter = original_model.active_adapter
                    base_model = original_model.get_base_model()
                    new_base_model_instance = type(base_model)(copied_config)

                    new_model_instance = type(original_model)(
                        new_base_model_instance,
                        original_model.peft_config[active_adapter])
                else:
                    new_model_instance = type(original_model)(copied_config)

                new_model_instance.to(dtype=self.dtype)
                new_model_instance.load_state_dict(state_dict)
                del state_dict

                log.debug('Saving Hugging Face checkpoint to disk')
                new_model_instance.save_pretrained(temp_save_dir)
                if original_tokenizer is not None:
                    assert isinstance(original_tokenizer,
                                      PreTrainedTokenizerBase)
                    original_tokenizer.save_pretrained(temp_save_dir)

                # Only need to edit files for MPT because it has custom code
                if original_model.config.model_type == 'mpt':
                    log.debug('Editing MPT files for HuggingFace compatibility')
                    edit_files_for_hf_compatibility(
                        temp_save_dir,
                        self.flatten_imports,
                    )

                if self.remote_ud is not None:
                    for filename in os.listdir(temp_save_dir):
                        remote_file_name = os.path.join(save_dir, filename)
                        remote_file_uri = self.remote_ud.remote_backend.get_uri(
                            remote_file_name)
                        log.info(
                            f'Uploading HuggingFace formatted checkpoint to {remote_file_uri}'
                        )
                        self.remote_ud.upload_file(
                            state=state,
                            remote_file_name=remote_file_name,
                            file_path=Path(os.path.join(temp_save_dir,
                                                        filename)),
                            overwrite=self.overwrite,
                        )

                if self.mlflow_registered_model_name and self._is_last_batch(
                        state):
                    components = {'model': new_model_instance}
                    if original_tokenizer is not None:
                        components['tokenizer'] = original_tokenizer

                    log.debug('Logging Hugging Face model to MLFlow')
                    for i, mlflow_logger in enumerate(self.mlflow_loggers):
                        log.debug(
                            f'Registering model to UC at {mlflow_logger.model_registry_prefix}.{self.mlflow_registered_model_name}'
                        )
                        local_save_path = str(
                            Path(temp_save_dir) / f'mlflow_save_{i}')

                        # TODO: Remove after mlflow fixes the bug that makes this necessary
                        import mlflow
                        mlflow.store._unity_catalog.registry.rest_store.get_feature_dependencies = lambda *args, **kwargs: ''
                        mlflow_logger.save_model(
                            flavor='transformers',
                            transformers_model=components,
                            path=local_save_path,
                            **self.mlflow_logging_config,
                        )
                        mlflow_logger.register_model(
                            model_uri=local_save_path,
                            name=self.mlflow_registered_model_name,
                            await_registration_for=3600,
                        )
