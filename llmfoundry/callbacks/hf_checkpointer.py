# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import copy
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional, Union

import torch
from composer.callbacks.utils import create_interval_scheduler
from composer.core import Callback, Event, State, Time
from composer.core.state import fsdp_state_dict_type_context
from composer.loggers import Logger, MLFlowLogger
from composer.loggers.remote_uploader_downloader import RemoteUploaderDownloader
from composer.models import HuggingFaceModel
from composer.utils import dist, format_name_with_dist_and_time, parse_uri
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from llmfoundry.models.mpt import MPTConfig, MPTForCausalLM
from llmfoundry.utils.huggingface_hub_utils import \
    edit_files_for_hf_compatibility

log = logging.getLogger(__name__)


class HuggingFaceCheckpointer(Callback):
    """Save a huggingface formatted checkpoint during training.

    Args:
        save_folder (str): Top level folder to save checkpoints to (can be a URI). It is likely that
            this would be the same as your save_folder.
        save_interval: Union[str, int, Time]: The interval describing how often checkpoints should be
            saved. If an integer, it will be assumed to be in :attr:`.TimeUnit.EPOCH`.
            Otherwise, the unit must be either :attr:`.TimeUnit.EPOCH`, :attr:`.TimeUnit.BATCH`,
            :attr:`.TimeUnit.TOKEN`, or :attr:`.TimeUnit.SAMPLE`.
        huggingface_folder_name (str): Folder to save each checkpoint under (can be a format string). Default is ``ba{batch}``.
        precision: The precision to save the model in. Default is ``float32``. Options are ``bfloat16``, ``float16``, or ``float32``.
        overwrite (bool): Whether to overwrite previous checkpoints.
        log_to_mlflow (bool): Whether to log and register the checkpoint to MLflow. Default is ``False``.
        mlflow_logging_config (Optional[dict]): A dictionary of config arguments that will get passed along to the MLflow ``log_model`` call.
            Expected to contain ``metadata`` and ``task`` keys. If either is unspecified, the defaults are ``'text-generation'`` and
            
    """

    def __init__(
        self,
        save_folder: str,
        save_interval: Union[str, int, Time],
        huggingface_folder_name: str = 'ba{batch}',
        precision: str = 'float32',
        overwrite: bool = False,
        log_to_mlflow: bool = False,
        mlflow_logging_config: Optional[dict] = None,
    ):
        self.backend, self.bucket_name, self.save_dir_format_str = parse_uri(
            save_folder)
        self.overwrite = overwrite
        self.precision = precision
        self.dtype = {
            'float32': torch.float32,
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
        }[precision]

        # mlflow config setup
        self.log_to_mlflow = log_to_mlflow
        if mlflow_logging_config is None:
            mlflow_logging_config = {}
        if 'metadata' not in mlflow_logging_config:
            mlflow_logging_config['metadata'] = {'task': 'llm/v1/completions'}
        if 'task' not in mlflow_logging_config:
            mlflow_logging_config['task'] = 'text-generation'
        self.mlflow_logging_config = mlflow_logging_config

        self.huggingface_folder_name_fstr = os.path.join(
            'huggingface', huggingface_folder_name)
        self.check_interval = create_interval_scheduler(
            save_interval, include_end_of_training=True)
        self.upload_to_object_store = (self.backend != '')
        if self.upload_to_object_store:
            self.remote_ud = RemoteUploaderDownloader(
                bucket_uri=f'{self.backend}://{self.bucket_name}',
                num_concurrent_uploads=4)
        else:
            self.remote_ud = None

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
            if self.upload_to_object_store and self.remote_ud is not None:
                self.remote_ud.init(state, logger)
                state.callbacks.append(self.remote_ud)

            if self.log_to_mlflow:
                self.mlflow_loggers = [
                    logger_destination
                    for logger_destination in logger.destinations
                    if isinstance(logger_destination, MLFlowLogger)
                ]
                if len(self.mlflow_loggers) == 0:
                    raise ValueError(
                        f'`log_to_mlflow` was set to `True` but no `MLFlowLogger` was found in the `logger.destinations` list. '
                        +
                        'Please add an `MLFlowLogger` or set `log_to_mlflow` to `False`.'
                    )

                import mlflow
                mlflow.environment_variables.MLFLOW_HUGGINGFACE_MODEL_MAX_SHARD_SIZE.set(
                    '5GB')
                # mlflow.set_registry_uri('databricks-uc')

    def _save_checkpoint(self, state: State, logger: Logger):
        del logger  # unused

        self.last_checkpoint_batch = state.timestamp.batch

        log.info('Saving HuggingFace formatted checkpoint')

        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        CONFIG_MAPPING._extra_content['mpt'] = MPTConfig
        MPTConfig.register_for_auto_class()
        MPTForCausalLM.register_for_auto_class('AutoModelForCausalLM')

        assert isinstance(state.model, HuggingFaceModel)

        save_dir = format_name_with_dist_and_time(
            str(
                Path(self.save_dir_format_str) /
                self.huggingface_folder_name_fstr), state.run_name,
            state.timestamp)
        dir_context_mgr = tempfile.TemporaryDirectory(
        ) if self.upload_to_object_store else contextlib.nullcontext(
            enter_result=save_dir)

        with dir_context_mgr as temp_save_dir:
            assert isinstance(temp_save_dir,
                              str)  # pyright doesn't know about enter_result

            log.debug('Gathering state dict')
            with fsdp_state_dict_type_context(state.model.model,
                                              state_dict_type='full'):
                state_dict = state.model.model.state_dict()

                # convert the state dict to the requested precision
                for k, v in state_dict.items():
                    if isinstance(v, torch.Tensor):
                        state_dict[k] = v.to(dtype=self.dtype)

            if dist.get_global_rank() == 0:
                log.debug('Saving Hugging Face checkpoint to disk')
                # We raise above if the model is not a HuggingFaceModel, so this assert is safe
                assert hasattr(state.model.model, 'save_pretrained')
                state.model.model.save_pretrained(temp_save_dir,
                                                  state_dict=state_dict)

                if state.model.tokenizer is not None:
                    assert isinstance(state.model.tokenizer,
                                      PreTrainedTokenizerBase)
                    state.model.tokenizer.save_pretrained(temp_save_dir)

                # Only need to edit files for MPT because it has custom code
                if state.model.model.config.model_type == 'mpt':
                    edit_files_for_hf_compatibility(temp_save_dir)

                with open(os.path.join(temp_save_dir, 'config.json'), 'r') as f:
                    edited_config = json.load(f)

                if state.model.model.config.model_type == 'mpt':
                    edited_config['attn_config']['attn_impl'] = 'torch'
                    edited_config['init_device'] = 'cpu'

                edited_config['torch_dtype'] = self.precision
                with open(os.path.join(temp_save_dir, 'config.json'), 'w') as f:
                    json.dump(edited_config, f, indent=4)

                if self.upload_to_object_store:
                    assert self.remote_ud is not None
                    log.info(
                        f'Uploading HuggingFace formatted checkpoint to {self.backend}://{self.bucket_name}/{save_dir}'
                    )
                    for filename in os.listdir(temp_save_dir):
                        self.remote_ud.upload_file(
                            state=state,
                            remote_file_name=os.path.join(save_dir, filename),
                            file_path=Path(os.path.join(temp_save_dir,
                                                        filename)),
                            overwrite=self.overwrite,
                        )

                elapsed_duration = state.get_elapsed_duration()
                if self.log_to_mlflow and elapsed_duration is not None and elapsed_duration >= 1.0:
                    log.debug('Reloading model to log to MLFlow')

                    from torch.distributed.fsdp import \
                        FullyShardedDataParallel as FSDP
                    if isinstance(state.model.model, FSDP):
                        model_class = state.model.model.module
                    else:
                        model_class = state.model.model

                    copied_config = copy.deepcopy(state.model.model.config)
                    if state.model.model.config.model_type == 'mpt':
                        copied_config.attn_config['attn_impl'] = 'torch'
                        copied_config.init_device = 'cpu'

                    new_model_instance = type(model_class)(copied_config)
                    new_model_instance.to(dtype=self.dtype)
                    new_model_instance.load_state_dict(state_dict)
                    del state_dict
                    components = {'model': new_model_instance}
                    if state.model.tokenizer is not None:
                        new_tokenizer_instance = AutoTokenizer.from_pretrained(
                            temp_save_dir)
                        components['tokenizer'] = new_tokenizer_instance

                    log.debug('Logging Hugging Face model to MLFlow')
                    registered_model_name = f'{state.run_name}_{os.path.basename(save_dir)}'
                    for i, mlflow_logger in enumerate(self.mlflow_loggers):
                        log.debug(f'Registering model to UC at {mlflow_logger.model_registry_prefix}.{registered_model_name}')
                        local_save_path = str(
                            Path(temp_save_dir) / f'mlflow_save_{i}')
                        import mlflow
                        mlflow.store._unity_catalog.registry.rest_store.get_feature_dependencies = lambda *args, **kwargs: ""
                        mlflow_logger.save_model(
                            flavor='transformers',
                            transformers_model=components,
                            path=local_save_path,
                            **self.mlflow_logging_config,
                        )
                        mlflow_logger.register_model(
                            model_uri=local_save_path,
                            name=registered_model_name,
                            await_registration_for=3600,
                        )

