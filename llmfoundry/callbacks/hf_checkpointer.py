# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import copy
import logging
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import torch
from composer.core import Callback, Event, State, Time, TimeUnit
from composer.core.state import fsdp_state_dict_type_context
from composer.loggers import Logger, MLFlowLogger
from composer.models import HuggingFaceModel
from composer.utils import (dist, format_name_with_dist_and_time,
                            maybe_create_remote_uploader_downloader_from_uri,
                            parse_uri)
from composer.utils.misc import create_interval_scheduler
from mlflow.transformers import _fetch_model_card, _write_license_information
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from llmfoundry.models.mpt import MPTConfig, MPTForCausalLM
from llmfoundry.models.utils import init_empty_weights
from llmfoundry.utils.huggingface_hub_utils import \
    edit_files_for_hf_compatibility

log = logging.getLogger(__name__)

_LICENSE_FILE_PATTERN = re.compile(r'license(\.[a-z]+|$)', re.IGNORECASE)


def _maybe_get_license_filename(
        local_dir: str,
        pretrained_model_name: Optional[str] = None) -> Optional[str]:
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
        license_filename = next(file for file in os.listdir(local_dir)
                                if _LICENSE_FILE_PATTERN.search(file))

        # If a pretrained model name is provided, replace the license file with the correct info from HF Hub.
        if pretrained_model_name is not None:
            log.info(
                f'Overwriting license file {license_filename} with license info for model {pretrained_model_name} from Hugging Face Hub'
            )
            os.remove(os.path.join(local_dir, license_filename))
            model_card = _fetch_model_card(pretrained_model_name)

            local_dir_path = Path(local_dir).absolute()
            _write_license_information(pretrained_model_name, model_card,
                                       local_dir_path)
            license_filename = next(file for file in os.listdir(local_dir)
                                    if _LICENSE_FILE_PATTERN.search(file))

        return license_filename

    except StopIteration:
        return None


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
            import numpy as np
            from mlflow.models.signature import ModelSignature
            from mlflow.types.schema import ColSpec, Schema

            # Both the metadata and the task are needed in order for mlflow
            # and databricks optimized model serving to work
            default_metadata = {'task': 'llm/v1/completions'}
            passed_metadata = mlflow_logging_config.get('metadata', {})
            mlflow_logging_config['metadata'] = {
                **default_metadata,
                **passed_metadata
            }
            mlflow_logging_config.setdefault('task', 'text-generation')

            # Define a default input/output that is good for standard text generation LMs
            input_schema = Schema([
                ColSpec('string', 'prompt'),
                ColSpec('double', 'temperature', optional=True),
                ColSpec('integer', 'max_tokens', optional=True),
                ColSpec('string', 'stop', optional=True),
                ColSpec('integer', 'candidate_count', optional=True)
            ])

            output_schema = Schema([ColSpec('string', 'predictions')])

            default_signature = ModelSignature(inputs=input_schema,
                                               outputs=output_schema)

            default_input_example = {
                'prompt': np.array(['What is Machine Learning?'])
            }
            mlflow_logging_config.setdefault('input_example',
                                             default_input_example)
            mlflow_logging_config.setdefault('signature', default_signature)

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
                composer_model = state.model.module
                original_model: PreTrainedModel = state.model.module.model
                state_dict_model = state.model.module.model
                original_tokenizer = state.model.module.tokenizer
            elif isinstance(state.model.model, FSDP):
                composer_model = state.model
                original_model: PreTrainedModel = state.model.model.module
                state_dict_model = state.model.model
                original_tokenizer = state.model.tokenizer
            else:
                composer_model = state.model
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

                log.debug(f'Creating new model instance')

                if composer_model.using_peft:
                    # We don't use meta here because the state dict does not contain the full
                    # model, only the adapter weights.
                    active_adapter = original_model.active_adapter
                    base_model = original_model.get_base_model()
                    new_base_model_instance = type(base_model)(copied_config)

                    new_model_instance = type(original_model)(
                        new_base_model_instance,
                        original_model.peft_config[active_adapter])
                    new_model_instance.to(dtype=self.dtype)
                else:
                    # First create the model instance on meta device to avoid the
                    # initialization cost.
                    with init_empty_weights():
                        new_model_instance = type(original_model)(copied_config)

                # Then load the state dict in with "assign" so that the state dict
                # is loaded properly even though the model is initially on meta device.
                new_model_instance.load_state_dict(state_dict, assign=True)
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
                        model_saving_kwargs: Dict[str, Any] = {
                            'path': local_save_path
                        }
                        if composer_model.using_peft:
                            model_saving_kwargs['flavor'] = 'peft'
                            model_saving_kwargs[
                                'save_pretrained_dir'] = temp_save_dir
                            model_saving_kwargs[
                                'metadata'] = self.mlflow_logging_config[
                                    'metadata']
                        else:
                            model_saving_kwargs['flavor'] = 'transformers'
                            model_saving_kwargs[
                                'transformers_model'] = components
                            model_saving_kwargs.update(
                                self.mlflow_logging_config)

                        mlflow_logger.save_model(**model_saving_kwargs)

                        # Upload the license file generated by mlflow during the model saving.
                        license_filename = _maybe_get_license_filename(
                            local_save_path,
                            self.mlflow_logging_config['metadata'].get(
                                'pretrained_model_name', None))
                        if license_filename is not None:
                            mlflow_logger._mlflow_client.log_artifact(
                                mlflow_logger._run_id,
                                os.path.join(local_save_path, license_filename),
                            )

                        mlflow_logger.register_model_with_run_id(
                            model_uri=local_save_path,
                            name=self.mlflow_registered_model_name,
                            await_creation_for=3600,
                        )
