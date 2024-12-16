# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Re-usable :class:`.ComposerModel` for LLM HF Models."""

from __future__ import annotations

import logging
import os
import warnings
from typing import TYPE_CHECKING, Any, Mapping, Optional, Union

import transformers
from composer.models.huggingface import HuggingFaceModel, peft_installed
from composer.utils import dist
from torchmetrics import Metric
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    GenerationConfig,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.models.auto.auto_factory import _BaseAutoModelClass
from transformers.utils.generic import ModelOutput

from llmfoundry.models.hf.hf_fsdp import (
    hf_get_init_device,
    prepare_hf_model_for_fsdp,
)
from llmfoundry.models.layers.attention import is_flash_v2_installed
from llmfoundry.models.utils import init_empty_weights
from llmfoundry.utils.config_utils import set_config_overrides

if TYPE_CHECKING:
    from peft import PeftConfig, PeftModel

__all__ = ['BaseHuggingFaceModel']

log = logging.getLogger(__name__)


class BaseHuggingFaceModel(HuggingFaceModel):
    """Wrapper around HuggingFaceModel.

    Base class for HuggingFace based models.
    """

    model_cls: Union[_BaseAutoModelClass,
                     PreTrainedModel] = AutoModelForCausalLM
    default_train_metrics: tuple = ()
    default_eval_metrics: tuple = ()

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        pretrained: bool = True,
        pretrained_lora_id_or_path: Optional[str] = None,
        trust_remote_code: bool = True,
        use_auth_token: bool = False,
        use_flash_attention_2: bool = False,
        load_in_8bit: bool = False,
        init_device: str = 'cpu',
        config_overrides: Optional[dict[str, Any]] = None,
        use_logits: bool = True,
        shift_labels: bool = False,
        peft_config: Optional[dict[str, Any]] = None,
        allow_embedding_resizing: bool = False,
        use_train_metrics: bool = True,
        additional_train_metrics: Optional[list] = None,
        additional_eval_metrics: Optional[list] = None,
        should_save_peft_only: bool = True,
    ):
        config_overrides = config_overrides or {}

        model = self.build_inner_model(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            pretrained_lora_id_or_path=pretrained_lora_id_or_path,
            trust_remote_code=trust_remote_code,
            init_device=init_device,
            use_flash_attention_2=use_flash_attention_2,
            use_auth_token=use_auth_token,
            config_overrides=config_overrides,
            load_in_8bit=load_in_8bit,
            pretrained=pretrained,
        )

        model = self.transform_model(model)

        metrics, eval_metrics = self.build_metrics(
            use_train_metrics=use_train_metrics,
            additional_train_metrics=additional_train_metrics,
            additional_eval_metrics=additional_eval_metrics,
        )

        if peft_config is not None and not peft_installed:
            raise ValueError(
                'PEFT is not installed, but peft_config was passed. Please install LLM Foundry with the peft extra to use peft_config.',
            )

        peft_config_object = None
        if peft_config is not None:
            peft_config_object = self.get_peft_config(peft_config)

        super().__init__(
            model=model,
            tokenizer=tokenizer,
            use_logits=use_logits,
            metrics=metrics,
            eval_metrics=eval_metrics,
            shift_labels=shift_labels,
            allow_embedding_resizing=allow_embedding_resizing,
            peft_config=peft_config_object,
            should_save_peft_only=should_save_peft_only,
        )

        # Prepare for FSDP needs to happen after the super init, so that any model
        # architecture changes are completed
        self.prepare_inner_model(self.model, init_device)

    def loss(self, outputs: ModelOutput, batch: Mapping):
        if self.config.use_return_dict:
            return outputs['loss']
        # loss is at index 0 in the output tuple, logits are at index 1
        return outputs[:2]

    def transform_model(self, model: PreTrainedModel) -> PreTrainedModel:
        """Transforms the model after initialization.

        Args:
            model (PreTrainedModel): The model to transform.

        Returns:
            PreTrainedModel: The transformed model.
        """
        return model

    @classmethod
    def build_config(
        cls,
        pretrained_model_name_or_path: str,
        trust_remote_code: bool,
        use_auth_token: bool,
        attn_implementation: str,
        config_overrides: dict[str, Any],
    ) -> PretrainedConfig:
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            use_auth_token=use_auth_token,
            attn_implementation=attn_implementation,
            use_cache=
            False,  # Necessary due to https://github.com/huggingface/transformers/issues/28056
        )

        set_config_overrides(config, config_overrides)

        return config

    @classmethod
    def build_metrics(
        cls,
        use_train_metrics: bool,
        additional_train_metrics: Optional[list[str]] = None,
        additional_eval_metrics: Optional[list[str]] = None,
    ) -> tuple[list[Metric], list[Metric]]:
        """Builds the training and evaluation metrics for the model.

        Args:
            use_train_metrics (bool): Whether to use training metrics.
            additional_train_metrics (Optional[List[str]]): Additional training metrics to include.
            additional_eval_metrics (Optional[List[str]]): Additional evaluation metrics to include.

        Returns:
            Tuple[List[Metric], List[Metric]]: A tuple containing the list of training metrics and evaluation metrics.
        """
        from llmfoundry.utils.builders import build_metric

        train_metric_names = list(
            cls.default_train_metrics,
        ) + (additional_train_metrics or [])
        train_metrics = [
            build_metric(metric, {}) for metric in train_metric_names
        ] if use_train_metrics else []
        eval_metric_names = list(
            cls.default_eval_metrics,
        ) + (additional_eval_metrics or [])
        eval_metrics = [
            build_metric(metric, {}) for metric in eval_metric_names
        ]

        return train_metrics, eval_metrics

    @classmethod
    def build_inner_model(
        cls,
        pretrained_model_name_or_path: str,
        pretrained_lora_id_or_path: Optional[str],
        trust_remote_code: bool,
        init_device: str,
        use_flash_attention_2: bool,
        use_auth_token: bool,
        config_overrides: dict[str, Any],
        load_in_8bit: bool,
        pretrained: bool,
        model_cls: Optional[Union[_BaseAutoModelClass, PreTrainedModel]] = None,
        prepare_for_fsdp: bool = False,
    ) -> Union[PreTrainedModel, 'PeftModel']:
        """Builds the inner model for the ComposerHFCausalLM.

        Args:
            pretrained_model_name_or_path (str): The pretrained model name or path.
            pretrained_lora_id_or_path (Optional[str]): The pretrained LORA ID or path.
            trust_remote_code (bool): Whether to trust remote code.
            init_device (str): The initialization device.
            use_flash_attention_2 (bool): Whether to use flash attention 2.
            use_auth_token (bool): Whether to use an authentication token.
            config_overrides (Dict[str, Any]): The configuration overrides.
            load_in_8bit (bool): Whether to load in 8-bit.
            pretrained (bool): Whether the model is pretrained.
            model_cls (Union[Type, Type[PreTrainedModel]]): Kept for backwards compatibility.
            prepare_for_fsdp (bool, optional): Kept for backwards compatilbility.

        Returns:
            Union[PreTrainedModel, 'PeftModel']: The built inner model.
        """
        if not trust_remote_code and pretrained_model_name_or_path.startswith(
            'mosaicml/mpt',
        ):
            raise ValueError(
                'trust_remote_code must be set to True for MPT models. Without this, the MPT model code will come from the transformers library, '
                +
                'which is significantly slower and not compatible with the LLM foundry training code, rather than the code release by MosaicML.',
            )
        # Resolve "mixed" init device to either "cpu" or "meta"
        resolved_init_device = hf_get_init_device(init_device)
        requested_attention_implementation = 'flash_attention_2' if use_flash_attention_2 else 'eager'

        if use_flash_attention_2 and not is_flash_v2_installed():
            raise ValueError(
                'use_flash_attention_2 is set to True, but flash-attention 2 is not installed. '
                + 'Please `pip install llm-foundry[gpu]`.',
            )

        auto_model_cls: Union[
            _BaseAutoModelClass,
            PreTrainedModel] = cls.model_cls if model_cls is None else model_cls

        if not (
            hasattr(auto_model_cls, 'from_pretrained') and
            hasattr(auto_model_cls, 'from_config')
        ):
            raise AttributeError(
                f'{auto_model_cls=} is missing `from_pretrained` and `from_config` support.',
            )

        # Hugging Face copies the modules into the
        # transformers modules cache. On particular systems, this operation seems to cause contention between
        # the different processes. To avoid this contention, we first create the config and generation config on local rank
        # zero. This will set up the transformers module cache and avoid the future contention.
        if dist.get_local_rank() == 0:
            AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=trust_remote_code,
                use_auth_token=use_auth_token,
                attn_implementation=requested_attention_implementation,
                use_cache=
                False,  # Necessary due to https://github.com/huggingface/transformers/issues/28056
            )
            try:
                GenerationConfig.from_pretrained(
                    pretrained_model_name_or_path,
                    use_auth_token=use_auth_token,
                )
            except OSError:
                pass

        dist.barrier()

        # Construct the Hugging Face config to use
        config = cls.build_config(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            use_auth_token=use_auth_token,
            attn_implementation=requested_attention_implementation,
            config_overrides=config_overrides,
        )

        # We need to have all non-zero local ranks be not-pretrained
        # Rank 0 will still be pretrained, and distribute the weights appropriately
        if dist.get_local_rank() != 0 and init_device == 'mixed':
            pretrained = False

        # Hugging Face copies the modules into the
        # transformers modules cache. On particular systems, this operation seems to cause contention between
        # the different processes. To avoid this contention, we first create the model (on meta device) on local rank
        # zero. This will set up the transformers model cache and avoid the future contention.
        if dist.get_local_rank() == 0:
            if pretrained and os.path.isdir(pretrained_model_name_or_path):
                with init_empty_weights(include_buffers=False):
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', UserWarning)
                        auto_model_cls.from_pretrained(
                            pretrained_model_name_or_path,
                            trust_remote_code=trust_remote_code,
                            use_auth_token=use_auth_token,
                            attn_implementation=
                            requested_attention_implementation,
                            config=config,
                        )
            else:
                with init_empty_weights(include_buffers=False):
                    auto_model_cls.from_config(
                        config,
                        trust_remote_code=trust_remote_code,
                        attn_implementation=requested_attention_implementation,
                    )

        dist.barrier()

        # initialize the model on the correct device
        if resolved_init_device == 'cpu':
            if pretrained:
                model = auto_model_cls.from_pretrained(
                    pretrained_model_name_or_path,
                    trust_remote_code=trust_remote_code,
                    use_auth_token=use_auth_token,
                    load_in_8bit=load_in_8bit,
                    attn_implementation=requested_attention_implementation,
                    config=config,
                )
            else:
                model = auto_model_cls.from_config(
                    config,
                    trust_remote_code=trust_remote_code,
                    attn_implementation=requested_attention_implementation,
                )
        elif resolved_init_device == 'meta':
            if pretrained:
                raise ValueError(
                    'Setting cfg.pretrained=True is not supported when init_device="meta".',
                )
            with init_empty_weights(include_buffers=False):
                model = auto_model_cls.from_config(
                    config,
                    trust_remote_code=trust_remote_code,
                    attn_implementation=requested_attention_implementation,
                )
        else:
            raise ValueError(
                f'init_device="{init_device}" must be either "cpu" or "meta".',
            )

        signal_file_path = dist.get_node_signal_file_name()
        if dist.get_local_rank() == 0:
            with open(signal_file_path, 'wb') as f:
                f.write(b'local_rank0_completed_download')

        # Avoid the collective call until the local rank zero has finished trying to download the checkpoint
        # so that we don't timeout for large downloads. This syncs all processes on the node
        with dist.local_rank_zero_download_and_wait(signal_file_path):
            # Then, wait to ensure every node has finished downloading the checkpoint
            dist.barrier()

        if dist.get_local_rank() == 0:
            os.remove(signal_file_path)

        # Use the pretrained generation config for the model if it exists.
        try:
            model.generation_config = GenerationConfig.from_pretrained(
                pretrained_model_name_or_path,
                use_auth_token=use_auth_token,
            )
        except OSError:
            log.warning(
                f'No existing generation config found for the model with name or path {pretrained_model_name_or_path}. Using default generation config.',
            )

        # Hugging Face's weight tying does not succeed if the model is inited on meta device
        # so we manually apply the weight tying here
        if model.config.tie_word_embeddings and resolved_init_device == 'meta':
            model.tie_weights()

        if pretrained_lora_id_or_path is not None:
            if not peft_installed:
                raise ValueError(
                    'PEFT is not installed, but lora_id_or_path was passed. Please install LLM Foundry with the peft extra to use lora_id_or_path.',
                )
            from peft import PeftModelForCausalLM
            model = PeftModelForCausalLM.from_pretrained(
                model,
                pretrained_lora_id_or_path,
                is_trainable=True,
            )

        if prepare_for_fsdp:
            cls.prepare_inner_model(model, init_device)

        return model

    def get_peft_config(self, peft_config_dict: dict[str, Any]) -> 'PeftConfig':
        if peft_installed:
            from peft import LoraConfig
            peft_type = peft_config_dict.get('peft_type', '')
            if peft_type.upper() != 'LORA':
                raise ValueError(
                    f'Only LORA is supported for peft_type, but got {peft_type}.',
                )
            task_type = peft_config_dict.get('task_type', '')
            if task_type.upper() != 'CAUSAL_LM':
                raise ValueError(
                    f'Only CAUSAL_LM is supported for task_type, but got {task_type}.',
                )
            return LoraConfig(**peft_config_dict)
        else:
            raise ValueError(
                'PEFT is not installed, but peft_config was passed. Please install LLM Foundry with the peft extra to use peft_config.',
            )

    @staticmethod
    def prepare_inner_model(
        model: Union[transformers.PreTrainedModel, 'PeftModel'],
        init_device: Optional[str] = None,
    ):
        """Prepare the inner model for FSDP wrapping.

        Args:
            model: The model to prepare.
            init_device: The device to initialize the model on.
        """
        # Note: We need to add the FSDP related attributes to the model AFTER the super init,
        # so that the (possible) embedding resizing doesn't destroy them
        prepare_hf_model_for_fsdp(model, init_device)

        # This provides support for meta initialization when using FSDP
        model.param_init_fn = lambda module: model._init_weights(module)
