# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Implements a Hugging Causal LM wrapped inside a :class:`.ComposerModel`."""

import logging
import os
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

from composer.models.huggingface import peft_installed
from composer.utils import dist
from torchmetrics import Metric
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from llmfoundry.metrics import (
    DEFAULT_CAUSAL_LM_EVAL_METRICS,
    DEFAULT_CAUSAL_LM_TRAIN_METRICS,
)
from llmfoundry.models.hf.hf_fsdp import hf_get_init_device
from llmfoundry.models.hf.model_wrapper import HuggingFaceModelWithFSDP
from llmfoundry.models.layers.attention import is_flash_v2_installed
from llmfoundry.models.utils import init_empty_weights

if TYPE_CHECKING:
    from peft import PeftConfig, PeftModel

__all__ = ['ComposerHFCausalLM']

log = logging.getLogger(__name__)


class ComposerHFCausalLM(HuggingFaceModelWithFSDP):
    """Configures a :class:`.HuggingFaceModel` around a Causal LM.

    Args:
        pretrained_model_name_or_path (str): The name of or local path to
            the HF Causal LM (e.g., `gpt2` to instantiate a GPT2LMHeadModel).
        config_overrides (dict, optional): An optional dictionary of keyword
            arguments that override the default configuration associated with
            cfg.pretrained_model_name_or_path.
        pretrained (bool): Whether to instantiate the model with pre-trained
            weights coming from cfg.pretrained_model_name_or_path. If ``True``,
            cfg.config_overrides must be compatible with the pre-trained weights.
        init_device ('cpu' | 'meta'): Which device, 'cpu' or 'meta', to
            initialize the model on. Currently, `meta` is only supported when
            cfg.pretrained is ``False``. Default: ``'cpu'``.
        peft_config (dict, optional): An optional dictionary of keyword arguments to be
            passed to the PeftConfig constructor. If provided, the model will be wrapped in a PeftModel.
        trust_remote_code (bool, optional): Whether to trust remote code when loading from Hugging Face
            Hub. Default: ``True``.
        use_auth_token (bool, optional): Whether to use the Hugging Face authentication token when
            loading from Hugging Face Hub. Default: ``False``.
        use_train_metrics (bool, optional): Whether to use training metrics. Default: ``True``.
        load_in_8bit (bool, optional): Whether to load the model in 8-bit mode. Default: ``False``.
        init_device (str, optional): Which device to initialize the model on. Default: ``'cpu'``.
        use_flash_attention_2 (bool, optional): Whether to use flash-attention 2. Default: ``False``.
        tokenizer (PreTrainedTokenizer): The tokenizer that the model will use.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        pretrained_model_name_or_path: str,
        pretrained: bool = True,
        pretrained_lora_id_or_path: Optional[str] = None,
        trust_remote_code: bool = True,
        use_auth_token: bool = False,
        use_flash_attention_2: bool = False,
        load_in_8bit: bool = False,
        init_device: str = 'cpu',
        config_overrides: Optional[Dict[str, Any]] = None,
        peft_config: Optional[Dict[str, Any]] = None,
        use_train_metrics: bool = True,
        additional_train_metrics: Optional[List] = None,
        additional_eval_metrics: Optional[List] = None,
    ):

        config_overrides = config_overrides or {}

        model = ComposerHFCausalLM.build_inner_model(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            pretrained_lora_id_or_path=pretrained_lora_id_or_path,
            trust_remote_code=trust_remote_code,
            init_device=init_device,
            use_flash_attention_2=use_flash_attention_2,
            use_auth_token=use_auth_token,
            config_overrides=config_overrides,
            load_in_8bit=load_in_8bit,
            pretrained=pretrained,
            prepare_for_fsdp=True,
        )

        train_metrics, eval_metrics = ComposerHFCausalLM.build_metrics(
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
            peft_config_object = self._get_peft_config(peft_config)

        # Set up config args for the model construction and base classes
        super().__init__(
            model=model,
            shift_labels=True,
            tokenizer=tokenizer,
            metrics=train_metrics,
            eval_metrics=eval_metrics,
            init_device=init_device,
            peft_config=peft_config_object,
        )

    @staticmethod
    def build_metrics(
        use_train_metrics: bool,
        additional_train_metrics: Optional[List[str]] = None,
        additional_eval_metrics: Optional[List[str]] = None,
    ) -> Tuple[List[Metric], List[Metric]]:
        """Builds the training and evaluation metrics for the model.

        Args:
            use_train_metrics (bool): Whether to use training metrics.
            additional_train_metrics (Optional[List[str]]): Additional training metrics to include.
            additional_eval_metrics (Optional[List[str]]): Additional evaluation metrics to include.

        Returns:
            Tuple[List[Metric], List[Metric]]: A tuple containing the list of training metrics and evaluation metrics.
        """
        from llmfoundry.utils.builders import build_metric

        train_metric_names = DEFAULT_CAUSAL_LM_TRAIN_METRICS + (
            additional_train_metrics or []
        )
        train_metrics = [
            build_metric(metric, {}) for metric in train_metric_names
        ] if use_train_metrics else []
        eval_metric_names = DEFAULT_CAUSAL_LM_EVAL_METRICS + (
            additional_eval_metrics or []
        )
        eval_metrics = [
            build_metric(metric, {}) for metric in eval_metric_names
        ]

        return train_metrics, eval_metrics

    @staticmethod
    def build_inner_model(
        pretrained_model_name_or_path: str,
        pretrained_lora_id_or_path: Optional[str],
        trust_remote_code: bool,
        init_device: str,
        use_flash_attention_2: bool,
        use_auth_token: bool,
        config_overrides: Dict[str, Any],
        load_in_8bit: bool,
        pretrained: bool,
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
            prepare_for_fsdp (bool, optional): Whether to prepare the model for FSDP wrapping. Default: False.

        Returns:
            Union[PreTrainedModel, 'PeftModel']: The built inner model.
            prepare_for_fsdp (bool): Whether to prepare the model for FSDP wrapping. Default: ``False``.
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

        # Construct the Hugging Face config to use
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            use_auth_token=use_auth_token,
            attn_implementation=requested_attention_implementation,
            use_cache=
            False,  # Necessary due to https://github.com/huggingface/transformers/issues/28056
        )

        # This is not ideal, however Hugging Face's _autoset_attn_implementation function
        # forces you to load the model in fp16/bf16 if you want to use flash attention. Rather than loading
        # the model and then casting it back to fp32, we are monkeypatching their check.
        # https://github.com/huggingface/transformers/issues/28052
        def _autoset_attn_implementation_monkeypatch(
            cls,  # type: ignore
            config,  # type: ignore
            *args,  # type: ignore
            **kwargs,  # type: ignore
        ):  # type: ignore
            config._attn_implementation = requested_attention_implementation
            return config

        PreTrainedModel._autoset_attn_implementation = classmethod(
            _autoset_attn_implementation_monkeypatch,
        )

        # set config overrides
        for k, v in config_overrides.items():
            if not hasattr(config, k):
                raise ValueError(
                    f'config does not have attribute "{k}" to override ({k}: {v}).',
                )

            attr = getattr(config, k)
            # attempt to disallow typos in nested configs
            if isinstance(attr, Mapping):
                extra_keys = [_k for _k in v.keys() if _k not in attr.keys()]
                if extra_keys:
                    raise ValueError(
                        f'Config dict override got unknown keys. ' +
                        f'Extra keys: {extra_keys}. ' +
                        f'Expected (a subset of) keys: {list(attr.keys())}.',
                    )
                getattr(config, k).update(v)
            # necessary case to allow for rope_scaling to be overriden in llama config
            elif attr is None and isinstance(v, Mapping):
                setattr(config, k, {})
                getattr(config, k).update(v)
            elif isinstance(attr, PretrainedConfig):
                if not isinstance(v, Mapping):
                    raise ValueError(
                        f'Expected a dictionary for config override {k}, but got {v}.',
                    )

                for _k, _v in v.items():
                    if not hasattr(attr, _k):
                        raise ValueError(
                            f'config does not have attribute "{_k}" to override ({k}: {_k}: {_v}).',
                        )
                    setattr(attr, _k, _v)
            else:
                setattr(config, k, v)

        if hasattr(config, 'attn_config') and config.attn_config.get(
            'seq_parallel_world_size',
            None,
        ) is not None:
            raise NotImplementedError(
                'Sequence Parallelism is not supported for HuggingFace models.',
            )

        # We need to have all non-zero local ranks be not-pretrained
        # Rank 0 will still be pretrained, and distribute the weights appropriately
        if dist.get_local_rank() != 0 and init_device == 'mixed':
            pretrained = False

        # If the HuggingFace model is coming from a local folder, Hugging Face copies the modules into the
        # transformers modules cache. On particular systems, this operation seems to cause contention between
        # the different processes. To avoid this contention, we first create the model (on meta device) on local rank
        # zero. This will set up the transformers model cache and avoid the future contention.
        if dist.get_local_rank(
        ) == 0 and os.path.isdir(pretrained_model_name_or_path):
            with init_empty_weights(include_buffers=False):
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)
                    AutoModelForCausalLM.from_pretrained(
                        pretrained_model_name_or_path,
                        trust_remote_code=trust_remote_code,
                        use_auth_token=use_auth_token,
                        config=config,
                    )

        dist.barrier()

        # initialize the model on the correct device
        if resolved_init_device == 'cpu':
            if pretrained:
                model = AutoModelForCausalLM.from_pretrained(
                    pretrained_model_name_or_path,
                    trust_remote_code=trust_remote_code,
                    use_auth_token=use_auth_token,
                    load_in_8bit=load_in_8bit,
                    config=config,
                )
            else:
                model = AutoModelForCausalLM.from_config(
                    config,
                    trust_remote_code=trust_remote_code,
                )
        elif resolved_init_device == 'meta':
            if pretrained:
                raise ValueError(
                    'Setting cfg.pretrained=True is not supported when init_device="meta".',
                )
            with init_empty_weights(include_buffers=False):
                model = AutoModelForCausalLM.from_config(
                    config,
                    trust_remote_code=trust_remote_code,
                )
        else:
            raise ValueError(
                f'init_device="{init_device}" must be either "cpu" or "meta".',
            )

        signal_file_path = f'.node_{dist.get_node_rank()}_local_rank0_completed'
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
            )

        if prepare_for_fsdp:
            ComposerHFCausalLM.prepare_inner_model(model, init_device)
        return model

    @staticmethod
    def _get_peft_config(peft_config_dict: Dict[str, Any]) -> 'PeftConfig':
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
