# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Implements a Hugging Causal LM wrapped inside a :class:`.ComposerModel`."""

import logging
import os
import warnings
from typing import TYPE_CHECKING, Any, Dict, Mapping

from composer.models.huggingface import peft_installed
from composer.utils import dist
from omegaconf import DictConfig
from transformers import (AutoConfig, AutoModelForCausalLM, PretrainedConfig,
                          PreTrainedModel, PreTrainedTokenizerBase)

from llmfoundry.metrics import (DEFAULT_CAUSAL_LM_EVAL_METRICS,
                                DEFAULT_CAUSAL_LM_TRAIN_METRICS)
from llmfoundry.models.hf.hf_fsdp import hf_get_init_device
from llmfoundry.models.hf.model_wrapper import HuggingFaceModelWithFSDP
from llmfoundry.models.layers.attention import is_flash_v2_installed
from llmfoundry.models.utils import init_empty_weights
from llmfoundry.utils.config_utils import pop_config

if TYPE_CHECKING:
    from peft import PeftConfig

__all__ = ['ComposerHFCausalLM']

log = logging.getLogger(__name__)


class ComposerHFCausalLM(HuggingFaceModelWithFSDP):
    """Configures a :class:`.HuggingFaceModel` around a Causal LM.

    Args:
        om_model_config (DictConfig): An OmegaConf DictConfig specifying the configuration options
            cfg.pretrained_model_name_or_path (str): The name of or local path to
                the HF Causal LM (e.g., `gpt2` to instantiate a GPT2LMHeadModel).
            cfg.config_overrides (dict, optional): An optional dictionary of keyword
                arguments that override the default configuration associated with
                cfg.pretrained_model_name_or_path.
            cfg.pretrained (bool): Whether to instantiate the model with pre-trained
                weights coming from cfg.pretrained_model_name_or_path. If ``True``,
                cfg.config_overrides must be compatible with the pre-trained weights.
            cfg.init_device ('cpu' | 'meta'): Which device, 'cpu' or 'meta', to
                initialize the model on. Currently, `meta` is only supported when
                cfg.pretrained is ``False``. Default: ``'cpu'``.
            cfg.peft_config (dict, optional): An optional dictionary of keyword arguments to be
                passed to the PeftConfig constructor. If provided, the model will be wrapped in a PeftModel.
            cfg.trust_remote_code (bool, optional): Whether to trust remote code when loading from Hugging Face
                Hub. Default: ``True``.
            cfg.use_auth_token (bool, optional): Whether to use the Hugging Face authentication token when
                loading from Hugging Face Hub. Default: ``False``.
            cfg.use_train_metrics (bool, optional): Whether to use training metrics. Default: ``True``.
            cfg.load_in_8bit (bool, optional): Whether to load the model in 8-bit mode. Default: ``False``.
            cfg.init_device (str, optional): Which device to initialize the model on. Default: ``'cpu'``.
            cfg.use_flash_attention_2 (bool, optional): Whether to use flash-attention 2. Default: ``False``.
        tokenizer (PreTrainedTokenizer): The tokenizer that the model will use.
    """

    def __init__(self, om_model_config: DictConfig,
                 tokenizer: PreTrainedTokenizerBase):
        from llmfoundry.utils.builders import build_metric

        pretrained_model_name_or_path = om_model_config.pretrained_model_name_or_path
        pretrained_lora_id_or_path = om_model_config.get(
            'pretrained_lora_id_or_path', None)

        if not om_model_config.get(
                'trust_remote_code', True
        ) and pretrained_model_name_or_path.startswith('mosaicml/mpt'):
            raise ValueError(
                'trust_remote_code must be set to True for MPT models. Without this, the MPT model code will come from the transformers library, '
                +
                'which is significantly slower and not compatible with the LLM foundry training code, rather than the code release by MosaicML.'
            )

        # Set up Hugging Face args
        trust_remote_code = om_model_config.get('trust_remote_code', True)
        use_auth_token = om_model_config.get('use_auth_token', False)
        use_flash_attention_2 = om_model_config.get('use_flash_attention_2',
                                                    False)
        load_in_8bit = om_model_config.get('load_in_8bit', False)

        # Set up config args for the model construction and base classes
        init_device = om_model_config.get('init_device', 'cpu')
        # Resolve "mixed" init device to either "cpu" or "meta"
        resolved_init_device = hf_get_init_device(init_device)
        requested_attention_implementation = 'flash_attention_2' if use_flash_attention_2 else 'eager'

        if use_flash_attention_2 and not is_flash_v2_installed():
            raise ValueError(
                'use_flash_attention_2 is set to True, but flash-attention 2 is not installed. '
                + 'Please `pip install llm-foundry[gpu]`.')

        peft_config_dict = pop_config(om_model_config,
                                      'peft_config',
                                      must_exist=False,
                                      convert=True)
        if peft_config_dict is not None and not peft_installed:
            raise ValueError(
                'PEFT is not installed, but peft_config was passed. Please install LLM Foundry with the peft extra to use peft_config.'
            )

        use_train_metrics = om_model_config.get('use_train_metrics', True)
        train_metric_names = DEFAULT_CAUSAL_LM_TRAIN_METRICS + om_model_config.get(
            'additional_train_metrics', [])
        train_metrics = [
            build_metric(metric, {}) for metric in train_metric_names
        ] if use_train_metrics else []
        eval_metric_names = DEFAULT_CAUSAL_LM_EVAL_METRICS + om_model_config.get(
            'additional_eval_metrics', [])
        eval_metrics = [
            build_metric(metric, {}) for metric in eval_metric_names
        ]

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
                **kwargs):  # type: ignore
            config._attn_implementation = requested_attention_implementation
            return config

        PreTrainedModel._autoset_attn_implementation = classmethod(
            _autoset_attn_implementation_monkeypatch)

        # set config overrides
        for k, v in om_model_config.get('config_overrides', {}).items():
            if not hasattr(config, k):
                raise ValueError(
                    f'config does not have attribute "{k}" to override ({k}: {v}).'
                )

            attr = getattr(config, k)
            # attempt to disallow typos in nested configs
            if isinstance(attr, Mapping):
                extra_keys = [_k for _k in v.keys() if _k not in attr.keys()]
                if extra_keys:
                    raise ValueError(
                        f'Config dict override got unknown keys. ' +
                        f'Extra keys: {extra_keys}. ' +
                        f'Expected (a subset of) keys: {list(attr.keys())}.')
                getattr(config, k).update(v)
            # necessary case to allow for rope_scaling to be overriden in llama config
            elif attr is None and isinstance(v, Mapping):
                setattr(config, k, {})
                getattr(config, k).update(v)
            elif isinstance(attr, PretrainedConfig):
                if not isinstance(v, Mapping):
                    raise ValueError(
                        f'Expected a dictionary for config override {k}, but got {v}.'
                    )

                for _k, _v in v.items():
                    if not hasattr(attr, _k):
                        raise ValueError(
                            f'config does not have attribute "{_k}" to override ({k}: {_k}: {_v}).'
                        )
                    setattr(attr, _k, _v)
            else:
                setattr(config, k, v)

        # We need to have all non-zero local ranks be not-pretrained
        # Rank 0 will still be pretrained, and distribute the weights appropriately
        if dist.get_local_rank() != 0 and init_device == 'mixed':
            om_model_config.pretrained = False

        # If the HuggingFace model is coming from a local folder, Hugging Face copies the modules into the
        # transformers modules cache. On particular systems, this operation seems to cause contention between
        # the different processes. To avoid this contention, we first create the model (on meta device) on local rank
        # zero. This will set up the transformers model cache and avoid the future contention.
        # if dist.get_local_rank() == 0 and os.path.isdir( # doesnt work with quantization
        #         pretrained_model_name_or_path):
        #     with init_empty_weights(include_buffers=False):
        #         with warnings.catch_warnings():
        #             warnings.simplefilter('ignore', UserWarning)
        #             AutoModelForCausalLM.from_pretrained(
        #                 pretrained_model_name_or_path,
        #                 trust_remote_code=trust_remote_code,
        #                 use_auth_token=use_auth_token,
        #                 config=config,
        #             )

        dist.barrier()

        # initialize the model on the correct device
        if resolved_init_device == 'cpu':
            if om_model_config.pretrained:
                device_map = None
                pre_quantized = config.to_dict().get('quantization_config', None)
                if pre_quantized and pre_quantized['quant_method'] == "gptq":
                    device_map = 'auto'
                quantization_config = om_model_config.get('quantization_config', None)
                if quantization_config:
                    from transformers.quantizers.auto import AutoQuantizationConfig
                    quantization_config = AutoQuantizationConfig.from_dict(quantization_config)
                model = AutoModelForCausalLM.from_pretrained(
                    pretrained_model_name_or_path,
                    trust_remote_code=trust_remote_code,
                    use_auth_token=use_auth_token,
                    load_in_8bit=load_in_8bit,
                    config=config,
                    device_map=device_map,
                    quantization_config=quantization_config,
                )
            else:
                model = AutoModelForCausalLM.from_config(
                    config,
                    trust_remote_code=trust_remote_code,
                )
        elif resolved_init_device == 'meta':
            if om_model_config.pretrained:
                raise ValueError(
                    'Setting cfg.pretrained=True is not supported when init_device="meta".'
                )
            with init_empty_weights(include_buffers=False):
                model = AutoModelForCausalLM.from_config(
                    config,
                    trust_remote_code=trust_remote_code,
                )
        else:
            raise ValueError(
                f'init_device="{init_device}" must be either "cpu" or "meta".')

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

        peft_config = None
        if peft_config_dict is not None:
            peft_config = self._get_peft_config(peft_config_dict)

        if pretrained_lora_id_or_path is not None:
            if not peft_installed:
                raise ValueError(
                    'PEFT is not installed, but lora_id_or_path was passed. Please install LLM Foundry with the peft extra to use lora_id_or_path.'
                )
            from peft import PeftModelForCausalLM
            model = PeftModelForCausalLM.from_pretrained(
                model, pretrained_lora_id_or_path)

        super().__init__(
            model=model,
            shift_labels=True,
            tokenizer=tokenizer,
            metrics=train_metrics,
            eval_metrics=eval_metrics,
            init_device=init_device,
            peft_config=peft_config,
        )

    @staticmethod
    def _get_peft_config(peft_config_dict: Dict[str, Any]) -> 'PeftConfig':
        if peft_installed:
            from peft import LoraConfig
            peft_type = peft_config_dict.get('peft_type', '')
            if peft_type.upper() != 'LORA':
                raise ValueError(
                    f'Only LORA is supported for peft_type, but got {peft_type}.'
                )
            task_type = peft_config_dict.get('task_type', '')
            if task_type.upper() != 'CAUSAL_LM':
                raise ValueError(
                    f'Only CAUSAL_LM is supported for task_type, but got {task_type}.'
                )
            return LoraConfig(**peft_config_dict)
        else:
            raise ValueError(
                'PEFT is not installed, but peft_config was passed. Please install LLM Foundry with the peft extra to use peft_config.'
            )
