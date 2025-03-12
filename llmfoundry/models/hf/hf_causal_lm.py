# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Implements a Hugging Causal LM wrapped inside a :class:`.ComposerModel`."""

import logging
from typing import (
    Any,
    Optional,
    Union,
)

from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.models.auto.auto_factory import _BaseAutoModelClass

from llmfoundry.metrics import (
    DEFAULT_CAUSAL_LM_EVAL_METRICS,
    DEFAULT_CAUSAL_LM_TRAIN_METRICS,
)
from llmfoundry.models.hf.hf_base import BaseHuggingFaceModel

__all__ = ['ComposerHFCausalLM']

log = logging.getLogger(__name__)


class ComposerHFCausalLM(BaseHuggingFaceModel):
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

    model_cls: Union[_BaseAutoModelClass,
                     PreTrainedModel] = AutoModelForCausalLM
    default_train_metrics: tuple = tuple(DEFAULT_CAUSAL_LM_TRAIN_METRICS)
    default_eval_metrics: tuple = tuple(DEFAULT_CAUSAL_LM_EVAL_METRICS)

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
        config_overrides: Optional[dict[str, Any]] = None,
        peft_config: Optional[dict[str, Any]] = None,
        use_train_metrics: bool = True,
        allow_embedding_resizing: bool = False,
        additional_train_metrics: Optional[list] = None,
        additional_eval_metrics: Optional[list] = None,
        should_save_peft_only: bool = True,
    ):
        super().__init__(
            pretrained_model_name_or_path,
            tokenizer=tokenizer,
            pretrained=pretrained,
            pretrained_lora_id_or_path=pretrained_lora_id_or_path,
            trust_remote_code=trust_remote_code,
            use_auth_token=use_auth_token,
            use_flash_attention_2=use_flash_attention_2,
            load_in_8bit=load_in_8bit,
            init_device=init_device,
            config_overrides=config_overrides,
            shift_labels=True,
            peft_config=peft_config,
            allow_embedding_resizing=allow_embedding_resizing,
            use_train_metrics=use_train_metrics,
            additional_train_metrics=additional_train_metrics,
            additional_eval_metrics=additional_eval_metrics,
            should_save_peft_only=should_save_peft_only,
        )
