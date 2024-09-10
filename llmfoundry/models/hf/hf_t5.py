# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Implements a Hugging Face T5 wrapped inside a :class:`.ComposerModel`."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Union

from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.models.auto.auto_factory import _BaseAutoModelClass

from llmfoundry.metrics import DEFAULT_ENC_DEC_METRICS
from llmfoundry.models.hf.hf_base import BaseHuggingFaceModel
from llmfoundry.utils.warnings import experimental_class

__all__ = ['ComposerHFT5']


@experimental_class('ComposerHFT5')
class ComposerHFT5(BaseHuggingFaceModel):
    """Configures a :class:`.HuggingFaceModel` around a T5.

    Note: This function uses `transformers.T5ForConditionalGeneration`. Future releases
        will expand support to more general classes of HF Encoder-Decoder models.

    Args:
        pretrained_model_name_or_path (str): The name of or local path to
            the HF model (e.g., `t5-base` to instantiate a T5 using the base config).
        config_overrides (dict, optional): An optional dictionary of keyword
            arguments that override the default configuration associated with
            cfg.pretrained_model_name_or_path. Default: ``{}``.
        pretrained (bool): Whether to instantiate the model with pre-trained
            weights coming from cfg.pretrained_model_name_or_path. If ``True``,
            cfg.config_overrides must be compatible with the pre-trained weights.
        init_device ('cpu' | 'meta'): Which device, 'cpu' or 'meta', to
            initialize the model on. Currently, `meta` is only supported when
            cfg.pretrained is ``False``. Default: ``'cpu'``.
        tokenizer (PreTrainedTokenizer): The tokenizer that the model will use.
    """

    model_cls: Union[_BaseAutoModelClass,
                     PreTrainedModel] = AutoModelForSeq2SeqLM
    default_train_metrics: tuple = tuple(DEFAULT_ENC_DEC_METRICS)
    default_eval_metrics: tuple = ()

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        pretrained_model_name_or_path: str,
        pretrained: bool = True,
        trust_remote_code: bool = True,
        use_auth_token: bool = False,
        config_overrides: Optional[dict[str, Any]] = None,
        init_device: str = 'cpu',
        additional_train_metrics: Optional[list] = None,
        name: Optional[str] = None,
    ):
        super().__init__(
            pretrained_model_name_or_path,
            tokenizer=tokenizer,
            pretrained=pretrained,
            trust_remote_code=trust_remote_code,
            use_auth_token=use_auth_token,
            init_device=init_device,
            config_overrides=config_overrides,
            shift_labels=True,
            additional_train_metrics=additional_train_metrics,
        )

    @classmethod
    def build_config(
        cls,
        pretrained_model_name_or_path: str,
        trust_remote_code: bool,
        use_auth_token: bool,
        attn_implementation: str,
        config_overrides: dict[str, Any],
        **kwargs: Any,
    ) -> PretrainedConfig:
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            use_auth_token=use_auth_token,
        )

        # set config overrides
        for k, v in config_overrides.items():
            if not hasattr(config, k):
                raise ValueError(
                    f'config does not have attribute "{k}" to override ({k}: {v}).',
                )

            attr = getattr(config, k)
            if isinstance(attr, Mapping):
                extra_keys = [_k for _k in v.keys() if _k not in attr.keys()]
                if extra_keys:
                    raise ValueError(
                        f'Config dict override got unknown keys. ' +
                        f'Extra keys: {extra_keys}. ' +
                        f'Expected (a subset of) keys: {list(attr.keys())}.',
                    )
                getattr(config, k).update(v)
            else:
                setattr(config, k, v)

        if not config.is_encoder_decoder:
            raise ValueError(f'Model type "hf_t5" currently only supports T5 models ' +\
                             f'using configs where `is_encoder_decoder` is ``True``.')

        return config
