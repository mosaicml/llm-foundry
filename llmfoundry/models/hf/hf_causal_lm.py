# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Implements a Hugging Causal LM wrapped inside a :class:`.ComposerModel`."""

from typing import Mapping, Union

from composer.metrics.nlp import (InContextLearningLMAccuracy,
                                  InContextLearningLMExpectedCalibrationError,
                                  InContextLearningMCExpectedCalibrationError,
                                  InContextLearningMultipleChoiceAccuracy,
                                  InContextLearningQAAccuracy,
                                  LanguageCrossEntropy, LanguagePerplexity)
from omegaconf import DictConfig
from transformers import (AutoConfig, AutoModelForCausalLM, PreTrainedTokenizer,
                          PreTrainedTokenizerFast)

from llmfoundry.models.hf.model_wrapper import HuggingFaceModelWithZLoss
from llmfoundry.models.utils import init_empty_weights

__all__ = ['ComposerHFCausalLM']

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


class ComposerHFCausalLM(HuggingFaceModelWithZLoss):
    """Configures a :class:`.HuggingFaceModel` around a Causal LM.

    Args:
        cfg (DictConfig): An omegaconf dictionary used to configure the model:
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
        tokenizer (PreTrainedTokenizer): The tokenizer that the model will use.
    """

    def __init__(self, om_model_config: DictConfig, tokenizer: Tokenizer):
        trust_remote_code = om_model_config.get('trust_remote_code', True)
        use_auth_token = om_model_config.get('use_auth_token', False)
        config = AutoConfig.from_pretrained(
            om_model_config.pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            use_auth_token=use_auth_token,
        )

        # set config overrides
        for k, v in om_model_config.get('config_overrides', {}).items():
            if not hasattr(config, k):
                raise ValueError(
                    f'config does not have attribute "{k}" to override ({k}: {v}).'
                )

            attr = getattr(config, k)
            if isinstance(attr, Mapping):
                extra_keys = [_k for _k in v.keys() if _k not in attr.keys()]
                if extra_keys:
                    raise ValueError(
                        f'Config dict override got unknown keys. '
                        f'Extra keys: {extra_keys}. '
                        f'Expected (a subset of) keys: {list(attr.keys())}.')
                getattr(config, k).update(v)
            else:
                setattr(config, k, v)

        train_metrics = [
            LanguageCrossEntropy(),
            LanguagePerplexity(),
        ]
        eval_metrics = [
            LanguageCrossEntropy(),
            LanguagePerplexity(),
            InContextLearningLMAccuracy(),
            InContextLearningMultipleChoiceAccuracy(),
            InContextLearningQAAccuracy(),
            InContextLearningLMExpectedCalibrationError(),
            InContextLearningMCExpectedCalibrationError()
        ]

        init_device = om_model_config.get('init_device', 'cpu')
        if init_device == 'cpu':
            if om_model_config.pretrained:
                model = AutoModelForCausalLM.from_pretrained(
                    om_model_config.pretrained_model_name_or_path,
                    trust_remote_code=trust_remote_code,
                    use_auth_token=use_auth_token,
                    config=config)
            else:
                model = AutoModelForCausalLM.from_config(
                    config,
                    trust_remote_code=trust_remote_code,
                )
        elif init_device == 'meta':
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

        composer_model = super().__init__(model=model,
                                          shift_labels=True,
                                          tokenizer=tokenizer,
                                          metrics=train_metrics,
                                          eval_metrics=eval_metrics,
                                          z_loss=om_model_config.get(
                                              'z_loss', 0.0))

        return composer_model
