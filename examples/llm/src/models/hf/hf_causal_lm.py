# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""Implements a Hugging Causal LM wrapped inside a :class:`.ComposerModel`."""

from typing import Optional

from composer.metrics.nlp import (InContextLearningLMAccuracy,
                                  InContextLearningMultipleChoiceAccuracy,
                                  LanguageCrossEntropy, LanguagePerplexity)
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from examples.llm.src.models.hf.model_wrapper import HuggingFaceModelWithZLoss
from examples.llm.src.models.utils import init_empty_weights

__all__ = ['ComposerHFCausalLM']


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
            cfg.add_exact_match (bool, optional): CURRENTLY UNUSED. Whether to add ExactMatch metric used
                in some fine-tuning settings. Default: ``False``.
            cfg.add_rouge (bool, optional): CURRENTLY UNUSED. Whether to add RougeWithDetokenizer metric
                to validation metrics. Default: ``False``.
    """

    def __init__(self,
                 om_model_config: DictConfig,
                 om_tokenizer_config: Optional[DictConfig] = None):
        config = AutoConfig.from_pretrained(
            om_model_config.pretrained_model_name_or_path,
            **om_model_config.get('config_overrides', {}))

        resolved_om_tokenizer_config = om.to_container(om_tokenizer_config,
                                                       resolve=True)
        tokenizer_kwargs = resolved_om_tokenizer_config.get(  # type: ignore
            'kwargs', {})
        tokenizer_name = resolved_om_tokenizer_config['name']  # type: ignore
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,
                                                  **tokenizer_kwargs)

        train_metrics = [
            LanguageCrossEntropy(len(tokenizer)),
            LanguagePerplexity(len(tokenizer)),
        ]
        eval_metrics = [
            LanguageCrossEntropy(len(tokenizer)),
            LanguagePerplexity(len(tokenizer)),
            InContextLearningLMAccuracy(),
            InContextLearningMultipleChoiceAccuracy(),
        ]

        init_device = om_model_config.get('init_device', 'cpu')
        if init_device == 'cpu':
            if om_model_config.pretrained:
                model = AutoModelForCausalLM.from_pretrained(
                    om_model_config.pretrained_model_name_or_path,
                    config=config)
            else:
                model = AutoModelForCausalLM.from_config(config)
        elif init_device == 'meta':
            if om_model_config.pretrained:
                raise ValueError(
                    'Setting cfg.pretrained=True is not supported when init_device="meta".'
                )
            with init_empty_weights(include_buffers=False):
                model = AutoModelForCausalLM.from_config(config)
        else:
            raise ValueError(
                f'init_device="{init_device}" must be either "cpu" or "meta".')

        # if cfg.add_exact_match:
        #     metrics.append(ExactMatch(ignore_index=_HF_IGNORE_INDEX))

        composer_model = super().__init__(model=model,
                                          tokenizer=tokenizer,
                                          metrics=train_metrics,
                                          eval_metrics=eval_metrics,
                                          z_loss=om_model_config.get(
                                              'z_loss', 0.0))

        # if cfg.add_rouge:
        #     rouge_metric = RougeWithDetokenizer(detokenizer=tokenizer)
        #     composer_model.val_metrics[RougeWithDetokenizer.__name__] = rouge_metric

        return composer_model
