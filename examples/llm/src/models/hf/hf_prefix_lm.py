# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""Implements a Hugging Prefix LM wrapped inside a :class:`.ComposerModel`."""

from __future__ import annotations

import torch
from composer.metrics.nlp import LanguageCrossEntropy, MaskedAccuracy
from omegaconf import DictConfig
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from examples.llm.src.models.hf.model_wrapper import HuggingFaceModelWithZLoss
from examples.llm.src.models.utils import (AutoTokenizerForMOD,
                                           convert_hf_causal_lm_to_prefix_lm,
                                           init_empty_weights)

__all__ = ['ComposerHFPrefixLM']

# HuggingFace hardcodes the ignore index to -100
_HF_IGNORE_INDEX = -100


class ComposerHFPrefixLM(HuggingFaceModelWithZLoss):
    """Configures a :class:`.HuggingFaceModel` around a Prefix LM.

    Note: HuggingFace does not natively support Prefix LM-style models. This function uses
    `transformers.AutoModelForCausalLM` to instantiate a Causal LM, then uses a conversion utility
    to turn the model into a Prefix LM. Currently, that conversion utility only supports the
    following HuggingFace Causal LM types:
        - `GPT2LMHeadModel`
        - `GPTNeoForCausalLM`
        - `GPTNeoXForCausalLM`
        - `GPTJForCausalLM`
        - `BloomForCausalLM`
        - `OPTForCausalLM`

    Args:
        cfg (DictConfig): An omegaconf dictionary used to configure the model:
            cfg.pretrained_model_name_or_path (str): The name of or local path to
                the HF model (e.g., `gpt2` to instantiate a GPT2LMHeadModel). The model
                will be converted to a Prefix LM during initialization.
            cfg.config_overrides (dict, optional): An optional dictionary of keyword
                arguments that override the default configuration associated with
                cfg.pretrained_model_name_or_path. Default: ``{}``.
            cfg.pretrained (bool): Whether to instantiate the model with pre-trained
                weights coming from cfg.pretrained_model_name_or_path. If ``True``,
                cfg.config_overrides must be compatible with the pre-trained weights.
            cfg.init_device ('cpu' | 'meta'): Which device, 'cpu' or 'meta', to
                initialize the model on. Currently, `meta` is only supported when
                cfg.pretrained is ``False``. Default: ``'cpu'``.
            cfg.z_loss (float, optional): The coefficient of the z-loss. If >0.0, this
                the z-loss will be multiplied by this value before being added to the
                standard loss term. Default: ``0.0``.
            cfg.adapt_vocab_for_denoising (bool, optional):  Whether to adapt the vocab
                of the model/tokenizer to include sentinel tokens that are used in denoising
                tasks like Span Corruption. If you intend to load from an existing Composer
                checkpoint that was trained on such a task, set this to ``True`` to ensure
                that the model vocab size matches your checkpoint's vocab size when loading
                the weights. Default: ``False``.
            cfg.add_exact_match (bool, optional): CURRENTLY UNUSED. Whether to add ExactMatch metric used
                in some fine-tuning settings. Default: ``False``.
            cfg.add_rouge (bool, optional): CURRENTLY UNUSED. Whether to add RougeWithDetokenizer metric
                to validation metrics. Default: ``False``.
    """

    def __init__(self, cfg: DictConfig):
        config = AutoConfig.from_pretrained(cfg.pretrained_model_name_or_path,
                                            **cfg.get('config_overrides', {}))

        # Set up the tokenizer (add tokens for denoising sentinels if needed)
        if cfg.get('adapt_vocab_for_denoising', False):
            tokenizer = AutoTokenizerForMOD.from_pretrained(
                cfg.pretrained_model_name_or_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                cfg.pretrained_model_name_or_path)
        vocab_size = len(tokenizer)

        init_device = cfg.get('init_device', 'cpu')
        if init_device == 'cpu':
            if cfg.pretrained:
                model = AutoModelForCausalLM.from_pretrained(
                    cfg.pretrained_model_name_or_path, config=config)
            else:
                model = AutoModelForCausalLM.from_config(config)
        elif init_device == 'meta':
            if cfg.pretrained:
                raise ValueError(
                    'Setting cfg.pretrained=True is not supported when init_device="meta".'
                )
            with init_empty_weights(include_buffers=False):
                model = AutoModelForCausalLM.from_config(config)
        else:
            raise ValueError(
                f'init_device="{init_device}" must be either "cpu" or "meta".')

        # Convert the Causal LM into a Prefix LM via our custom wrapper
        model = convert_hf_causal_lm_to_prefix_lm(model)

        metrics = [
            LanguageCrossEntropy(vocab_size=vocab_size,
                                 ignore_index=_HF_IGNORE_INDEX),
            MaskedAccuracy(ignore_index=_HF_IGNORE_INDEX)
        ]

        # if cfg.add_exact_match:
        #     metrics.append(ExactMatch(ignore_index=_HF_IGNORE_INDEX))

        composer_model = super().__init__(model=model,
                                          tokenizer=tokenizer,
                                          metrics=metrics,
                                          z_loss=cfg.get('z_loss', 0.0))

        # if cfg.add_rouge:
        #     rouge_metric = RougeWithDetokenizer(detokenizer=tokenizer)
        #     composer_model.val_metrics[RougeWithDetokenizer.__name__] = rouge_metric

        return composer_model

    def forward(self, batch):
        # Add bidirectional_mask if it is missing and can be constructed
        if 'bidirectional_mask' not in batch:
            if batch.get('mode', None) == 'icl_task':
                batch['bidirectional_mask'] = batch['attention_mask'].clone()
                for i, continuation_indices in enumerate(
                        batch['continuation_indices']):
                    batch['bidirectional_mask'][i, continuation_indices] = 0
            elif ('labels' in batch) and ('attention_mask' in batch):
                batch['bidirectional_mask'] = torch.logical_and(
                    torch.eq(batch['attention_mask'], 1),
                    torch.eq(batch['labels'], _HF_IGNORE_INDEX),
                ).type_as(batch['attention_mask'])
            else:
                raise KeyError(
                    'No bidirectional_mask in batch and not sure how to construct one.'
                )
        return super().forward(batch)
