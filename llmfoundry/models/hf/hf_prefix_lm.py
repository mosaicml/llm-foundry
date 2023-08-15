# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Implements a Hugging Prefix LM wrapped inside a :class:`.ComposerModel`."""

from __future__ import annotations

from typing import Mapping, MutableMapping

from composer.metrics.nlp import LanguageCrossEntropy, MaskedAccuracy
from composer.utils import dist
from omegaconf import DictConfig
from transformers import (AutoConfig, AutoModelForCausalLM,
                          PreTrainedTokenizerBase)

from llmfoundry.models.hf.hf_fsdp import hf_get_init_device
from llmfoundry.models.hf.model_wrapper import HuggingFaceModelWithZLoss
from llmfoundry.models.utils import (adapt_tokenizer_for_denoising,
                                     add_bidirectional_mask_if_missing,
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
        tokenizer (PreTrainedTokenizer): The tokenizer that the model will use.
    """

    def __init__(self, om_model_config: DictConfig,
                 tokenizer: PreTrainedTokenizerBase):
        config = AutoConfig.from_pretrained(
            om_model_config.pretrained_model_name_or_path,
            trust_remote_code=om_model_config.get('trust_remote_code', True),
            use_auth_token=om_model_config.get('use_auth_token', False),
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
                        f'Config dict override got unknown keys. ' +
                        f'Extra keys: {extra_keys}. ' +
                        f'Expected (a subset of) keys: {list(attr.keys())}.')
                getattr(config, k).update(v)
            else:
                setattr(config, k, v)

        # Set up the tokenizer (add tokens for denoising sentinels if needed)
        if om_model_config.get('adapt_vocab_for_denoising', False):
            adapt_tokenizer_for_denoising(tokenizer)

        init_device = om_model_config.get('init_device', 'cpu')

        # Get the device we want to initialize, and use the
        # resolved version to initialize the HF model
        resolved_init_device = hf_get_init_device(init_device)

        # We need to have all non-zero local ranks be not-pretrained
        # Rank 0 will still be pretrained, and distribute the weights appropriately
        if dist.get_local_rank() != 0 and init_device == 'mixed':
            om_model_config.pretrained = False

        if resolved_init_device == 'cpu':
            if om_model_config.pretrained:
                model = AutoModelForCausalLM.from_pretrained(
                    om_model_config.pretrained_model_name_or_path,
                    config=config)
            else:
                model = AutoModelForCausalLM.from_config(config)
        elif resolved_init_device == 'meta':
            if om_model_config.pretrained:
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
            LanguageCrossEntropy(ignore_index=_HF_IGNORE_INDEX),
            MaskedAccuracy(ignore_index=_HF_IGNORE_INDEX)
        ]

        composer_model = super().__init__(model=model,
                                          shift_labels=True,
                                          tokenizer=tokenizer,
                                          metrics=metrics,
                                          z_loss=om_model_config.get(
                                              'z_loss', 0.0),
                                          init_device=init_device)

        return composer_model

    def forward(self, batch: MutableMapping):
        # Add bidirectional_mask if it is missing and can be constructed
        add_bidirectional_mask_if_missing(batch)
        return super().forward(batch)
