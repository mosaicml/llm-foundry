# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Implements a Hugging Face T5 wrapped inside a :class:`.ComposerModel`."""

from __future__ import annotations

from typing import List, Mapping, Optional

from composer.utils import dist
from omegaconf import OmegaConf as om
from transformers import (AutoConfig, PreTrainedTokenizerBase,
                          T5ForConditionalGeneration)

from llmfoundry.metrics import DEFAULT_ENC_DEC_METRICS
from llmfoundry.models.hf.hf_fsdp import hf_get_init_device
from llmfoundry.models.hf.model_wrapper import HuggingFaceModelWithFSDP
from llmfoundry.models.utils import init_empty_weights
from llmfoundry.utils.warnings import experimental_class

__all__ = ['ComposerHFT5']


@experimental_class('ComposerHFT5')
class ComposerHFT5(HuggingFaceModelWithFSDP):
    """Configures a :class:`.HuggingFaceModel` around a T5.

    Note: This function uses `transformers.T5ForConditionalGeneration`. Future releases
        will expand support to more general classes of HF Encoder-Decoder models.

    Args:
        cfg (DictConfig): An omegaconf dictionary used to configure the model:
            cfg.pretrained_model_name_or_path (str): The name of or local path to
                the HF model (e.g., `t5-base` to instantiate a T5 using the base config).
            cfg.config_overrides (dict, optional): An optional dictionary of keyword
                arguments that override the default configuration associated with
                cfg.pretrained_model_name_or_path. Default: ``{}``.
            cfg.pretrained (bool): Whether to instantiate the model with pre-trained
                weights coming from cfg.pretrained_model_name_or_path. If ``True``,
                cfg.config_overrides must be compatible with the pre-trained weights.
            cfg.init_device ('cpu' | 'meta'): Which device, 'cpu' or 'meta', to
                initialize the model on. Currently, `meta` is only supported when
                cfg.pretrained is ``False``. Default: ``'cpu'``.
        tokenizer (PreTrainedTokenizer): The tokenizer that the model will use.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        pretrained_model_name_or_path: str,
        pretrained: Optional[bool] = True,
        trust_remote_code: bool = True,
        use_auth_token: bool = False,
        config_overrides: Optional[Mapping] = None,
        init_device: str = 'cpu',
        additional_train_metrics: Optional[List] = None,
        name: Optional[str] = None,
    ):
        from llmfoundry.utils.builders import build_metric

        config_overrides = om.to_container(config_overrides or {}, resolve=True)

        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            use_auth_token=use_auth_token,
        )

        # set config overrides
        for k, v in (config_overrides or {}).items():
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

        if not config.is_encoder_decoder:
            raise ValueError(f'Model type "hf_t5" currently only supports T5 models ' +\
                             f'using configs where `is_encoder_decoder` is ``True``.')

        init_device = init_device

        # Get the device we want to initialize, and use the
        # resolved version to initialize the HF model
        resolved_init_device = hf_get_init_device(init_device)

        # We need to have all non-zero local ranks be not-pretrained
        # Rank 0 will still be pretrained, and distribute the weights appropriately
        if dist.get_local_rank() != 0 and init_device == 'mixed':
            pretrained = False

        if resolved_init_device == 'cpu':
            if pretrained:
                model = T5ForConditionalGeneration.from_pretrained(
                    pretrained_model_name_or_path, config=config)
            else:
                model = T5ForConditionalGeneration(config)
        elif resolved_init_device == 'meta':
            if pretrained:
                raise ValueError(
                    'Setting cfg.pretrained=True is not supported when init_device="meta".'
                )
            with init_empty_weights(include_buffers=False):
                model = T5ForConditionalGeneration(config)
        else:
            raise ValueError(
                f'init_device="{init_device}" must be either "cpu" or "meta".')

        metrics = [
            build_metric(metric, {}) for metric in DEFAULT_ENC_DEC_METRICS +
            (additional_train_metrics or [])
        ]

        composer_model = super().__init__(model=model,
                                          tokenizer=tokenizer,
                                          metrics=metrics,
                                          init_device=init_device)

        return composer_model
