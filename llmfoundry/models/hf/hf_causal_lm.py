# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Implements a Hugging Causal LM wrapped inside a :class:`.ComposerModel`."""

import os
from typing import Mapping, Union

# required for loading a python model into composer
import transformers
from composer.metrics.nlp import (InContextLearningLMAccuracy,
                                  InContextLearningLMExpectedCalibrationError,
                                  InContextLearningMCExpectedCalibrationError,
                                  InContextLearningMultipleChoiceAccuracy,
                                  InContextLearningQAAccuracy,
                                  LanguageCrossEntropy, LanguagePerplexity)
from composer.utils import dist
from omegaconf import DictConfig
from transformers import (AutoConfig, AutoModelForCausalLM,
                          PreTrainedTokenizerBase)

from llmfoundry.models.hf.hf_fsdp import hf_get_init_device
from llmfoundry.models.hf.model_wrapper import HuggingFaceModelWithZLoss
from llmfoundry.models.utils import init_empty_weights

try:
    from peft.peft_model import PeftModel
    model_types = PeftModel, transformers.PreTrainedModel
    _om_model_config_type = Union[DictConfig, PeftModel,
                                  transformers.PreTrainedModel]

except ImportError:
    model_types = transformers.PreTrainedModel
    _om_model_config_type = Union[DictConfig, transformers.PreTrainedModel]

__all__ = ['ComposerHFCausalLM']


class ComposerHFCausalLM(HuggingFaceModelWithZLoss):
    """Configures a :class:`.HuggingFaceModel` around a Causal LM.

    Args:
        om_model_config (DictConfig | PeftModel | transformers.PreTrainedModel): either an omegaconf dictionary used to configure the model, or an instantiated model object from the peft or transformers library.
        if DictConfig, the following keys are required:
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

    def __init__(
            self,
            om_model_config: _om_model_config_type,  # type: ignore
            tokenizer: PreTrainedTokenizerBase):

        # set up training and eval metrics
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

        # if we are passed a DictConfig, we need to instantiate the model
        if isinstance(om_model_config, DictConfig):

            # load the model config
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
                    extra_keys = [
                        _k for _k in v.keys() if _k not in attr.keys()
                    ]
                    if extra_keys:
                        raise ValueError(
                            f'Config dict override got unknown keys. ' +
                            f'Extra keys: {extra_keys}. ' +
                            f'Expected (a subset of) keys: {list(attr.keys())}.'
                        )
                    getattr(config, k).update(v)
                else:
                    setattr(config, k, v)

            # below we set up the device to initialize the model on
            init_device = om_model_config.get('init_device', 'cpu')

            # Get the device we want to initialize, and use the
            # reolved version to initialize the HF model
            resolved_init_device = hf_get_init_device(init_device)

            # We need to have all non-zero local ranks be not-pretrained
            # Rank 0 will still be pretrained, and distribute the weights appropriately
            if dist.get_local_rank() != 0 and init_device == 'mixed':
                om_model_config.pretrained = False

            # initialize the model on the correct device
            if resolved_init_device == 'cpu':
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
                    f'init_device="{init_device}" must be either "cpu" or "meta".'
                )

            signal_file_path = '.local_rank0_completed_autoresume'
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

            z_loss = om_model_config.get('z_loss', 0.0)

        # elif the model is either a PeftModel or a PreTrainedModel
        elif isinstance(om_model_config, model_types):
            model = om_model_config
            init_device = 'cpu'
            z_loss = 0.0

        # else, unsupported type
        else:
            raise ValueError(
                f'om_model_config must be either a DictConfig, PeftModel, or PreTrainedModel, but got {type(om_model_config)}'
            )

        composer_model = super().__init__(model=model,
                                          shift_labels=True,
                                          tokenizer=tokenizer,
                                          metrics=train_metrics,
                                          eval_metrics=eval_metrics,
                                          z_loss=z_loss,
                                          init_device=init_device)

        return composer_model
