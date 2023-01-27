# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0
# helper functions from https://github.com/CarperAI/trlx/blob/main/trlx/utils/modeling.py
# which is MIT licensed

import functools
from typing import Any, Iterable, List, Optional

import torch
import torch.nn.functional as F
from composer.metrics.nlp import (HFCrossEntropy, LanguageCrossEntropy,
                                  Perplexity)
from composer.models.huggingface import HuggingFaceModel
from omegaconf import DictConfig
from torch import Tensor
from torchmetrics import Metric
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# helper functions


def rhasattr(obj: Any, attr: str):
    """A chain-able attribute version of hasattr.

    For example, to check if
    `obj` has the attribute `foo.bar.baz`, you can use:
        `rhasattr(obj, "foo.bar.baz")`
    Reference: https://stackoverflow.com/a/67303315
    """
    _nested_attrs = attr.split('.')
    _curr_obj = obj
    for _a in _nested_attrs[:-1]:
        if hasattr(_curr_obj, _a):
            _curr_obj = getattr(_curr_obj, _a)
        else:
            return False
    return hasattr(_curr_obj, _nested_attrs[-1])


def rgetattr(obj: Any, attr: str, *args: List[Any]):
    """A chain-able attribute version of getattr.

    For example, to get the attribute `foo.bar.baz` from `obj`, you can use:
        `rgetattr(obj, "foo.bar.baz")`
    Reference: https://stackoverflow.com/a/31174427
    """

    def _getattr(obj: Any, attr: str):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))


def findattr(obj: Any, attrs: Iterable[str]):
    for attr in attrs:
        if rhasattr(obj, attr):
            return rgetattr(obj, attr)
    return None


def hf_get_causal_base_model(model: AutoModelForCausalLM):
    """Returns the causal decoder backbone of the specified HuggingFace model.

    NOTE: Different model configurations have different causal decoder attribute
    names.
        - transformer: (GPT2LMHeadModel, GPTJConfig)
        - model.decoder: (OPTConfig, BloomConfig)
        - gpt_neox: (GPTNeoXConfig)
    """
    decoder_attrs = ('transformer', 'model.decoder', 'gpt_neox')
    return findattr(model, decoder_attrs)


def hf_get_lm_head(model: AutoModelForCausalLM):
    """Returns the lm head of the specified HuggingFace model.

    NOTE: Different model configurations have different `lm_head` attribute names.
        - lm_head: (GPT2LMHeadModel, BloomForCausalLM)
        - embed_out: (GPTNeoXForCausalLM)
    """
    return model.get_output_embeddings()


def hf_get_causal_hidden_layers(model: torch.nn.Module):
    """Returns the hidden layers of the specified model.

    NOTE: Different model configurations have different hidden layer attribute names.
        - transformer.h: (BloomForCausalLM, GPT2LMHeadModel, GPTJForCausalLM)
        - model.decoder.layers: (OPTForCausalLM)
        - gpt_neox.layers: (GPTNeoXForCausalLM)
    """
    hidden_layers_attrs = (
        'transformer.h',
        'model.decoder.layers',
        'gpt_neox.layers',
    )
    return findattr(model, hidden_layers_attrs)


def hf_get_tied_embedding_weights(model: torch.nn.Module):
    """Returns the embeddings, which are weight tied layers.

    NOTE: Different model configurations have different embedding attribute names.
        - wte: (GPT2LMHeadModel, GPTJForCausalLM, GPTNeoForCausalLM)
        - word_embeddings: (BloomForCausalLM)
        - embed_tokens: (OPTForCausalLM)
        - GPT NeoX doesn't weight tie
    """
    tied_embedding_attrs = (
        'wte',
        'word_embeddings',
        'embed_tokens',
    )
    return findattr(model, tied_embedding_attrs)


# /end helper functions


def prepare_hf_causal_lm_model_for_fsdp(model: AutoModelForCausalLM):
    """FSDP wrap a HuggingFace model.

    Wrap any model for FSDP which follows one of the 3 existing conventions from
    HuggingFace for decoder-only LLMs.
    """
    causal_base_model = hf_get_causal_base_model(model)
    model_block = hf_get_causal_hidden_layers(model)  # type: ignore
    lm_head = hf_get_causal_hidden_layers(model)
    tied_embeddings = hf_get_tied_embedding_weights(
        causal_base_model)  # type: ignore
    modules = [causal_base_model, model_block, lm_head, tied_embeddings]
    if any(module is None for module in modules):
        raise ValueError('Unable to FSDP-wrap this model! It does not follow \
                          common layer/weight naming conventions.')
    block_type = type(model_block[0])  # type: ignore
    # When using the HF LM models,
    # the weights of the self.lm_head and self.transformer.wte are tied.
    # This tying occurs inside the `self.post_init()` function.
    # This is a hurdle for FSDP because they need to be in the same FSDP block
    # These lines ensures that both modules stay together in the top-most block when
    # the model has this tying enabled (almost all do; this property defaults to True)
    if model.config.tie_word_embeddings:
        causal_base_model._fsdp_wrap = False  # type: ignore
        tied_embeddings._fsdp_wrap = False  # type: ignore
        lm_head._fsdp_wrap = False  # type: ignore

    # FSDP Wrap and Activation Checkpoint every model block
    model.fsdp_wrap_fn = lambda module: isinstance(module, block_type)
    model.activation_checkpointing_fn = lambda module: isinstance(
        module, block_type)


class ComposerHFCausalLM(HuggingFaceModel):

    def __init__(self, cfg: DictConfig):
        config = AutoConfig.from_pretrained(cfg.hf_config_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(cfg.hf_config_name_or_path)

        if cfg.pretrained:
            model = AutoModelForCausalLM.from_pretrained(
                cfg.hf_config_name_or_path, config=config)
            metrics = [HFCrossEntropy(), Perplexity()]
        else:
            model = AutoModelForCausalLM.from_config(config)
            metrics = [LanguageCrossEntropy(len(tokenizer)), Perplexity()]

        prepare_hf_causal_lm_model_for_fsdp(model)

        super().__init__(model=model,
                         tokenizer=tokenizer,
                         metrics=metrics,
                         use_logits=True)

    def get_targets(self, batch: dict):
        targets = torch.roll(batch['labels'], shifts=-1)
        targets[:, -1] = -100
        return targets

    def forward(self, batch: dict):
        return self.model(input_ids=batch['input_ids'],
                          attention_mask=batch['attention_mask'].bool()).logits

    def eval_forward(self, batch: dict, outputs: Optional[Tensor] = None):
        return outputs if outputs is not None else self.forward(batch)

    def loss(self, outputs: Tensor, batch: dict):
        targets = self.get_targets(batch)
        return F.cross_entropy(outputs.view(-1, outputs.size(-1)),
                               targets.view(-1),
                               ignore_index=-100)

    def update_metric(self, batch: dict, outputs: Tensor, metric: Metric):
        outputs = outputs.view(-1, outputs.size(-1))
        targets = self.get_targets(batch).view(-1)
        metric.update(outputs, targets)
