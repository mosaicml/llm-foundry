# Copyright 2022 MosaicML Benchmarks authors
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
import transformers
from composer.metrics.nlp import LanguageCrossEntropy, Perplexity
from composer.models.base import ComposerModel
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2LMHeadModel
from transformers.models.gpt_neo.modeling_gpt_neo import (GPTNeoBlock,
                                                          GPTNeoForCausalLM)
from transformers.models.gpt_neox.modeling_gpt_neox import (GPTNeoXForCausalLM,
                                                            GPTNeoXLayer)

_SUPPORTED_HF_MODELS = (
    GPT2LMHeadModel,
    GPTNeoForCausalLM,
    GPTNeoXForCausalLM,
)

_HF_MODEL_BLOCKS = (
    GPT2Block,
    GPTNeoBlock,
    GPTNeoXLayer,
)

def prepare_hf_causal_lm_model_for_fsdp(model):
    assert isinstance(model, _SUPPORTED_HF_MODELS)
    # When using the HF Causal LM models,
    # the weights of the self.lm_head and self.transformer.wte are tied.
    # This tying occurs inside the `self.post_init()` function.
    # This is a hurdle for FSDP because they need to be in the same FSDP block
    # These lines ensures that both modules stay together in the top-most block
    model.transformer._fsdp_wrap = False
    model.transformer.wte._fsdp_wrap = False
    model.lm_head._fsdp_wrap = False

    # FSDP Wrap and Activation Checkpoint every GPT2Block
    model.fsdp_wrap_fn = lambda module: isinstance(module, _HF_MODEL_BLOCKS)
    model.activation_checkpointing_fn = lambda module: isinstance(module, _HF_MODEL_BLOCKS)


class ComposerHFCausalLM(ComposerModel):
    def __init__(self, cfg):
        super().__init__()
        config = AutoConfig.from_pretrained(cfg.hf_config_name_or_path)
        self.model = AutoModelForCausalLM.from_config(config)
        self.train_metrics = {
            'LanguageCrossEntropy': LanguageCrossEntropy(config.vocab_size),
            'Perplexity': Perplexity(),
        }
        self.eval_metrics = {
            'LanguageCrossEntropy': LanguageCrossEntropy(config.vocab_size),
            'Perplexity': Perplexity(),
        }
        prepare_hf_causal_lm_model_for_fsdp(self.model)

    def get_targets(self, batch):
        targets = torch.roll(batch["labels"], shifts=-1)
        targets[:, -1] = -100
        return targets

    def forward(self, batch):
        return self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'].bool()).logits

    def eval_forward(self, batch, outputs=None):
        return outputs if outputs is not None else self.forward(batch)

    def loss(self, outputs, batch):
        targets = self.get_targets(batch)
        return F.cross_entropy(outputs.view(-1, outputs.size(-1)),
                               targets.view(-1),
                               ignore_index=-100)

    def get_metrics(self, is_train=False):
        return self.train_metrics if is_train else self.eval_metrics

    def update_metric(self, batch, outputs, metric):
        outputs = outputs.view(-1, outputs.size(-1))
        targets = self.get_targets(batch).view(-1)
        metric.update(outputs, targets)
