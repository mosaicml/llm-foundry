# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch
import torch.nn.functional as F
from composer.metrics.nlp import (HFCrossEntropy, LanguageCrossEntropy,
                                  Perplexity)
from composer.models.huggingface import HuggingFaceModel
from omegaconf import DictConfig
from torch import Tensor
from torchmetrics import Metric
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from examples.common.hf_fsdp import prepare_hf_causal_lm_model_for_fsdp


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
