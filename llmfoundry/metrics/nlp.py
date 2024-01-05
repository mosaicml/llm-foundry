# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""A collection of common torchmetrics for NLP tasks."""

import logging
from typing import Any, Dict, List, Mapping, Optional, Union

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from composer.metrics import InContextLearningMetric
log = logging.getLogger(__name__)

__all__ = [
    'InContextLearningLMPerplexity',
    'InContextLearningLMClippedPerplexity'
    'InContextLearningMultipleChoicePerplexity',
    'InContextLearningMultipleChoiceClippedPerplexity',
]

class InContextLearningLMClippedPerplexity(InContextLearningMetric):
    r"""Computes accuracy for In-context learning (ICL) language modeling (LM) tasks.

    ICL LM tasks consist of some number of example language modeling tasks (referred to as the 'context'), followed by a test task where the model must correctly predict all the tokens
    following tokens in some passage (referred to as the 'continuation').

    For example, the model may be provided the context below and evaluated on its ability to correctly predict the continuation. Note: it doesn't matter
    whether the model correctly predicts the context tokens.

    Context: `The dog is->fuzzy\nthe water is->hot\nthe tree is->`
    Continuation: `green`

    Adds metric state variables:
        correct (float): The number of instances where the prediction masked the target.
        total (float): The number of total instances that were predicted.

    Args:
        dist_sync_on_step (bool, optional): Synchronize metric state across processes at
            each forward() before returning the value at the step. Default: ``False``.
    """

    # Make torchmetrics call update only once
    full_state_update = False

    def __init__(self, dist_sync_on_step: bool = False):
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state('pp', default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0.), dist_reduce_fx='sum')
        
    def _clip_pp(self, pp, **kwargs):
        cont_tok_pred = kwargs['cont_tok_pred']
        cont_tok_targ = kwargs['cont_tok_targ']
        if (cont_tok_pred == cont_tok_targ).all():
            return 0
        return pp
    
    def update(self, batch: dict, output_logits: torch.Tensor, labels: torch.Tensor):
        self.perplexities = []
        for batch_idx, cont_idx in enumerate(batch['continuation_indices']):
            cont_tok_pred = output_logits[batch_idx].index_select(dim=0, index=cont_idx - 1).argmax(dim=-1)

            self.total += torch.tensor(1.0)
            
            cont_tok_logits = output_logits[batch_idx].index_select(dim=0, index=cont_idx - 1)
            # labels have been shifted left by one index, so the cont_idx needs to be shifted as well.
            cont_tok_targ = labels[batch_idx].index_select(dim=0, index=cont_idx - 1)
            cross_entropy = F.cross_entropy(cont_tok_logits, cont_tok_targ)
            perplexity = torch.exp(cross_entropy)
            
            perplexity = self._clip_pp(perplexity, cont_tok_pred=cont_tok_pred, cont_tok_targ=cont_tok_targ)  

            self.perplexities.append(perplexity)
            self.total += torch.tensor(1.0)
            
        self.perplexities = torch.tensor(self.perplexities)

    def compute(self):
        assert isinstance(self.pp, Tensor)
        assert isinstance(self.total, Tensor)
        self.pp = (self.perplexities/self.total).sum()
        return self.pp.float() 

class InContextLearningLMPerplexity(InContextLearningLMClippedPerplexity):
    def _clip_pp(self, pp, **kwargs):
        return pp
       
class InContextLearningMultipleChoiceClippedPerplexity(InContextLearningMetric):
    r"""Computes accuracy for In-context learning (ICL) multiple choice (MC) tasks.

    ICL MC tasks consists of a series of questions with some number of possible choices (only one of which can be correct).
    At inference time each possible choice is given to the model as a separate input and the one for which the model assigns
    the lowest perplexity to the choice is considered the model's choice. The model is correct if it "chooses" the right answer.

    Context: `The dog is->fuzzy\nthe water is->hot\nthe tree is->`
    Continuation: `green`

    Adds metric state variables:
        correct (float): The number of instances where the prediction masked the target.
        total (float): The number of total instances that were predicted.

    Args:
        dist_sync_on_step (bool, optional): Synchronize metric state across processes at
            each forward() before returning the value at the step. Default: ``False``.
    """

    # Make torchmetrics call update only once
    full_state_update = False

    def __init__(self, dist_sync_on_step: bool = False):
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state('pp', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0.0), dist_reduce_fx='sum')
        
    def _clip_pp(self, pp, **kwargs):
        idx_min = kwargs['idx_min']
        gold_idx = kwargs['gold_idx']
        if idx_min == gold_idx:
            return 0
        return pp

    def update(self, batch: dict, output_logits: torch.Tensor, labels: torch.Tensor):
        perplexities = []
        for batch_idx, cont_idx in enumerate(batch['continuation_indices']):
            # continuation indices refer to indices in the original input's token space
            cont_tok_logits = output_logits[batch_idx].index_select(dim=0, index=cont_idx - 1)
            # labels have been shifted left by one index, so the cont_idx needs to be shifted as well.
            cont_tok_targ = labels[batch_idx].index_select(dim=0, index=cont_idx - 1)
            cross_entropy = F.cross_entropy(cont_tok_logits, cont_tok_targ)
            perplexity = torch.exp(cross_entropy)
            perplexities.append(perplexity)
            
        self.gold_perplexities = []
        for (start, end), gold_idx in zip(batch['choice_groupings'], batch['gold_indices']):
            gold_perplexity = perplexities[start:end][gold_idx]
            subset = perplexities[start:end]
            idx_min = subset.index(min(subset))
            gold_perplexity = self._clip_pp(gold_perplexity, idx_min=idx_min, gold_idx=gold_idx)
            self.gold_perplexities.append(gold_perplexity)
            self.total += torch.tensor(1.0)
            
        self.gold_perplexities = torch.tensor(self.gold_perplexities)

    def compute(self):
        assert isinstance(self.pp, Tensor)
        assert isinstance(self.total, Tensor)
        self.pp = (self.gold_perplexities/self.total).sum()
        return self.pp.float()
    
    
class InContextLearningMultipleChoicePerplexity(InContextLearningMultipleChoiceClippedPerplexity):
    def _clip_pp(self, pp, **kwargs):
        return pp
