# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import re
import string
from collections import Counter
from typing import Any, Dict, List, Optional

import torch
from composer.metrics import InContextLearningMetric
from torch import Tensor


class InContextLearningGenerationF1Score(InContextLearningMetric):
    r"""Computes F1 score for In-context learning (ICL) generation (QA) tasks.

    ICL QA tasks consist of some number of example question answering tasks (referred to as the 'context'), followed by a test task where the model must
    match one of the possible answer aliases (referred to as the 'continuation').

    For example, the model may be provided the context below and evaluated on its ability to correctly predict the continuation.

    Context: `Question: Who was president of the United States in 2012?\nAnswer: Barack Obama\nQuestion: Is water wet?\nAnswer: `
    Continuation: [`yes`, `no`]

    Both predictions and answers will be normalized before comparison.

    Adds metric state variables:
        correct (float): The number of instances where the prediction was a prefix for any of the answer aliases.
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
        self.add_state('correct',
                       default=torch.tensor(0.),
                       dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0.), dist_reduce_fx='sum')

    def normalize_answer(self, answer: str):
        """Taken from official evaluation script for v1.1 of the SQuAD.

        Lower text and remove punctuation, articles and extra whitespace.
        """

        def remove_articles(text: str):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text: str):
            return ' '.join(text.split())

        def remove_punc(text: str):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text: str):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(answer))))

    def update(self, batch: Optional[Dict[str, Any]], outputs: List[str],
               labels: List[List[str]]):
        if batch is None:
            batch = {}
        for sample_output, sample_labels in zip(outputs, labels):
            prediction_tokens = self.normalize_answer(sample_output).split()
            max_f1 = 0
            for label in sample_labels:
                references_tokens = self.normalize_answer(label).split()
                common = Counter(prediction_tokens) & Counter(references_tokens)
                num_same = sum(common.values())
                if num_same == 0:
                    f1 = 0
                else:
                    precision = 1.0 * num_same / len(prediction_tokens)
                    recall = 1.0 * num_same / len(references_tokens)
                    f1 = (2 * precision * recall) / (precision + recall)
                max_f1 = max(max_f1, f1)

            self.correct += torch.tensor(max_f1)
            self.total += torch.tensor(1.0)

    def compute(self):
        assert isinstance(self.correct, Tensor)
        assert isinstance(self.total, Tensor)
        return self.correct / self.total
