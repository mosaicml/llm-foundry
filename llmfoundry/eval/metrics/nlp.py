# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""A collection of common torchmetrics for NLP tasks."""

import copy
import functools
import logging
import re
import string
from typing import Any, Callable, Dict, List

import torch
from torch import Tensor
from torch.nn import functional as F
from torchmetrics import Metric

log = logging.getLogger(__name__)

__all__ = [
    'InContextLearningMetric',
    'InContextLearningLMAccuracy',
    'InContextLearningMultipleChoiceAccuracy',
    'InContextLearningGenerationExactMatchAccuracy',
    'InContextLearningLMExpectedCalibrationError',
    'InContextLearningMCExpectedCalibrationError',
]


class InContextLearningMetric(Metric):

    def __init__(self, *args, **kwargs):  # pyright: ignore
        super().__init__(*args, **kwargs, sync_on_compute=False)
        self.needs_batch = True

    def _wrap_update(self, update: Callable) -> Callable:
        """Overwrite default _wrap_update to return result of update().

        Torch metrics wraps update with following wrapped_func but explicitly
        does not return the value. In general, torchmetrics update() does not
        return a value, but we want to in order to pass it on to
        state.metric_outputs.
        """

        @functools.wraps(update)
        def wrapped_func(*args: Any, **kwargs: Any) -> None:
            self._computed = None
            self._update_count += 1
            with torch.set_grad_enabled(self._enable_grad):
                try:
                    update_result = update(*args, **kwargs)
                except RuntimeError as err:
                    if 'Expected all tensors to be on' in str(err):
                        raise RuntimeError(
                            'Encountered different devices in metric calculation (see stacktrace for details).' + \
                            ' This could be due to the metric class not being on the same device as input.' + \
                            f' Instead of `metric={self.__class__.__name__}(...)` try to do' + \
                            f' `metric={self.__class__.__name__}(...).to(device)` where' + \
                            ' device corresponds to the device of the input.',
                        ) from err
                    raise err

            if self.compute_on_cpu:
                self._move_list_states_to_cpu()
            return update_result

        return wrapped_func

    def update(
        self,
        batch: dict,
        outputs: torch.Tensor,
        labels: torch.Tensor,
    ):
        """Abstract interface for computing an in-context learning metrics.

        The `output_logits` argument is deprecated and will be removed in v0.21 while it's functionality will
        be moved to `outputs`.

        Args:
            batch (dict): Batch must consist minimally of `input_ids` as well as any other structure needed
                to compute the metric.
            output_logits (torch.Tensor): The model outputs evaluated on the batch `input_ids`
            labels (torch.Tensor): The correct outputs.

        Raises:
            NotImplementedError: Abstract method must be implemented by subclasses
        """
        raise NotImplementedError


class InContextLearningGenerationExactMatchAccuracy(InContextLearningMetric):
    r"""Computes exact match for in-context learning generation tasks.

    ICL generation tasks consist of some number of prompted generation tasks with correct answers
    followed by a test task where the model must correctly produce one of a number of valid answers.

    For example, the model may be provided the context below and evaluated on its ability to correctly predict the continuation.

    Context: `Question: Who was president of the United States in 2012?\nAnswer: Barack Obama\nQuestion: Is water wet?\nAnswer: `
    Answers: [`yes`]

    The model will be expected to correctly produce one of the answers, following some optional normalization.

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
        self.add_state(
            'correct',
            default=torch.tensor(0.),
            dist_reduce_fx='sum',
        )
        self.add_state('total', default=torch.tensor(0.), dist_reduce_fx='sum')
        self.metric_result_dict = {
            'cleaned_output': [],
            'original_label': [],
            'cleaned_label': [],
            'result': [],
        }

    def normalize_answer(self, answer: str):
        """Lower text and remove punctuation, articles and extra whitespace.

        Copied from https://github.com/mandarjoshi90/triviaqa/blob/master/evaluation/triviaqa_evaluation.py
        """

        def remove_articles(text: str) -> str:
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text: str) -> str:
            return ' '.join(text.split())

        def handle_punc(text: str) -> str:
            exclude = set(
                string.punctuation + ''.join([u'‘', u'’', u'´', u'`']),
            )
            return ''.join(ch if ch not in exclude else ' ' for ch in text)

        def lower(text: str) -> str:
            return text.lower()

        def replace_underscore(text: str) -> str:
            return text.replace('_', ' ')

        return white_space_fix(
            remove_articles(handle_punc(lower(replace_underscore(answer)))),
        ).strip()

    def update(
        self,
        batch: Dict[str, Any],
        outputs: List[str],
        labels: List[List[str]],
    ):
        cot_delimiter = batch.get('cot_delimiter', '')
        do_normalization = batch.get('do_normalization', True)
        stopping_criteria = batch.get('stopping_criteria', None)
        metric_result_dict = copy.deepcopy(self.metric_result_dict)
        for sample_output, sample_labels in zip(outputs, labels):
            final_answer = sample_output

            if stopping_criteria is not None and len(stopping_criteria) > 0:
                final_answer = re.split(
                    '|'.join(stopping_criteria),
                    final_answer,
                )[0]

            if cot_delimiter is not None and len(cot_delimiter) > 0:
                final_answer = final_answer.split(cot_delimiter)[-1]

            if do_normalization:
                cleaned_final_answer = self.normalize_answer(final_answer)
                cleaned_sample_labels = {
                    self.normalize_answer(label) for label in sample_labels
                }
            else:
                # even if normalization is off, we should still strip leading/trailing whitespaces
                cleaned_final_answer = final_answer.strip()
                cleaned_sample_labels = {
                    sample_label.strip() for sample_label in sample_labels
                }
            metric_result_dict['original_label'].append(sample_labels)
            metric_result_dict['cleaned_output'].append(cleaned_final_answer)
            metric_result_dict['cleaned_label'].append(cleaned_sample_labels)

            if any(
                cleaned_final_answer.startswith(label)
                for label in cleaned_sample_labels
            ):
                self.correct += torch.tensor(1.0)
                metric_result_dict['result'].append(1)
            else:
                metric_result_dict['result'].append(0)

            self.total += torch.tensor(1.0)

        return metric_result_dict

    def compute(self):
        assert isinstance(self.correct, Tensor)
        assert isinstance(self.total, Tensor)
        return self.correct / self.total


class InContextLearningLMAccuracy(InContextLearningMetric):
    r"""Computes accuracy for In-context learning language modeling tasks.

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
        self.add_state(
            'correct',
            default=torch.tensor(0.),
            dist_reduce_fx='sum',
        )
        self.add_state('total', default=torch.tensor(0.), dist_reduce_fx='sum')
        self.metric_result_dict = {
            'context': [],
            'label': [],
            'output': [],
            'result': [],
        }

    def update(self, batch: dict, outputs: torch.Tensor, labels: torch.Tensor):

        metric_result_dict = copy.deepcopy(self.metric_result_dict)
        for batch_idx, cont_idx in enumerate(batch['continuation_indices']):
            cont_tok_pred = outputs[batch_idx].index_select(
                dim=0,
                index=cont_idx - 1,
            ).argmax(dim=-1)
            cont_tok_targ = labels[batch_idx].index_select(
                dim=0,
                index=cont_idx - 1,
            )

            metric_result_dict['context'].append(
                batch['input_ids'][batch_idx][:cont_idx[0]],
            )
            metric_result_dict['label'].append(cont_tok_targ)
            metric_result_dict['output'].append(cont_tok_pred)

            correct = (cont_tok_pred == cont_tok_targ).all().int()
            self.correct += correct
            metric_result_dict['result'].append(int(correct))

            self.total += torch.tensor(1.0)

        return metric_result_dict

    def compute(self):
        assert isinstance(self.correct, Tensor)
        assert isinstance(self.total, Tensor)
        return self.correct / self.total


class InContextLearningMultipleChoiceAccuracy(InContextLearningMetric):
    r"""Computes accuracy for In-context learning multiple choice tasks.

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
        self.add_state(
            'correct',
            default=torch.tensor(0.0),
            dist_reduce_fx='sum',
        )
        self.add_state('total', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.metric_result_dict = {
            'context': [],
            'correct_choice': [],
            'correct_choice_idx': [],
            'selected_choice': [],
            'selected_choice_idx': [],
            'all_choices': [],
            'result': [],
        }

    def update(self, batch: dict, outputs: torch.Tensor, labels: torch.Tensor):

        perplexities = []
        for batch_idx, cont_idx in enumerate(batch['continuation_indices']):
            # continuation indices refer to indices in the original input's token space
            cont_tok_logits = outputs[batch_idx].index_select(
                dim=0,
                index=cont_idx - 1,
            )
            # labels have been shifted left by one index, so the cont_idx needs to be shifted as well.
            cont_tok_targ = labels[batch_idx].index_select(
                dim=0,
                index=cont_idx - 1,
            )
            cross_entropy = F.cross_entropy(cont_tok_logits, cont_tok_targ)
            perplexity = torch.exp(cross_entropy)
            perplexities.append(perplexity)

        metric_result_dict = copy.deepcopy(self.metric_result_dict)
        for (start, end), gold_idx in zip(
            batch['choice_groupings'],
            batch['gold_indices'],
        ):
            subset = perplexities[start:end]
            idx_min = subset.index(min(subset))

            if idx_min == gold_idx:
                self.correct += torch.tensor(1.0)
                metric_result_dict['result'].append(1)
            else:
                metric_result_dict['result'].append(0)

            question = batch['input_ids'][
                start][:batch['continuation_indices'][start][0]]

            correct_choice = batch['input_ids'][start:end][gold_idx][
                batch['continuation_indices'][start:end][gold_idx][0]:
                batch['continuation_indices'][start:end][gold_idx][-1] + 1]
            selected_choice = batch['input_ids'][start:end][idx_min][
                batch['continuation_indices'][start:end][idx_min][0]:
                batch['continuation_indices'][start:end][idx_min][-1] + 1]
            metric_result_dict['context'].append(question)
            metric_result_dict['correct_choice'].append(correct_choice)
            metric_result_dict['correct_choice_idx'].append(gold_idx)
            metric_result_dict['selected_choice'].append(selected_choice)
            metric_result_dict['selected_choice_idx'].append(idx_min)
            all_choices = batch['input_ids'][start:end]
            # Unpads the choices. Necessary in case different choices have different token lengths.
            if 'attention_mask' in batch:
                all_choices_list = [
                    choice[batch['attention_mask'][i]]
                    for i, choice in enumerate(all_choices)
                ]
                metric_result_dict['all_choices'].append(all_choices_list)

            self.total += torch.tensor(1.0)

        # Don't return all_choices if we didn't fill it up (i.e. didn't use causal lms)
        if metric_result_dict['all_choices'] == []:
            metric_result_dict.pop('all_choices')

        return metric_result_dict

    def compute(self):
        assert isinstance(self.correct, Tensor)
        assert isinstance(self.total, Tensor)
        return self.correct.float() / self.total


class InContextLearningExpectedCalibrationError(InContextLearningMetric):
    """Generic class for Expected Calibration Error (ECE).

    Citation: https://arxiv.org/pdf/1706.04599.pdf

    Expected calibration error is calculated by dividing predictions into buckets based on the model's confidence (a probability value between 0 and 1).
    We then calculate the accuracy within each bucket and calculate the average gap between confidence and accuracy
    across buckets, weighted by the number of samples in each bucket.

    Each task must implement its own definition of "confidence" to be computed via the `update` method.

    Adds metric state variables:
    bucket_totals (float): The number of instances where the prediction masked the target per bucket.
    bucket_correct (float): The number of total instances that were predicted per bucket.

    Args:
        dist_sync_on_step (bool, optional): Synchronize metric state across processes at
            each forward() before returning the value at the step. Default: ``False``.
        n_buckets (int): Number of distinct buckets to split the confidence distribution into
    """

    def __init__(self, dist_sync_on_step: bool = False, n_buckets: int = 10):
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.n_buckets = n_buckets
        if n_buckets < 1:
            raise Exception('`n_buckets`')
        self.add_state(
            'bucket_totals',
            default=torch.zeros(n_buckets),
            dist_reduce_fx='sum',
        )
        self.add_state(
            'bucket_correct',
            default=torch.zeros(n_buckets),
            dist_reduce_fx='sum',
        )

    def update(self, batch: dict, outputs: torch.Tensor, labels: torch.Tensor):
        pass

    def compute(self):
        assert isinstance(self.bucket_correct, Tensor)
        assert isinstance(self.bucket_totals, Tensor)

        result = torch.tensor(0.0, device=self.bucket_correct.device)
        total_obs = torch.sum(self.bucket_totals)
        for i in range(self.n_buckets):
            if self.bucket_totals[i] == 0:
                continue

            acc_bucket_i = self.bucket_correct[i] / self.bucket_totals[i]
            upper_bound = (i + 1) / self.n_buckets
            lower_bound = i / self.n_buckets
            conf_bucket_i = torch.tensor((upper_bound + lower_bound) / 2,
                                         device=self.bucket_correct.device)
            result += (self.bucket_totals[i] /
                       total_obs) * torch.abs(acc_bucket_i - conf_bucket_i)
        return result


class InContextLearningMCExpectedCalibrationError(
    InContextLearningExpectedCalibrationError,
):
    r"""Computes Expected Calibration Error (ECE) for In-context learning (ICL)

    multiple choice (MC) tasks. (source: https://arxiv.org/abs/2012.00955).

    For MC tasks, the model confidence is defined as the softmax of average per-token probability assigned to the top question choice.

    See `InContextLearningExpectedCalibrationError` for more info.
    """

    # Make torchmetrics call update only once
    full_state_update = False

    def update(self, batch: dict, outputs: torch.Tensor, labels: torch.Tensor):

        outputs = torch.softmax(outputs, dim=2)
        probabilities = []
        for batch_idx, cont_idx in enumerate(batch['continuation_indices']):
            cont_tok_logits = outputs[batch_idx].index_select(
                dim=0,
                index=cont_idx - 1,
            )
            cont_tok_targ = labels[batch_idx].index_select(
                dim=0,
                index=cont_idx - 1,
            )
            probability = cont_tok_logits.index_select(
                dim=1,
                index=cont_tok_targ,
            ).diagonal().mean()
            probabilities.append(probability)

        for (start, end), gold_idx in zip(
            batch['choice_groupings'],
            batch['gold_indices'],
        ):
            subset = probabilities[start:end]
            idx_max = subset.index(max(subset))
            confidence = torch.tensor(subset).max() / torch.tensor(subset).sum()

            assert confidence >= 0.0 and confidence <= 1.0
            bucket_idx = int(confidence * self.n_buckets)
            if bucket_idx == self.n_buckets:
                bucket_idx -= 1

            if idx_max == gold_idx:
                self.bucket_correct[
                    bucket_idx
                ] += 1  # pyright: ignore [reportGeneralTypeIssues]

            self.bucket_totals[
                bucket_idx] += 1  # pyright: ignore [reportGeneralTypeIssues]


class InContextLearningLMExpectedCalibrationError(
    InContextLearningExpectedCalibrationError,
):
    r"""Computes Expected Calibration Error (ECE) for In-context learning (ICL)

    language modeling (LM) tasks. (cite: https://arxiv.org/pdf/1706.04599.pdf).

    For LM tasks, the model confidence is defined as the minimum probability assigned to all tokens in the continuation.

    See `InContextLearningExpectedCalibrationError` for more info.
    """

    # Make torchmetrics call update only once
    full_state_update = False

    def update(self, batch: dict, outputs: torch.Tensor, labels: torch.Tensor):

        outputs = torch.softmax(outputs, dim=2)
        for batch_idx, cont_idx in enumerate(batch['continuation_indices']):
            cont_tok_logits = outputs[batch_idx].index_select(
                dim=0,
                index=cont_idx - 1,
            )
            cont_tok_pred = cont_tok_logits.argmax(dim=-1)
            confidence = cont_tok_logits.max(dim=-1).values.min()
            cont_tok_targ = labels[batch_idx].index_select(
                dim=0,
                index=cont_idx - 1,
            )
            assert confidence >= 0.0 and confidence <= 1.0
            bucket_idx = int(confidence * self.n_buckets)
            if bucket_idx == self.n_buckets:
                bucket_idx -= 1

            if (cont_tok_pred == cont_tok_targ).all():
                self.bucket_correct[
                    bucket_idx
                ] += 1  # pyright: ignore [reportGeneralTypeIssues]

            self.bucket_totals[
                bucket_idx] += 1  # pyright: ignore [reportGeneralTypeIssues]
