# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Aggregate ICL evals into composite scores."""

import math
from enum import Enum
from typing import Optional

from composer.core import Callback, State
from composer.loggers import Logger
from composer.loggers.in_memory_logger import InMemoryLogger

__all__ = ['ModelGauntlet']


class Weighting(Enum):
    EQUAL = 1
    SAMPLE_SZ = 2
    LOG_SAMPLE_SZ = 3


def get_in_memory_logger(logger: Logger):
    for lg in logger.destinations:
        if isinstance(lg, InMemoryLogger):
            return lg
    raise Exception("Couldn't find InMemoryLogger in logger destinations!")


class ModelGauntlet(Callback):
    """The ModelGauntlet aggregates ICL eval results.

    After `eval_end`, this callback inspects the logger for different ICL metrics and aggregates the scores according to the aggregation
    specification provided in the constructor.

    Args:
        metric_names (dict): These are the exact keys that the individual benchmark metrics will be
                            logged under in the logger after eval
        tasks (dict): This contains the list of categories, as well as the subtasks within them, the
                      random baseline accuracy of each subtask, and the number of fewshot examples
                      used for the task. See `llmfoundry/scripts/eval/yamls/model_gauntlet.yaml` to see the structure.
        weighting (Weighting): The weighting scheme used to balance different tasks within each category.
                               Either assign them all equal weight, assign them weight proportional
                               to the dataset size, or assign them weight proportional to the log2 of the dataset size.
                               Options are 'EQUAL', 'SAMPLE_SZ', and 'LOG_SAMPLE_SZ'.
        subtract_random_baseline (bool): Flag determining whether to subtract random baseline accuracy
                                          from the performance on each individual benchmark before aggregating.
        rescale_accuracy (bool): Flag determining whether to rescale the accuracy on each benchmark
                                 by (1-random_baseline_accuracy) before aggregating. Using this ensures that all benchmarks max out at 1.0.
        benchmark_sizes (Optional[dict]): Optional data on benchmark sizes, used when not relying on equal weighting.
    """

    def __init__(self,
                 metric_names: dict,
                 categories: dict,
                 weighting: str = 'EQUAL',
                 subtract_random_baseline: bool = True,
                 rescale_accuracy: bool = True,
                 benchmark_sizes: Optional[dict] = None):
        if weighting != Weighting.EQUAL and benchmark_sizes is None:
            raise Exception(
                'When not using equal weighting, you must provide the benchmark sizes.'
            )

        if rescale_accuracy and not subtract_random_baseline:
            raise Exception(
                'Only use accuracy rescaling in conjunction with subtracting random baseline accuracy.'
            )

        self.categories = categories
        self.weighting = Weighting[weighting]
        self.subtract_random_baseline = subtract_random_baseline
        self.rescale_accuracy = rescale_accuracy
        self.metric_names = metric_names

        for category in self.categories:

            for benchmark in category['benchmarks']:
                bench_name = f"{benchmark['name']}/{benchmark['num_fewshot']}-shot"

                if self.weighting != Weighting.EQUAL:
                    assert benchmark_sizes is not None
                    cumulative_samples = max(
                        sum(count for name, count in benchmark_sizes.items()
                            if name.startswith(bench_name)), 1)
                else:
                    cumulative_samples = -1  # pyright

                weight = None
                if self.weighting == Weighting.EQUAL:
                    weight = 1
                elif self.weighting == Weighting.SAMPLE_SZ:
                    weight = cumulative_samples
                elif self.weighting == Weighting.LOG_SAMPLE_SZ:
                    weight = max(math.log(cumulative_samples, 2), 1)

                assert weight is not None
                benchmark['weighting'] = weight

    def compute_averages(self, state: State):
        results = {}

        for key in self.metric_names:

            # starting at index 1 skips the "metric" part of they key which is superfluous
            dl_name, metric_name = key.split('/')[1:-1], key.split('/')[-1]
            if 'Accuracy' not in metric_name:
                continue

            metric = state.eval_metrics.get('/'.join(dl_name),
                                            {}).get(metric_name, None)
            if metric is None:
                continue
            val = metric.compute().item()

            # ending at index 2 allows us to aggregate over dataloaders w/ subcategories
            key = '/'.join(dl_name[0:2])
            if key not in results:
                results[key] = []

            results[key].append(val)

        return {k: sum(v) / len(v) for k, v in results.items()}

    def eval_after_all(self, state: State, logger: Logger):
        new_metrics = self.compute_averages(state)
        if len(new_metrics) == 0:
            return {}
        composite_scores = {}
        for category in self.categories:
            composite_scores[category['name']] = []
            for benchmark in category['benchmarks']:
                key = f"{benchmark['name']}/{benchmark['num_fewshot']}-shot"

                if key not in new_metrics:
                    print(
                        f"Warning: couldn't find results for benchmark: {benchmark}"
                    )
                else:
                    score = new_metrics[key]

                    if self.subtract_random_baseline:
                        score -= benchmark['random_baseline']

                    if self.rescale_accuracy and self.subtract_random_baseline:
                        score /= 1.0 - benchmark['random_baseline']

                    composite_scores[category['name']].append({
                        'name': benchmark['name'],
                        'score': score,
                        'weighting': benchmark['weighting']
                    })
            total_weight = sum(
                k['weighting'] for k in composite_scores[category['name']])
            composite_scores[category['name']] = sum(
                k['score'] * (k['weighting'] / total_weight)
                for k in composite_scores[category['name']])

        composite_scores = {
            f'icl/metrics/model_gauntlet/{k}': v
            for k, v in composite_scores.items()
        }

        composite_scores['icl/metrics/model_gauntlet/average'] = sum(
            composite_scores.values()) / len(composite_scores.values())

        if logger is not None:
            print(
                f'Logging metrics {composite_scores} to logger destinations: {logger.destinations}'
            )
            logger.log_metrics(composite_scores)

        return composite_scores
