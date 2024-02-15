# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Aggregate ICL evals into composite scores."""

import logging
import math
from enum import Enum
from typing import Dict, Optional

from composer.core import Callback, State
from composer.loggers import Logger

__all__ = ['EvalGauntlet']

log = logging.getLogger(__name__)


class Weighting(Enum):
    EQUAL = 1
    SAMPLE_SZ = 2
    LOG_SAMPLE_SZ = 3


def calculate_named_averages(average_names: Dict[str, list],
                             category_scores: Dict[str, float]):
    """Calculates the named averages based off the raw category scores.

    For each named average, take a simple average of all the category scores associated with that named average.

    Args:
        average_names (dict[str, list]):  Contains a mapping of named averages to which category scores that average should consist of.
        category_scores (dict[str, float]): Contains the raw scores corresponding to each category.
    """
    average_scores = {}
    for avg_name, category_list in average_names.items():
        composite_subset = {
            category: score
            for category, score in category_scores.items()
            if category in category_list
        }
        if len(composite_subset.values()) > 0:
            average_scores[avg_name] = sum(composite_subset.values()) / len(
                composite_subset.values())
        else:
            average_scores[avg_name] = 0

    return average_scores


class EvalGauntlet(Callback):
    """The EvalGauntlet aggregates ICL eval results.

    After `eval_end`, this callback inspects the logger for different ICL metrics and aggregates the scores according to the aggregation
    specification provided in the constructor.

    Args:
        logger_keys (list): These are the exact keys that the individual benchmark metrics will be
                            logged under in the logger after eval
        categories (dict): This contains the list of categories, as well as the subtasks within them, the
                      random baseline accuracy of each subtask, and the number of fewshot examples
                      used for the task. See `llmfoundry/scripts/eval/yamls/eval_gauntlet_v0.2.yaml` to see the structure.
        weighting (Weighting): The weighting scheme used to balance different tasks within each category.
                               Either assign them all equal weight, assign them weight proportional
                               to the dataset size, or assign them weight proportional to the log2 of the dataset size.
                               Options are 'EQUAL', 'SAMPLE_SZ', and 'LOG_SAMPLE_SZ'.
        subtract_random_baseline (bool): Flag determining whether to subtract random baseline accuracy
                                          from the performance on each individual benchmark before aggregating.
        rescale_accuracy (bool): Flag determining whether to rescale the accuracy on each benchmark
                                 by (1-random_baseline_accuracy) before aggregating. Using this ensures that all benchmarks max out at 1.0.
        benchmark_sizes (Optional[dict]): Optional data on benchmark sizes, used when not relying on equal weighting.
        averages (Optional[dict]): Optional dictionary specifying a mapping from a average names to lists of categories used produce each named average.
    """

    def __init__(self,
                 logger_keys: list,
                 categories: dict,
                 weighting: str = 'EQUAL',
                 subtract_random_baseline: bool = True,
                 rescale_accuracy: bool = True,
                 benchmark_sizes: Optional[dict] = None,
                 averages: Optional[dict] = None):
        if isinstance(logger_keys, dict):
            raise ValueError(
                'logger_keys now requires a list type as input, not a dict')
        if weighting != Weighting.EQUAL and benchmark_sizes is None:
            raise Exception(
                'When not using equal weighting, you must provide the benchmark sizes.'
            )

        if rescale_accuracy and not subtract_random_baseline:
            raise Exception(
                'Only use accuracy rescaling in conjunction with subtracting random baseline accuracy.'
            )

        self.categories = categories
        self.category_names = [conf.get('name') for conf in self.categories]
        self.weighting = Weighting[weighting]
        self.subtract_random_baseline = subtract_random_baseline
        self.rescale_accuracy = rescale_accuracy
        self.logger_keys = logger_keys
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
                    weight = max(math.log2(cumulative_samples), 1)
                assert weight is not None
                benchmark['weighting'] = weight

        self.averages = {}
        if averages is not None:
            self.averages = averages
        else:
            # if no averages spec provided, simply average everything
            self.averages['default_average'] = self.category_names

        for avg_name in self.averages:
            if avg_name in self.category_names:
                raise ValueError(
                    f'Found average name `{avg_name}` used as category name. Average names and category names must be non-overlapping.'
                )

    def extract_metrics_from_state(self, state: State) -> Dict[str, float]:
        results = {}

        for key in self.logger_keys:

            # starting at index 1 skips the "metric" part of the key which is superfluous
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

    def eval_after_all(self, state: State, logger: Logger) -> Dict[str, float]:
        computed_metrics = self.extract_metrics_from_state(state)
        if len(computed_metrics) == 0:
            return {}
        category_scores = {}
        for category in self.categories:
            missing_metrics = []
            category_scores[category['name']] = []
            for benchmark in category['benchmarks']:
                key = f"{benchmark['name']}/{benchmark['num_fewshot']}-shot"

                if key not in computed_metrics:
                    log.warning(
                        f'Could not find results for benchmark: {benchmark}.')
                    missing_metrics.append(key)
                else:
                    score = computed_metrics[key]

                    if self.subtract_random_baseline:
                        score -= benchmark['random_baseline']

                    if self.rescale_accuracy and self.subtract_random_baseline:
                        score /= 1.0 - benchmark['random_baseline']

                    category_scores[category['name']].append({
                        'name': benchmark['name'],
                        'score': score,
                        'weighting': benchmark['weighting']
                    })

            if len(missing_metrics) > 0:
                log.warning(
                    f"Removing category `{category['name']}` from scores because benchmarks were missing: {missing_metrics}"
                )
                del category_scores[category['name']]
                continue
            total_weight = sum(
                k['weighting'] for k in category_scores[category['name']])
            category_scores[category['name']] = sum(
                k['score'] * (k['weighting'] / total_weight)
                for k in category_scores[category['name']])

        named_averages = calculate_named_averages(self.averages,
                                                  category_scores)
        category_scores.update(named_averages)
        category_scores = {
            f'icl/metrics/eval_gauntlet/{k}': v
            for k, v in category_scores.items()
        }
        if logger is not None:
            logger.log_metrics(category_scores)

        return category_scores
