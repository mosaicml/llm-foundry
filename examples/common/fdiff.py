# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Monitor rate of change of loss."""
from __future__ import annotations

from composer.core import Callback, State
from composer.loggers import Logger


class FDiffMetrics(Callback):
    """Rate of chage of metrics.

    tracks and plots the rate of change of metrics effectively taking the
    numerical derivative of the metrics
    """

    def __init__(self, diff_train_metrics=False, diff_eval_metrics=True):
        self.diff_train_metrics = diff_train_metrics
        self.diff_eval_metrics = diff_eval_metrics

        self.train_prev_loss = None
        self.train_prev_metric = {}
        self.eval_prev_metric = {}

    def batch_end(self, state: State, logger: Logger):
        if self.diff_train_metrics:
            if self.train_prev_loss:
                logger.log_metrics({
                    'loss/train/total_fdiff':
                        state.loss.item() - self.train_prev_loss
                })

            self.train_prev_loss = state.loss.item()

            for k in self.train_prev_metric.keys():
                logger.log_metrics({
                    f'metrics/train/{k}_fdiff':
                        state.train_metric_values[k].item() -
                        self.train_prev_metric[k]
                })

            for k in state.train_metric_values.keys():
                self.train_prev_metric[k] = state.train_metric_values[k].item()

    def eval_end(self, state: State, logger: Logger):
        if self.diff_eval_metrics:
            evaluator = state.dataloader_label
            metrics = list(state.eval_metrics[evaluator].keys())

            for k in metrics:
                mkey = '/'.join(['metrics', evaluator, k])
                if mkey in self.eval_prev_metric.keys():
                    logger.log_metrics({
                        f'{mkey}_fdiff':
                            state.eval_metric_values[k].item() -
                            self.eval_prev_metric[mkey]
                    })

            for k in metrics:
                mkey = '/'.join(['metrics', evaluator, k])
                value = state.eval_metric_values[k].item()
                self.eval_prev_metric[mkey] = value
