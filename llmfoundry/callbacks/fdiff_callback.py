# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Monitor rate of change of loss."""
from __future__ import annotations

import torch
from composer.core import Callback, State
from composer.loggers import Logger


class FDiffMetrics(Callback):
    """Rate of change of metrics.

    tracks and plots the rate of change of metrics effectively taking the
    numerical derivative of the metrics
    """

    def __init__(self,
                 diff_train_metrics: bool = False,
                 diff_eval_metrics: bool = True):
        self.diff_train_metrics = diff_train_metrics
        self.diff_eval_metrics = diff_eval_metrics

        self.train_prev_loss = None
        self.train_prev_metric = {}
        self.eval_prev_metric = {}

    def batch_end(self, state: State, logger: Logger) -> None:
        if self.diff_train_metrics:
            if not isinstance(state.loss, torch.Tensor):
                raise NotImplementedError('Multiple losses not supported yet')
            loss = state.loss.item()
            if self.train_prev_loss:
                logger.log_metrics(
                    {'loss/train/total_fdiff': loss - self.train_prev_loss})
            self.train_prev_loss = loss

            for k in self.train_prev_metric.keys():
                logger.log_metrics({
                    f'metrics/train/{k}_fdiff':
                        state.train_metric_values[k] - self.train_prev_metric[k]
                })

            for k in state.train_metric_values.keys():
                value = state.train_metric_values[k]
                self.train_prev_metric[k] = value

    def eval_end(self, state: State, logger: Logger) -> None:
        if self.diff_eval_metrics:
            evaluator = state.dataloader_label
            assert evaluator is not None, 'dataloader should have been set'

            metrics = list(state.eval_metrics[evaluator].keys())

            for k in metrics:
                mkey = '/'.join(['metrics', evaluator, k])
                if mkey in self.eval_prev_metric.keys():
                    logger.log_metrics({
                        f'{mkey}_fdiff':
                            state.eval_metric_values[k] -
                            self.eval_prev_metric[mkey]
                    })

            for k in metrics:
                mkey = '/'.join(['metrics', evaluator, k])
                self.eval_prev_metric[mkey] = state.eval_metric_values[k]
