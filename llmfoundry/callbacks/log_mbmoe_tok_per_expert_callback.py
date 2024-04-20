# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Log tokens per expert for MegaBlocks MoE."""
from __future__ import annotations

import torch
from composer.core import Callback, State
from composer.loggers import Logger
from composer.utils import dist

__all__ = ['MegaBlocksMoE_TokPerExpert']


class MegaBlocksMoE_TokPerExpert(Callback):
    """Log tokens per expert for MegaBlocks MoE.

    To compute the load balancing loss, MegaBlocks caches information including `tokens_per_expert`
    (tpe). At the :attr:`.Event.BATCH_END` event this callback gets load_balancing_loss from
    MegaBlocks to get `tokens_per_expert` then logs statistics (<STAT>) of the number of tokens
    assigned to experts for each layer index (l_idx) under ``mb_moe/layer<l_idx>_<STAT>_tpe``.


    The tokens_per_expert statistics are logged by the :class:`.Logger` to the following keys as
    described below.

    +----------------------------------+-----------------------------------------------------------+
    | Key                              | Logged data                                               |
    +==================================+===========================================================+
    | `mb_moe/alllayer_min_tpe`        | Minimum tokens per expert across all layers               |
    +----------------------------------+-----------------------------------------------------------+
    | `mb_moe/alllayer_max_tpe`        | Maximum tokens per expert across all layers               |
    +----------------------------------+-----------------------------------------------------------+
    | `mb_moe/alllayer_median_tpe`     | Median tokens per expert across all layers                |
    +----------------------------------+-----------------------------------------------------------+
    | `mb_moe/alllayer_std_tpe`        | Standard deviation of tokens per expert across all layers |
    +----------------------------------+-----------------------------------------------------------+
    | `mb_moe/layer<l_idx>_min_tpe`    | Minimum tokens per expert at l_idx layer                  |
    +----------------------------------+-----------------------------------------------------------+
    | `mb_moe/layer<l_idx>_max_tpe`    | Maximum tokens per expert at l_idx layer                  |
    +----------------------------------+-----------------------------------------------------------+
    | `mb_moe/layer<l_idx>_median_tpe` | Median tokens per expert at l_idx layer                   |
    +----------------------------------+-----------------------------------------------------------+
    | `mb_moe/layer<l_idx>_std_tpe`    | Standard deviation of tokens per expert at l_idx layer    |
    +----------------------------------+-----------------------------------------------------------+

    Args:
        log_interval (int, optional): The interval on which to log (Default: 10).
        log_every_layer (bool, optional): Enable logging ever layer's statisictics (True) or log
            only aggregate statistics (Default: False).
        all_reduce_stats (bool, optional): Enable aggregating statistics across gpus (True) or log
            statistics for GPU 0 (Default: False).
        normalize (bool, optional): Normalize token counts by total tokens (Default: True) or output
            raw token count (False). When normalize is True, the callback displays the fraction of
            unique tokens routed to each expert. When normalize is False, the callback displays the
            total number of tokens routed to each expert.
    """

    def __init__(
        self,
        log_interval: int = 10,
        log_every_layer: bool = False,
        all_reduce_stats: bool = False,
        normalize: bool = True,
    ):
        self.log_interval = log_interval
        self.log_every_layer = log_every_layer
        self.all_reduce_stats = all_reduce_stats
        self.normalize = normalize

        self.topk = None

    def fit_start(self, state: State, logger: Logger) -> None:
        if self.topk is None and self.normalize:
            try:
                from megablocks.layers.dmoe import dMoE
                from megablocks.layers.moe import MoE
            except:
                raise RuntimeError(
                    'Requirements for MegaBlocks not installed; see install instructions in `README.md`.'
                )
            for module in state.model.modules():
                if isinstance(module, (MoE, dMoE)):
                    self.topk = module.experts.args.moe_top_k
                    return

            raise RuntimeError(
                f'Callback not initialized correctly; self.topk not instantiated.'
            )

    def batch_end(self, state: State, logger: Logger) -> None:
        if state.timestamp.batch.value % self.log_interval == 0:
            try:
                from megablocks.layers.moe import get_load_balancing_loss
            except:
                raise RuntimeError(
                    'Requirements for MegaBlocks not installed; see install instructions in `README.md`.'
                )
            tokens_per_expert, _ = zip(*get_load_balancing_loss())

            tokens_per_expert = [
                tpe.clone().detach() for tpe in tokens_per_expert
            ]
            if self.all_reduce_stats:
                for tpe in tokens_per_expert:
                    dist.all_reduce(tpe)

            if self.normalize:
                tokens_per_expert = [
                    tpe / (tpe.sum() / self.topk) for tpe in tokens_per_expert
                ]

            all_tokens_per_expert = torch.concat(tokens_per_expert)

            min_tpe = all_tokens_per_expert.min().item()
            max_tpe = all_tokens_per_expert.max().item()
            median_tpe = all_tokens_per_expert.median().item()
            std_tpe = all_tokens_per_expert.float().std().item()

            log_info = {
                f'mb_moe/all_layers_min_tpe': min_tpe,
                f'mb_moe/all_layers_max_tpe': max_tpe,
                f'mb_moe/all_layers_median_tpe': median_tpe,
                f'mb_moe/all_layers_std_tpe': std_tpe,
            }

            if self.log_every_layer:
                for l_idx, tpe_layer in enumerate(tokens_per_expert):

                    min_tpe = tpe_layer.min().item()
                    max_tpe = tpe_layer.max().item()
                    median_tpe = tpe_layer.median().item()
                    std_tpe = tpe_layer.float().std().item()

                    log_info.update({
                        f'mb_moe/layer{l_idx}_min_tpe': min_tpe,
                        f'mb_moe/layer{l_idx}_max_tpe': max_tpe,
                        f'mb_moe/layer{l_idx}_median_tpe': median_tpe,
                        f'mb_moe/layer{l_idx}_std_tpe': std_tpe,
                    })

            logger.log_metrics(log_info)
