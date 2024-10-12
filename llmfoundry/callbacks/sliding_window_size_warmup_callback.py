# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from composer import Callback, Logger, State, Time
from git import Optional

from llmfoundry.utils.warnings import experimental_class

__all__ = [
    'SlidingWindowSizeWarmerUpper',
]


@experimental_class('SlidingWindowSizeWarmerUpper')
class SlidingWindowSizeWarmerUpper(Callback):
    """Warms up the sliding window size for the model based on a schedule.

    Args:
        t_warmup (str|None): Warmup duration for sliding window size, defaults to scheduler.t_warmup.
    """

    def __init__(
        self,
        train_config: dict[str, Any],
        t_warmup: Optional[str | Time] = None,
    ):
        if t_warmup is None:
            if 'scheduler' in train_config and 't_warmup' in train_config[
                'scheduler']:
                t_warmup = train_config['scheduler']['t_warmup']
            else:
                raise ValueError(
                    't_warmup must be provided if t_warmup is not in scheduler config',
                )
        if isinstance(t_warmup, str):
            t_warmup = Time.from_timestring(t_warmup)
        self.t_warmup = t_warmup.BATCH
        self.max_seq_len = train_config['max_seq_len']
        self.orig_sliding_window_size_list = None

    def before_train_batch(self, state: State, logger: Logger):
        del logger
        current_batch = state.timestamp.batch
        if self.orig_sliding_window_size_list is None:
            self.orig_sliding_window_size_list = [None] * len(
                state.model.model.transformer.blocks,
            )
        for idx, block in enumerate(state.model.model.transformer.blocks):
            attn_block = block.norm_attn_norm.attn if hasattr(
                block,
                'norm_attn_norm',
            ) else block.attn
            if self.orig_sliding_window_size_list[idx] is None:
                self.orig_sliding_window_size_list[
                    idx] = attn_block.sliding_window_size
            if self.orig_sliding_window_size_list[idx] != -1:
                attn_block.sliding_window_size = max(
                    self.orig_sliding_window_size_list[idx],
                    self.max_seq_len - (
                        self.max_seq_len -
                        self.orig_sliding_window_size_list[idx]
                    ) * current_batch / self.t_warmup,
                )
            block.norm_attn_norm.attn.sliding_window_size
