# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from composer import Logger, State, TimeUnit

from llmfoundry.interfaces import CallbackWithConfig
from llmfoundry.utils.warnings import experimental_class

__all__ = [
    'SlidingWindowSizeWarmerUpper',
]


@experimental_class('SlidingWindowSizeWarmerUpper')
class SlidingWindowSizeWarmerUpper(CallbackWithConfig):
    """Warms up the sliding window size for the model based on a schedule.

    Args:
        train_config (dict[str, Any]): Training configuration.
        t_warmup (float): Warmup duration (as a fraction of max duration) for sliding window size.
    """

    def __init__(
        self,
        train_config: dict[str, Any],
        t_warmup: float,
    ):
        self.t_warmup = t_warmup
        self.max_seq_len = train_config['max_seq_len']
        self.orig_sliding_window_size_list = None

    def before_train_batch(self, state: State, logger: Logger):
        del logger
        if state.max_duration is None:
            raise ValueError(
                'SlidingWindowSizeWarmerUpper callback requires max_duration to be set.',
            )
        if state.max_duration.unit == TimeUnit.TOKEN:
            current_time_frac = state.timestamp.token.value / state.max_duration.value
        elif state.max_duration.unit == TimeUnit.BATCH:
            current_time_frac = state.timestamp.batch.value / state.max_duration.value
        else:
            raise ValueError(
                f'Unsupported time unit {state.max_duration.unit=} for SlidingWindowSizeWarmerUpper callback.',
            )
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
            if self.orig_sliding_window_size_list[idx] != -1:  # type: ignore
                attn_block.sliding_window_size = round(
                    max(
                        self.orig_sliding_window_size_list[idx],
                        self.max_seq_len - (
                            self.max_seq_len -
                            self.orig_sliding_window_size_list[idx]
                        ) * current_time_frac / self.t_warmup,
                    ),
                )
