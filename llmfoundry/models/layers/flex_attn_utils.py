# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from functools import partial
from typing import Any, Optional

import torch
from packaging import version
from torch.nn.attention.flex_attention import (
    _DEFAULT_SPARSE_BLOCK_SIZE,
    _score_mod_signature,
    and_masks,
)

from llmfoundry.layers_registry import flex_attention_mods


class FlexAttentionMod(ABC):

    def _mask_mod_fn(
        self,
        b: torch.Tensor,
        h: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
        query_offset: torch.Tensor,
        sequence_id_info: Optional[dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        del sequence_id_info, query_offset, b, h, q_idx, kv_idx
        raise NotImplementedError

    def _score_mod_fn(
        self,
        score: torch.Tensor,
        b: torch.Tensor,
        h: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
        query_offset: torch.Tensor,
        sequence_id_info: Optional[dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        del sequence_id_info, query_offset, score, b, h, q_idx, kv_idx
        raise NotImplementedError

    def __init__(self, mod_type: str) -> None:
        assert mod_type in ['mask', 'score']
        self.mod_type = mod_type
        self.mod_fn = self._mask_mod_fn if mod_type == 'mask' else self._score_mod_fn


@flex_attention_mods.register('causal_mask')
class CausalMaskMod(FlexAttentionMod):

    def _mask_mod_fn(
        self,
        b: torch.Tensor,
        h: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
        query_offset: torch.Tensor,
        sequence_id_info: Optional[dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        del sequence_id_info, b, h
        q_idx = q_idx + query_offset
        return q_idx >= kv_idx

    def __init__(self) -> None:
        super().__init__(mod_type='mask')


@flex_attention_mods.register('sliding_window_mask')
class SlidingWindowMaskMod(FlexAttentionMod):

    def _mask_mod_fn(
        self,
        b: torch.Tensor,
        h: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
        query_offset: torch.Tensor,
        sequence_id_info: Optional[dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        del sequence_id_info, b, h
        q_idx = q_idx + query_offset
        return torch.abs(q_idx - kv_idx) <= self.sliding_window_size

    def __init__(self, sliding_window_size: torch.Tensor) -> None:
        super().__init__(mod_type='mask')
        self.sliding_window_size = sliding_window_size


@flex_attention_mods.register('sequence_id_mask')
class SequenceIdMaskMod(FlexAttentionMod):

    def _mask_mod_fn(
        self,
        b: torch.Tensor,
        h: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
        query_offset: torch.Tensor,
        sequence_id_info: Optional[dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        del h
        q_idx = q_idx + query_offset
        if sequence_id_info is None:
            raise ValueError(
                'sequence_id_info is required for SequenceIdMaskMod',
            )
        sequence_id = sequence_id_info['sequence_id']
        # Check if the query and key belong to the same sequence and the query token is not a padding token.
        return (sequence_id[b, q_idx]
                == sequence_id[b, kv_idx]) & (sequence_id[b, q_idx] != -1)

    def __init__(self) -> None:
        super().__init__(mod_type='mask')


@flex_attention_mods.register('attention_mask')
class AttentionMaskMod(FlexAttentionMod):

    def _mask_mod_fn(
        self,
        b: torch.Tensor,
        h: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
        query_offset: torch.Tensor,
        sequence_id_info: Optional[dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        del h, q_idx, query_offset
        if sequence_id_info is None:
            raise ValueError(
                'sequence_id_info is required for SequenceIdMaskMod',
            )
        attention_mask = sequence_id_info['attention_mask']
        # Check if the query and key belong to the same sequence and the query token is not a padding token.
        return attention_mask[b, kv_idx]

    def __init__(self) -> None:
        super().__init__(mod_type='mask')


@flex_attention_mods.register('local_global_mask')
class LocalGlobalMaskMod(FlexAttentionMod):

    def _mask_mod_fn(
        self,
        b: torch.Tensor,
        h: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
        query_offset: torch.Tensor,
        sequence_id_info: Optional[dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        del h
        q_idx = q_idx + query_offset
        if sequence_id_info is None:
            raise ValueError(
                'sequence_id_info is required for LocalGlobalMaskMod',
            )
        pos_in_seq = sequence_id_info['pos_in_seq']
        # Check if the query and key belong to the same sequence and the query token is not a padding token.

        if pos_in_seq is not None:
            global_window_mask = (
                pos_in_seq[b, kv_idx] <= self.global_window_size
            )
        else:
            global_window_mask = (kv_idx <= self.global_window_size)
        sliding_window_mask = (q_idx - kv_idx <= self.sliding_window_size)

        return global_window_mask | sliding_window_mask

    def __init__(
        self,
        sliding_window_size: int,
        global_window_size: int,
    ) -> None:
        super().__init__(mod_type='mask')
        self.sliding_window_size = sliding_window_size
        self.global_window_size = global_window_size


@flex_attention_mods.register('alibi_score_mod')
class AlibiScoreMod(FlexAttentionMod):

    def _score_mod_fn(
        self,
        score: torch.Tensor,
        b: torch.Tensor,
        h: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
        query_offset: torch.Tensor,
        sequence_id_info: Optional[dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        del sequence_id_info, b
        q_idx = q_idx + query_offset
        bias = -self.alibi_slopes[h] * torch.abs(q_idx - kv_idx)
        return score + bias

    def __init__(self, alibi_slopes: torch.Tensor) -> None:
        super().__init__(mod_type='score')
        self.alibi_slopes = alibi_slopes


@flex_attention_mods.register('softcap_score_mod')
class SoftcapScoreMod(FlexAttentionMod):

    def _score_mod_fn(
        self,
        score: torch.Tensor,
        b: torch.Tensor,
        h: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
        query_offset: torch.Tensor,
        sequence_id_info: Optional[dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        del sequence_id_info, query_offset, b, h, q_idx, kv_idx
        return self.attn_logit_softcapping * torch.tanh(
            score / self.attn_logit_softcapping,
        )

    def __init__(self, attn_logit_softcapping: torch.Tensor) -> None:
        super().__init__(mod_type='score')
        self.attn_logit_softcapping = attn_logit_softcapping


def generate_block_mask(
    Q_LEN: int,
    KV_LEN: int,
    B: int,
    block_mask_list: Optional[list[FlexAttentionMod]],
    compiled_create_block_mask: Any,
    query_offset: torch.Tensor,
    sequence_id_info: Optional[dict[str, torch.Tensor]],
):
    if block_mask_list is None:
        return None

    block_mask_fn = None
    for i, block_mask in enumerate(block_mask_list):
        if i == 0:
            block_mask_fn = partial(
                block_mask.mod_fn,
                query_offset=query_offset,
                sequence_id_info=sequence_id_info,
            )
        else:
            block_mask_fn = and_masks(
                block_mask_fn, # type: ignore
                partial(
                    block_mask.mod_fn,
                    query_offset=query_offset,
                    sequence_id_info=sequence_id_info,
                ),
            )

    extra_args = {}
    if version.parse(
        torch.__version__.split('.dev')[0],
    ) < version.parse('2.6.0') and Q_LEN % _DEFAULT_SPARSE_BLOCK_SIZE != 0:
        extra_args['BLOCK_SIZE'] = Q_LEN
    block_mask = compiled_create_block_mask(
        block_mask_fn,
        B=B,
        H=None, # Setting this to None speeds up block mask generation, but this means the mask has to be the same across all heads.
        Q_LEN=Q_LEN,
        KV_LEN=KV_LEN,
        **extra_args,
    )

    return block_mask


def generate_score_mod(
    score_mod_list: Optional[list[FlexAttentionMod]],
    query_offset: torch.Tensor,
    sequence_id_info: Optional[dict[str, torch.Tensor]],
):
    if score_mod_list is None:
        return None
    wrapped_score_mod = None
    for i, score_mod in enumerate(score_mod_list):
        if i == 0:
            wrapped_score_mod = partial(
                score_mod.mod_fn,
                query_offset=query_offset,
                sequence_id_info=sequence_id_info,
            )
        else:
            wrapped_score_mod = _wrap_score_mod_fns(
                wrapped_score_mod, # type: ignore
                partial(
                    score_mod.mod_fn,
                    query_offset=query_offset,
                    sequence_id_info=sequence_id_info,
                ),
            )

    return wrapped_score_mod


def _wrap_score_mod_fns(
    score_mod_fn_1: _score_mod_signature,
    score_mod_fn_2: _score_mod_signature,
) -> _score_mod_signature:

    def wrapped_score_mod_fn(
        score: torch.Tensor,
        b: torch.Tensor,
        h: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
    ) -> torch.Tensor:
        score = score_mod_fn_1(score, b, h, q_idx, kv_idx)
        score = score_mod_fn_2(score, b, h, q_idx, kv_idx)
        return score

    return wrapped_score_mod_fn
