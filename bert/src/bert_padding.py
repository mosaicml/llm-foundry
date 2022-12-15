# Copyright 2022 MosaicML Benchmarks authors
# SPDX-License-Identifier: Apache-2.0

# Adapted from https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/bert_padding.py
# Which was adapted from https://github.com/mlcommons/training_results_v1.1/blob/main/NVIDIA/benchmarks/bert/implementations/pytorch/padding.py

"""Helper functions for padding and unpadding batches.

These functions are used extensively throughout the Mosaic BERT implementation
in `bert_layers.py`.
"""

from typing import Tuple

import torch
import torch.nn.functional as F
from einops import rearrange, repeat


class IndexFirstAxis(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input: torch.Tensor,
                indices: torch.Tensor) -> torch.Tensor:
        """Get just the values of `input` which are at `indices`.

        Arguments:
            ctx: the autograd context object
            input: (b, ...) 2+ dimensional tensor
            indices: (num_idx) 1D tensor
        """
        ctx.save_for_backward(indices)
        assert input.ndim >= 2
        ctx.first_axis_dim, other_shape = input.shape[0], input.shape[
            1:]  # type: ignore
        second_dim = other_shape.numel(
        )  # product of sizes of all but first dimension
        # TD [2022-03-04] For some reason torch.gather is a bit faster than indexing.
        return torch.gather(
            rearrange(input, 'b ... -> b (...)'),  # (b, ...) -> (b, second_dim)
            0,
            repeat(indices, 'z -> z d',
                   d=second_dim)  # (indices,) -> (indices, second_dim)
        ).reshape(-1, *other_shape)  # (num_idx, ...)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        indices, = ctx.saved_tensors
        assert grad_output.ndim >= 2
        other_shape = grad_output.shape[1:]
        grad_output = rearrange(grad_output, 'b ... -> b (...)')
        grad_input = torch.zeros([ctx.first_axis_dim, grad_output.shape[1]],
                                 device=grad_output.device,
                                 dtype=grad_output.dtype)
        # TD [2022-03-04] For some reason torch.scatter is a bit faster than indexing.
        # grad_input[indices] = grad_output
        grad_input.scatter_(0,
                            repeat(indices, 'z -> z d', d=grad_output.shape[1]),
                            grad_output)
        return grad_input.reshape(ctx.first_axis_dim, *other_shape), None


index_first_axis = IndexFirstAxis.apply


class IndexPutFirstAxis(torch.autograd.Function):

    @staticmethod
    def forward(ctx, values: torch.Tensor, indices: torch.Tensor,
                first_axis_dim) -> torch.Tensor:
        ctx.save_for_backward(indices)
        assert indices.ndim == 1
        assert values.ndim >= 2
        output = torch.zeros(first_axis_dim,
                             *values.shape[1:],
                             device=values.device,
                             dtype=values.dtype)
        output[indices] = values
        return output

    @staticmethod
    def backward(ctx,
                 grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        indices, = ctx.saved_tensors
        grad_values = grad_output[indices]
        return grad_values, None, None


index_put_first_axis = IndexPutFirstAxis.apply


def unpad_input(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Remove padding from input sequences.

    Arguments:
        hidden_states: (batch, seqlen, ...)
        attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.

    Returns:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz)
        cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index into hidden_states.
        max_seqlen_in_batch: int ()
    """
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32),
                       (1, 0))
    # TD [2022-03-04] We don't want to index with a bool mask, because Pytorch will expand the
    # bool mask, then call nonzero to get the indices, then index with those. The indices is @dim
    # times larger than it needs to be, wasting memory. It's faster and more memory-efficient to
    # index with integer indices. Moreover, torch's index is a bit slower than it needs to be,
    # so we write custom forward and backward to make it a bit faster.
    return (index_first_axis(rearrange(hidden_states, 'b s ... -> (b s) ...'),
                             indices), indices, cu_seqlens, max_seqlen_in_batch
           )  # type: ignore


def unpad_input_only(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Like unpad_input, but only return the unpadded first tensor.

    Save a small amount of overhead.

    Arguments:
        hidden_states: (batch, seqlen, ...)
        attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.

    Returns:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
    """
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    return index_first_axis(rearrange(hidden_states, 'b s ... -> (b s) ...'),
                            indices)


def pad_input(hidden_states: torch.Tensor, indices: torch.Tensor, batch: int,
              seqlen: int) -> torch.Tensor:
    """Add padding to sequences.

    Arguments:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz)
        batch: int batch_size
        seqlen: int max sequence length

    Returns:
        hidden_states: (batch, seqlen, ...)
    """
    output = index_put_first_axis(hidden_states, indices, batch * seqlen)
    return rearrange(output, '(b s) ... -> b s ...', b=batch)
