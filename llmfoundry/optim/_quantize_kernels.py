# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import math
import warnings
from typing import Optional

import numba
import numpy as np
import torch
from numba import cuda

GROUP_SIZE = 32
BLOCK_SIZE = GROUP_SIZE * GROUP_SIZE  # 1024
MIN_QUANTIZE_SIZE = BLOCK_SIZE

# ignore a warning that's unavoidable with small tensors
warnings.filterwarnings('ignore', 'NumbaPerformanceWarning')


def _cdiv(x: int, y: int):
    return int(math.ceil(x / y))


@cuda.jit(device=True)
def warp_allreduce_max(x):
    x = max(x, cuda.shfl_xor_sync(-1, x, 1 << 0))
    x = max(x, cuda.shfl_xor_sync(-1, x, 1 << 1))
    x = max(x, cuda.shfl_xor_sync(-1, x, 1 << 2))
    x = max(x, cuda.shfl_xor_sync(-1, x, 1 << 3))
    x = max(x, cuda.shfl_xor_sync(-1, x, 1 << 4))
    return x


def encode_signed(x: torch.Tensor,
                  x_q_out: Optional[torch.Tensor] = None,
                  scales_out: Optional[torch.Tensor] = None,
                  scale_scales_out: Optional[torch.Tensor] = None,
                  simple_scaling: bool = True,
                  nbits: int = 8):
    GROUP_SIZE = 32
    BLOCK_SIZE = GROUP_SIZE * GROUP_SIZE
    assert nbits <= 8

    input_size = x.numel()
    out_shape = x.shape
    x = x.ravel()
    BLOCK_SIZE = GROUP_SIZE * GROUP_SIZE
    if x_q_out is None:
        x_q_out = torch.empty(input_size, dtype=torch.int8, device=x.device)
    if scales_out is None:
        scales_shape = (_cdiv(input_size, BLOCK_SIZE) * GROUP_SIZE, 2)
        scales_out = torch.empty(scales_shape,
                                 dtype=torch.uint8,
                                 device=x.device)
    if scale_scales_out is None:
        scale_scales_shape = (_cdiv(input_size, BLOCK_SIZE),)
        scale_scales_out = torch.empty(scale_scales_shape,
                                       dtype=torch.float32,
                                       device=x.device)

    if x.dtype == torch.bfloat16:  # numba doesn't know about bf16
        x = x.to(dtype=torch.float32)

    assert x.is_cuda
    assert x_q_out.is_cuda
    assert scales_out.is_cuda
    assert scale_scales_out.is_cuda

    # this isn't needed for correctness; just reduces overhead for small arrays
    x_cu = cuda.as_cuda_array(x)
    x_q_out_cu = cuda.as_cuda_array(x_q_out)
    scales_out_cu = cuda.as_cuda_array(scales_out)
    scale_scales_out_cu = cuda.as_cuda_array(scale_scales_out)

    if simple_scaling:
        scale_by = 2**(nbits - 1) - 1 + 1e-4  # assumes round to nearest
    else:
        scale_by = 2**(nbits - 1) - 1e-4  # assumes round down

    grid_shape = (_cdiv(input_size, BLOCK_SIZE),)
    block_shape = (BLOCK_SIZE,)  # now it's 1 load + store per thread
    # encode_numba[grid_shape, block_shape, 0, 2048](  # stream, smem bytes
    encode_numba[grid_shape, block_shape](x_cu, x_q_out_cu, scales_out_cu,
                                          scale_scales_out_cu, scale_by)

    return x_q_out.view(out_shape), scales_out, scale_scales_out


def decode_signed(x_q: torch.Tensor,
                  scales: torch.Tensor,
                  scale_scales: torch.Tensor,
                  x_out: Optional[torch.Tensor] = None,
                  simple_scaling: bool = True,
                  nbits: int = 8) -> torch.Tensor:
    GROUP_SIZE = 32
    BLOCK_SIZE = GROUP_SIZE * GROUP_SIZE
    assert nbits <= 8

    input_size = x_q.numel()
    out_shape = x_q.shape
    x_q = x_q.ravel()
    if x_out is None:
        x_out = torch.empty(input_size, dtype=torch.float32, device=x_q.device)
    x_out = x_out.ravel()

    # this isn't needed for correctness, but reduces time up to ~1.5x for
    # sufficiently small arrays
    x_q_cu = cuda.as_cuda_array(x_q)
    scales_cu = cuda.as_cuda_array(scales)
    scale_scales_cu = cuda.as_cuda_array(scale_scales)
    x_out_cu = cuda.as_cuda_array(x_out)

    if simple_scaling:
        scale_by = 1. / (2.**(nbits - 1) - 1)  # round to nearest
    else:
        scale_by = 2.**(-nbits + 1)  # round down

    grid_shape = (_cdiv(input_size, BLOCK_SIZE),)
    # block_shape = (GROUP_SIZE,)
    block_shape = (BLOCK_SIZE,)
    decode_numba[grid_shape, block_shape](x_q_cu, scales_cu, scale_scales_cu,
                                          x_out_cu, scale_by)

    return x_out.view(out_shape)


@cuda.jit()
def encode_numba(x, x_q_out, scales_out, scale_scales_out,
                 encode_scale_by: float):
    GROUP_SIZE = 32
    BLOCK_SIZE = 1024

    numel = len(x)
    bx = cuda.blockIdx.x
    tx = cuda.threadIdx.x
    laneid = cuda.laneid
    my_group = tx // GROUP_SIZE
    global_idx = bx * BLOCK_SIZE + tx

    # cooperatively load stuff into shared mem; each of the 32 threads
    # in the warp loads one of the 32 rows in our block
    block = cuda.shared.array(BLOCK_SIZE, dtype=np.float16)
    # my_val = x[global_idx] if global_idx < len(x) else 0
    if global_idx < numel:
        my_val = x[global_idx]
    else:
        my_val = 0.
    my_val = float(my_val)
    abs_sqrt = math.sqrt(abs(my_val))
    my_val = math.copysign(abs_sqrt, my_val)
    block[tx] = my_val

    # compute the maxabs value in this row
    row_maxabs = warp_allreduce_max(abs_sqrt)

    cuda.syncthreads()  # need full block in shared mem to get col maxs

    # compute col maxes
    # each thread in the warp loads up one value in the column; then we find
    # the max value within the col and write it to shared mem
    col_idx = my_group
    my_val = block[laneid * GROUP_SIZE + col_idx]
    col_maxabs = warp_allreduce_max(abs(my_val))

    col_maxs = cuda.shared.array((GROUP_SIZE,), dtype=np.float32)
    if laneid == 0:
        col_maxs[col_idx] = col_maxabs
    cuda.syncthreads()
    my_col_absmax = col_maxs[laneid]

    # quantize and write out my element
    if global_idx < numel:
        scale = min(row_maxabs, my_col_absmax)
        # scale = row_maxabs  # much higher cossim, so col maxs are prolly messed up
        x_q_val = (float(block[tx]) / scale) * encode_scale_by
        x_q_out[global_idx] = round(x_q_val)

    # compute the maximum scale; could be a for loop, but let's do it
    # the "right" way and allreduce within our warp. It suffices to look
    # at only the col maxs since the largest elem in the whole block will
    # always be the largest elem in some column.
    max_scale = warp_allreduce_max(my_col_absmax)

    # write out the scales and scale scale for this thread's row and column
    if tx == 0:
        scale_scales_out[bx] = max_scale
    if laneid == 0:
        # zeroth thread of each warp writes; has to have a writer from
        # each warp since warps only know about their own row scales
        my_scales_row = bx * GROUP_SIZE + my_group
        scales_out[my_scales_row, 0] = round((row_maxabs / max_scale) * 255)
        scales_out[my_scales_row, 1] = round(
            (col_maxs[my_group] / max_scale) * 255)


@cuda.jit()
def decode_numba(x_q, scales, scale_scales, x_hat_out, decode_scale_by: float):
    GROUP_SIZE = 32
    BLOCK_SIZE = GROUP_SIZE * GROUP_SIZE

    bx = cuda.blockIdx.x
    tx = cuda.threadIdx.x
    laneid = cuda.laneid
    my_group = tx // GROUP_SIZE
    global_idx = bx * BLOCK_SIZE + tx

    # load this group's row absmax and all the col absmaxs for this group
    scale_scale = scale_scales[bx]
    our_scales_start = bx * GROUP_SIZE
    my_row_absmax = (float(scales[our_scales_start + my_group, 0]) /
                     255) * scale_scale
    my_col_absmax = (float(scales[our_scales_start + laneid, 1]) /
                     255) * scale_scale

    # dequantize + write out our row
    if global_idx < len(x_q):
        x_int = x_q[global_idx]
        scale = min(my_row_absmax, my_col_absmax)
        x_sqrt = decode_scale_by * float(x_int) * scale
        x_hat = math.copysign(x_sqrt * x_sqrt, x_sqrt)
        x_hat_out[global_idx] = x_hat


# ================================================================
# Quantized LION kernel
# ================================================================


@cuda.jit(fastmath=True)
def _lion_step_numba(
        momentum_quantized: torch.Tensor,
        scales: torch.Tensor,
        scale_scales: torch.Tensor,
        weights: torch.Tensor,
        grads: torch.Tensor,
        lr: float,
        step_coef: float,  # beta1
        momentum_coef: float,  # beta2
        l2_penalty: float = 0,
        weight_decay: float = 0) -> None:
    GROUP_SIZE = 32
    BLOCK_SIZE = 1024  # GROUP_SIZE * GROUP_SIZE, but needs to be a raw int
    NBITS = 8
    ENCODE_SCALE_BY = 2**(NBITS - 1) - 1 + 1e-4  # assumes round to nearest
    DECODE_SCALE_BY = 1. / (2.**(NBITS - 1) - 1)  # round to nearest

    mom_q = momentum_quantized

    numel = len(mom_q)
    bx = cuda.blockIdx.x
    tx = cuda.threadIdx.x
    laneid = cuda.laneid
    my_group = tx // GROUP_SIZE
    global_idx = bx * BLOCK_SIZE + tx

    # ------------------------------------------------ decode

    # load this group's row absmax and all the col absmaxs for this group
    scale_scale = scale_scales[bx]
    our_scales_start = bx * GROUP_SIZE
    my_row_absmax = (float(scales[our_scales_start + my_group, 0]) /
                     255) * scale_scale
    my_col_absmax = (float(scales[our_scales_start + laneid, 1]) /
                     255) * scale_scale

    # dequantize our row + run our opt step
    block = cuda.shared.array(BLOCK_SIZE, dtype=np.float16)
    abs_sqrt = 0
    if global_idx >= numel:
        block[tx] = 0
    else:
        m_int = mom_q[global_idx]
        g = np.float32(grads[global_idx])  # might as well use higher precision
        w = np.float32(weights[global_idx])

        # decode compressed momentum
        scale = min(my_row_absmax, my_col_absmax)
        m_sqrt = DECODE_SCALE_BY * np.float32(m_int) * scale
        mom = math.copysign(m_sqrt * m_sqrt, m_sqrt)

        # ------------------------------------------------ opt step
        g += l2_penalty * w
        c = step_coef * (mom - g) + g
        update = math.copysign(1, c)  # workaround lack of sign() function
        update = update * (c != 0)  # copysign(1, 0) = 1, unlike torch.sign
        w -= w * weight_decay
        w -= lr * update
        weights[global_idx] = w

        # update the momentum; we write to shared mem instead of DRAM since
        # we need to encode the momentum; note that this happens *after*
        # we add the l2 penalty into the gradient
        # new_mom = (momentum_coef * mom) + ((1. - momentum_coef) * g_val)
        new_mom = momentum_coef * (mom - g) + g  # simplification of above
        abs_sqrt = math.sqrt(abs(new_mom))
        block[tx] = math.copysign(abs_sqrt, new_mom)

    # ------------------------------------------------ encode
    # now we just need to write out our quantized momentum, quantized scales,
    # and scale scales

    # compute the maxabs value in this row
    row_maxabs = warp_allreduce_max(abs_sqrt)

    cuda.syncthreads()  # need full block in shared mem to get col maxs

    # compute col maxes
    # each thread in the warp loads up one value in the column; then we find
    # the max value within the col and write it to shared mem
    col_idx = my_group
    my_val = block[laneid * GROUP_SIZE + col_idx]
    col_maxabs = warp_allreduce_max(abs(my_val))

    col_maxs = cuda.shared.array((GROUP_SIZE,), dtype=np.float32)
    if laneid == 0:
        col_maxs[col_idx] = col_maxabs
    cuda.syncthreads()
    my_col_absmax = col_maxs[laneid]

    # quantize and write out my momentum
    if global_idx < numel:
        scale = min(row_maxabs, my_col_absmax)
        # scale = row_maxabs  # much higher cossim, so col maxs are prolly messed up
        m_q_val = (float(block[tx]) / scale) * ENCODE_SCALE_BY
        mom_q[global_idx] = round(m_q_val)

    # compute the maximum scale; It suffices to look at only the col maxs
    # since the largest elem in the whole block will always be the largest
    # elem in some column.
    max_scale = warp_allreduce_max(my_col_absmax)

    # write out the scales and scale scale for this thread's row and column
    if tx == 0:
        scale_scales[bx] = max_scale
    if laneid == 0:
        # zeroth thread of each warp writes; has to have a writer from
        # each warp since warps only know about their own row scales
        my_scales_row = bx * GROUP_SIZE + my_group
        scales[my_scales_row, 0] = round((row_maxabs / max_scale) * 255)
        scales[my_scales_row, 1] = round((col_maxs[my_group] / max_scale) * 255)


def lion_step_fused(momentums_quantized: torch.Tensor,
                    momentums_scales: torch.Tensor,
                    momentums_scale_scales: torch.Tensor,
                    weights: torch.Tensor,
                    grads: torch.Tensor,
                    lr: float,
                    beta1: float,
                    beta2: float,
                    l2_penalty: float = 0,
                    weight_decay: float = 0) -> None:
    grid_shape = (_cdiv(momentums_quantized.numel(), BLOCK_SIZE),)
    block_shape = (BLOCK_SIZE,)

    # workaround numba cuda's inability to accept bf16 inputs
    orig_weights = weights
    if orig_weights.dtype == torch.bfloat16:
        weights = weights.to(dtype=torch.float32)
    if grads.dtype == torch.bfloat16:
        grads = grads.to(dtype=torch.float32)

    # explicit conversion to cuda arrays isn't needed for correctness, but
    # it reduces the call overhead by ~2x at small input sizes
    _lion_step_numba[grid_shape, block_shape](
        cuda.as_cuda_array(momentums_quantized.ravel()),
        cuda.as_cuda_array(momentums_scales),
        cuda.as_cuda_array(momentums_scale_scales),
        cuda.as_cuda_array(weights.detach().ravel()),
        cuda.as_cuda_array(grads.ravel()), lr, beta1, beta2, l2_penalty,
        weight_decay)

    if orig_weights.dtype == torch.bfloat16:
        orig_weights.copy_(weights)
