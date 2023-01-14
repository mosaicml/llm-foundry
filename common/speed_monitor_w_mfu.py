# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Monitor throughput during training."""
from __future__ import annotations

import warnings
from collections import deque
from typing import Any, Deque, Dict, Optional, Union

import torch
from composer.core import Callback, State
from composer.loggers import Logger
from composer.models.base import ComposerModel
from composer.utils import dist
from torch.utils.data import DataLoader

GPU_AVAILABLE_FLOPS = {
    # source: https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet
    # nvidia publishes spec sheet with a 2x sparsity factor
    'h100-sxm': {
        'fp64': 67e12,
        'fp32': 67e12,
        'tf32': 989e12 / 2,
        'fp16': 1.979e15 / 2,
        'amp_fp16': 1.979e15 / 2,
        'bf16': 1.979e15 / 2,
        'amp_bf16': 1.979e15 / 2,
        'fp8': 3.958e15 / 2,
        'amp_fp8': 3.958e15 / 2,
        'int8': 3.958e15 / 2,
    },
    'h100-pcie': {
        'fp64': 51e12,
        'fp32': 51e12,
        'tf32': 756e12 / 2,
        'fp16': 1.513e15 / 2,
        'amp_fp16': 1.513e15 / 2,
        'bf16': 1.513e15 / 2,
        'amp_bf16': 1.513e15 / 2,
        'fp8': 3.026e15 / 2,
        'amp_fp8': 3.026e15 / 2,
        'int8': 3.026e15 / 2,
    },
    # source: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf
    # sxm and pcie have same flop counts
    'a100': {
        'fp64': 19.5e12,
        'fp32': 19.5e12,
        'tf32': 156e12,
        'fp16': 312e12,
        'amp_fp16': 312e12,
        'bf16': 312e12,
        'amp_bf16': 312e12,
    },
    # source: https://images.nvidia.com/content/technologies/volta/pdf/volta-v100-datasheet-update-us-1165301-r5.pdf
    'v100-sxm': {
        'fp64': 7.8e12,
        'fp32': 15.7e12,
        'fp16': 125e12,
        'amp_fp16': 125e12,
    },
    'v100-pcie': {
        'fp64': 7e12,
        'fp32': 14e12,
        'fp16': 112e12,
        'amp_fp16': 112e12,
    },
    'v100s-pcie': {
        'fp64': 8.2e12,
        'fp32': 16.4e12,
        'fp16': 130e12,
        'amp_fp16': 130e12,
    },
    # source: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf
    # sxm and pcie have same flop counts
    't4': {
        'fp32': 8.1e12,
        'fp16': 65e12,
        'amp_fp16': 65e12,
        'int8': 130e12,
        'int4': 260e12,
    },
}

__all__ = ['SpeedMonitorMFU']


def get_gpu_flops_available(state: State):
    gpu_flops_available = None

    # torch.cuda.get_device_name() ex output: 'NVIDIA A100-SXM4-40GB'
    dev_name = torch.cuda.get_device_name().lower()
    if 'h100-sxm' in dev_name:
        dev_name = 'h100-sxm'
    elif 'h100-pcie' in dev_name:
        dev_name = 'h100-pcie'
    elif 'a100' in dev_name:
        dev_name = 'a100'
    elif 'v100-sxm' in dev_name:
        dev_name = 'v100-sxm'
    elif 'v100-pcie' in dev_name:
        dev_name = 'v100-pcie'
    elif 't4' in dev_name:
        dev_name = 't4'
    else:
        dev_name = None

    if dev_name:
        try:
            gpu_flops_available = int(
                GPU_AVAILABLE_FLOPS[dev_name][state.precision.value])
        except:
            gpu_flops_available = None

    if gpu_flops_available is None:
        warnings.warn(
            f'gpu_flop count not found for {dev_name=} with precision: {state.precision.value}; MFU cannot be calculated and reported. '
            f'gpu_flops_available can be manually overridden by setting gpu_flops_available in SpeedMonitorMFU()'
        )

    return gpu_flops_available


class SpeedMonitorMFU(Callback):
    """Logs the training throughput and MFU."""

    def __init__(self,
                 window_size: int = 100,
                 gpu_flops_available: Optional[Union[float, int]] = None):
        # Track the batch num samples and wct to compute throughput over a window of batches
        self.batch_start_num_samples = 0
        self.batch_start_wct = 0.0
        self.batch_wct_buffer: Deque[float] = deque(maxlen=window_size)
        self.batch_num_samples_buffer: Deque[int] = deque(maxlen=window_size)
        self.window_size = window_size

        self.set_gpu_flops_available = False
        self.gpu_flops_available = gpu_flops_available

        # Keep track of time spent evaluating
        self.total_eval_wct = 0.0

    def state_dict(self) -> Dict[str, Any]:
        return {
            'batch_start_num_samples': self.batch_start_num_samples,
            'batch_start_wct': self.batch_start_wct,
            'batch_wct_buffer': self.batch_wct_buffer,
            'batch_num_samples_buffer': self.batch_num_samples_buffer,
            'total_eval_wct': self.total_eval_wct,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.batch_start_num_samples = state['batch_start_num_samples']
        self.batch_start_wct = state['batch_start_wct']
        self.batch_wct_buffer = deque(
            [x for x in state['batch_wct_buffer']],
            maxlen=self.window_size,
        )
        self.batch_num_samples_buffer = deque(
            [x for x in state['batch_num_samples_buffer']],
            maxlen=self.window_size,
        )
        self.total_eval_wct = state['total_eval_wct']

    def before_dataloader(self, state: State, logger: Logger) -> None:
        del logger  # unused
        self.batch_start_wct = state.timestamp.total_wct.total_seconds()
        self.batch_start_num_samples = int(state.timestamp.sample)

        # Get available GPU FLOPS
        if not self.gpu_flops_available:
            self.gpu_flops_available = get_gpu_flops_available(state)

    def batch_end(self, state: State, logger: Logger):
        batch_num_samples = int(
            state.timestamp.sample) - self.batch_start_num_samples
        batch_wct = state.timestamp.total_wct.total_seconds(
        ) - self.batch_start_wct

        # Add the new element
        self.batch_wct_buffer.append(batch_wct)
        self.batch_num_samples_buffer.append(batch_num_samples)

        # Log the throughput
        if len(self.batch_num_samples_buffer) == self.window_size:
            world_size = dist.get_world_size()
            batches_per_sec = len(self.batch_num_samples_buffer) / sum(
                self.batch_wct_buffer)
            samples_per_sec = sum(self.batch_num_samples_buffer) / sum(
                self.batch_wct_buffer)
            dev_batches_per_sec = batches_per_sec / world_size
            dev_samples_per_sec = samples_per_sec / world_size
            logger.log_metrics({'throughput/batches_per_sec': batches_per_sec})
            logger.log_metrics({'throughput/samples_per_sec': samples_per_sec})
            logger.log_metrics(
                {'throughput/device/batches_per_sec': dev_batches_per_sec})
            logger.log_metrics(
                {'throughput/device/samples_per_sec': dev_samples_per_sec})

            if isinstance(state.dataloader, DataLoader) and hasattr(
                    state.dataloader.dataset, 'max_seq_len'):
                max_seq_len = state.dataloader.dataset.max_seq_len  # type: ignore
                # only applicable to seq data / models
                logger.log_metrics({
                    'throughput/tokens_per_sec': samples_per_sec * max_seq_len
                })
                logger.log_metrics({
                    'throughput/device/tokens_per_sec':
                        dev_samples_per_sec * max_seq_len
                })

            composer_model = state.model
            if not isinstance(composer_model, ComposerModel):
                composer_model = composer_model.module  # Handles DDP case until Composer fixes this
            if hasattr(composer_model, 'num_fwd_flops'):
                assert isinstance(composer_model.num_fwd_flops, (int, float))
                # Assume that training flops is 3x fwd flops (1 fwd, 2 bkw)
                flops_per_sec = 3 * composer_model.num_fwd_flops * samples_per_sec
                logger.log_metrics({'throughput/flops_per_sec': flops_per_sec})
                dev_flops_per_sec = flops_per_sec / world_size
                logger.log_metrics(
                    {'throughput/device/flops_per_sec': dev_flops_per_sec})
                if self.gpu_flops_available:
                    mfu = dev_flops_per_sec / self.gpu_flops_available
                    logger.log_metrics({'throughput/device/mfu': mfu})

        # Log the time
        # `state.timestamp` excludes any time spent in evaluation
        logger.log_metrics({
            'wall_clock/train':
                state.timestamp.total_wct.total_seconds(),
            'wall_clock/val':
                self.total_eval_wct,
            'wall_clock/total':
                (state.timestamp.total_wct.total_seconds() + self.total_eval_wct
                ),
        })

    def eval_end(self, state: State, logger: Logger):
        del logger  # unused
        self.total_eval_wct += state.eval_timestamp.total_wct.total_seconds()
