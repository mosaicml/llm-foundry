# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

try:
    from examples.common.builders import (build_algorithm, build_callback,
                                          build_dataloader, build_logger,
                                          build_optimizer, build_scheduler)
    from examples.common.config_utils import (calculate_batch_size_info,
                                              log_config,
                                              update_batch_size_info)
    from examples.common.speed_monitor_w_mfu import (SpeedMonitorMFU,
                                                     get_gpu_flops_available)
    from examples.common.text_data import (StreamingTextDataset,
                                           build_text_dataloader)
except ImportError as e:
    raise ImportError(
        'Please make sure to pip install . to get the common requirements for all examples.'
    ) from e

__all__ = [
    'build_callback',
    'build_logger',
    'build_algorithm',
    'build_optimizer',
    'build_scheduler',
    'build_dataloader',
    'calculate_batch_size_info',
    'update_batch_size_info',
    'log_config',
    'get_gpu_flops_available',
    'SpeedMonitorMFU',
    'StreamingTextDataset',
    'build_text_dataloader',
]
