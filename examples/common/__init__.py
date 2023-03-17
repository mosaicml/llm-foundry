# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

try:
    from examples.common.builders import (build_algorithm, build_callback,
                                          build_dataloader,
                                          build_icl_evaluators, build_logger,
                                          build_optimizer, build_scheduler)
    from examples.common.config_utils import (calculate_batch_size_info,
                                              log_config,
                                              update_batch_size_info)
    from examples.common.hf_fsdp import (prepare_hf_causal_lm_model_for_fsdp,
                                         prepare_hf_enc_dec_model_for_fsdp,
                                         prepare_hf_model_for_fsdp)
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
    'build_icl_evaluators',
    'calculate_batch_size_info',
    'update_batch_size_info',
    'log_config',
    'StreamingTextDataset',
    'build_text_dataloader',
    'prepare_hf_causal_lm_model_for_fsdp',
    'prepare_hf_enc_dec_model_for_fsdp',
    'prepare_hf_model_for_fsdp',
]
