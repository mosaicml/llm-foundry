# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import os
import warnings

import pytest
import torch
from omegaconf import OmegaConf as om

from examples.llm.main import main


def gpt_tiny_cfg(conf_path='yamls/mosaic_gpt/125m.yaml'):
    """Create gpt tiny cfg."""
    with open(conf_path) as f:
        test_cfg = om.load(f)
    # removes requirement to download / process train set
    test_cfg.train_loader.dataset = test_cfg.eval_loader.dataset

    test_cfg.global_train_batch_size = 8
    test_cfg.device_eval_batch_size = 4
    test_cfg.device_train_microbatch_size = 4

    test_cfg.max_duration = '4ba'
    test_cfg.eval_interval = '4ba'
    test_cfg.eval_loader.eval_subset_num_batches = 2
    test_cfg.save_interval = '4ba'
    test_cfg.run_name = 'gpt-mini-integration-test'
    test_cfg.model.d_model = 32
    test_cfg.model.n_heads = 2
    test_cfg.model.n_layers = 2
    test_cfg.max_seq_len = 256
    test_cfg.model.max_seq_len = test_cfg.max_seq_len
    test_cfg.tokenizer.kwargs.model_max_length = test_cfg.max_seq_len
    test_cfg.train_loader.dataset.max_seq_len = test_cfg.max_seq_len
    test_cfg.eval_loader.dataset.max_seq_len = test_cfg.max_seq_len

    return test_cfg


@pytest.mark.parametrize('device', [
    'cpu',
    pytest.param('cuda',
                 marks=pytest.mark.skipif(
                     not torch.cuda.is_available(),
                     reason='testing with cuda requires GPU')),
])
@pytest.mark.parametrize('logit_scale', [None, 0.036, 'inv_sqrt_d_model'])
def test_train(device, logit_scale):
    if not os.path.isdir('./my-copy-c4/val'):
        pytest.xfail('c4 dataset not set up as expected')

    warnings.filterwarnings(
        action='ignore',
        category=DeprecationWarning,
        message=
        "Using the 'grad_clip_norm' field in Trainer is deprecated. Please usethe GradientClipping Algorithm in composer.algorithms.gradient_clipping."
    )

    test_cfg = gpt_tiny_cfg(conf_path='yamls/mosaic_gpt/125m.yaml')
    test_cfg.eval_subset_num_batches = 2
    if logit_scale:
        test_cfg.model.logit_scale = logit_scale

    if device == 'cpu':
        pytest.xfail(
            'FSDP in PyTorch 1.13 does not support precision `Precision.FP32` with sharding_strategy `FULL_SHARD.`'
        )
        test_cfg.model.init_device = 'cpu'
        test_cfg.model.attn_impl = 'torch'
        test_cfg.precision = 'fp32'

    main(test_cfg)
