# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import transformers
from omegaconf import OmegaConf

from llmfoundry.models.hf.hf_t5 import ComposerHFT5
from llmfoundry.utils.warnings import ExperimentalWarning


def test_experimental_hf_t5(tiny_t5_tokenizer):
    cfg = OmegaConf.create({
        'pretrained_model_name_or_path': 't5-base',
        'config_overrides': {
            'num_layers': 2,
            'num_decoder_layers': 2,
        },
        'pretrained': False,
        'init_device': 'cpu',
    })

    with pytest.warns(ExperimentalWarning):
        _ = ComposerHFT5(**cfg, tokenizer=tiny_t5_tokenizer)
