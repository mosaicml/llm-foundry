# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import warnings

# bitsandbytes is a very noisy library. A lot of it is print statements that we can't easily suppress,
# but we can at least suppress a bunch of spurious warnings.
warnings.filterwarnings('ignore', category=UserWarning, module='bitsandbytes')

import logging

try:
    from flash_attn import flash_attn_func
    del flash_attn_func
except ImportError as e:
    if 'undefined symbol' in str(e):
        raise ImportError(
            'The flash_attn package is not installed correctly. Usually this means that your runtime version.'
            +
            ' of PyTorch is different from the version that flash_attn was installed with, which can occur when your'
            +
            ' workflow has resulted in PyTorch being reinstalled. This probably happened because you are using an old docker image'
            +
            ' with the latest version of LLM Foundry. Check that the PyTorch version in your Docker image matches the PyTorch version'
            +
            ' in LLM Foundry setup.py and update accordingly. The latest Docker image can be found in the README.',
        ) from e

from llmfoundry.utils.logging_utils import SpecificWarningFilter

# Filter out Hugging Face warning for not using a pinned revision of the model
logger = logging.getLogger('transformers.dynamic_module_utils')
new_files_warning_filter = SpecificWarningFilter(
    'A new version of the following files was downloaded from',
)

logger.addFilter(new_files_warning_filter)

from llmfoundry import (
    algorithms,
    callbacks,
    cli,
    data,
    eval,
    interfaces,
    loggers,
    metrics,
    models,
    optim,
    tokenizers,
    utils,
)
from llmfoundry.data import StreamingFinetuningDataset, StreamingTextDataset
from llmfoundry.eval import InContextLearningDataset, InContextLearningMetric
from llmfoundry.models.hf import ComposerHFCausalLM
from llmfoundry.models.mpt import (
    ComposerMPTCausalLM,
    MPTConfig,
    MPTForCausalLM,
    MPTModel,
    MPTPreTrainedModel,
)
from llmfoundry.optim import DecoupledLionW

__all__ = [
    'StreamingFinetuningDataset',
    'StreamingTextDataset',
    'InContextLearningDataset',
    'InContextLearningMetric',
    'ComposerHFCausalLM',
    'MPTConfig',
    'MPTPreTrainedModel',
    'MPTModel',
    'MPTForCausalLM',
    'ComposerMPTCausalLM',
    'DecoupledLionW',
    'algorithms',
    'callbacks',
    'cli',
    'data',
    'eval',
    'interfaces',
    'loggers',
    'metrics',
    'models',
    'optim',
    'tokenizers',
    'utils',
]

__version__ = '0.11.0.dev0'
