# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Deprecated Generate callback.

Please use composer.callbacks.Generate instead.
"""
import warnings
from typing import Any, List, Union

from composer.callbacks import Generate as ComposerGenerate
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from llmfoundry.utils.warnings import VersionedDeprecationWarning

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


class Generate(ComposerGenerate):

    def __init__(self, prompts: List[str], batch_log_interval: int,
                 **kwargs: Any):

        warnings.warn(
            VersionedDeprecationWarning('Accessing llmfoundry.callbacks.generate_callback.Generate ' + \
             'is deprecated. Please use composer.callbacks.Generate instead.',
             remove_version='0.5.0',
            )
        )

        interval = f'{batch_log_interval}ba'
        super().__init__(prompts=prompts, interval=interval, **kwargs)
