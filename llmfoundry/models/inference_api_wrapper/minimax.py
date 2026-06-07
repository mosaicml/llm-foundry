# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
"""Implements MiniMax chat and causal LM inference API wrappers.

MiniMax provides an OpenAI-compatible API at https://api.minimax.io/v1,
supporting models such as MiniMax-M3 (default), MiniMax-M2.7, and
MiniMax-M2.7-highspeed.
"""

import logging
import os

from composer.utils.import_helpers import MissingConditionalImportError
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase

from llmfoundry.models.inference_api_wrapper.openai_causal_lm import (
    OpenAIChatAPIEvalWrapper,
    OpenAIEvalInterface,
)

log = logging.getLogger(__name__)

__all__ = [
    'MiniMaxChatAPIEvalWrapper',
    'MiniMaxEvalInterface',
]

MINIMAX_API_BASE_URL = 'https://api.minimax.io/v1'


class MiniMaxEvalInterface(OpenAIEvalInterface):
    """MiniMax evaluation interface using the OpenAI-compatible API.

    Automatically configures the base URL and API key for MiniMax.
    The API key is read from the ``MINIMAX_API_KEY`` environment variable,
    falling back to ``OPENAI_API_KEY`` if set.
    """

    def __init__(
        self,
        om_model_config: DictConfig,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        try:
            import openai
        except ImportError as e:
            raise MissingConditionalImportError(
                extra_deps_group='openai',
                conda_package='openai',
                conda_channel='conda-forge',
            ) from e

        # Resolve the API key: prefer MINIMAX_API_KEY, fall back to
        # OPENAI_API_KEY
        api_key = os.environ.get('MINIMAX_API_KEY')
        if api_key is None:
            api_key = os.environ.get('OPENAI_API_KEY')
        if api_key is None:
            raise ValueError(
                'No MiniMax API key found. Set the MINIMAX_API_KEY '
                'environment variable.',
            )

        # Set base_url to MiniMax default if not already specified
        base_url = om_model_config.get('base_url', MINIMAX_API_BASE_URL)

        # Initialize the InferenceAPIEvalWrapper grandparent directly,
        # bypassing OpenAIEvalInterface which would overwrite the API key
        # with 'placeholder' for custom base URLs.
        from llmfoundry.models.inference_api_wrapper.interface import \
            InferenceAPIEvalWrapper
        InferenceAPIEvalWrapper.__init__(self, om_model_config, tokenizer)

        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)

        if 'version' in om_model_config:
            self.model_name = om_model_config['version']
        else:
            self.model_name = om_model_config['name']


class MiniMaxChatAPIEvalWrapper(MiniMaxEvalInterface, OpenAIChatAPIEvalWrapper):
    """MiniMax chat API wrapper for evaluating chat models.

    Uses the OpenAI-compatible ``/v1/chat/completions`` endpoint.
    Configure the model version via the ``version`` field in the model
    config (e.g. ``MiniMax-M3``, ``MiniMax-M2.7`` or
    ``MiniMax-M2.7-highspeed``).
    """
