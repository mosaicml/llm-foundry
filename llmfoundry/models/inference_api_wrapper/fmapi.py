# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import time
from typing import Dict

import requests
from transformers import AutoTokenizer

from llmfoundry.models.inference_api_wrapper.openai_causal_lm import (
    OpenAICausalLMEvalWrapper, OpenAIChatAPIEvalWrapper, OpenAIEvalInterface)

__all__ = [
    'FMAPICasualLMEvalWrapper',
    'FMAPIChatAPIEvalWrapper',
]

log = logging.getLogger(__name__)


def block_until_ready(base_url: str):
    """Block until the endpoint is ready."""
    sleep_s = 5
    remaining_s = 5 * 50  # At max, wait 5 minutes

    ping_url = f'{base_url}/ping'

    while True:
        try:
            requests.get(ping_url)
            break
        except requests.exceptions.ConnectionError:
            log.debug(
                f'Endpoint {ping_url} not ready yet. Sleeping {sleep_s} seconds'
            )
            time.sleep(sleep_s)
            remaining_s -= sleep_s
        else:
            log.info(f'Endpoint {ping_url} is ready')
            break

        if remaining_s <= 0:
            raise TimeoutError(
                f'Endpoint {ping_url} never became ready, exiting')


class FMAPIEvalInterface(OpenAIEvalInterface):

    def __init__(self, model_cfg: Dict, tokenizer: AutoTokenizer):
        is_local = model_cfg.pop('local', False)
        if is_local:
            base_url = os.environ.get('MOSAICML_MODEL_ENDPOINT',
                                      'http://0.0.0.0:8080/v2')
            model_cfg['base_url'] = base_url
            block_until_ready(base_url)

        if 'base_url' not in model_cfg:
            raise ValueError(
                'Must specify base_url in model_cfg for FMAPIsEvalWrapper')

        super().__init__(model_cfg, tokenizer)


class FMAPICasualLMEvalWrapper(FMAPIEvalInterface, OpenAICausalLMEvalWrapper):
    """Databricks Foundational Model API wrapper for causal LM models."""


class FMAPIChatAPIEvalWrapper(FMAPIEvalInterface, OpenAIChatAPIEvalWrapper):
    """Databricks Foundational Model API wrapper for chat models."""
