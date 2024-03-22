# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import time
from typing import Dict, Optional

import requests
from transformers import AutoTokenizer

from llmfoundry.models.inference_api_wrapper.openai_causal_lm import (
    OpenAICausalLMEvalWrapper, OpenAIChatAPIEvalWrapper, OpenAIEvalInterface)

__all__ = [
    'FMAPICasualLMEvalWrapper',
    'FMAPIChatAPIEvalWrapper',
]

log = logging.getLogger(__name__)


class FMAPIEvalInterface(OpenAIEvalInterface):

    def block_until_ready(self, base_url: str):
        """Block until the endpoint is ready."""
        sleep_s = 5
        timout_s = 5 * 60  # At max, wait 5 minutes

        ping_url = f'{base_url}/ping'

        waited_s = 0
        while True:
            try:
                requests.get(ping_url)
                log.info(f'Endpoint {ping_url} is ready')
                break
            except requests.exceptions.ConnectionError:
                log.debug(
                    f'Endpoint {ping_url} not ready yet. Sleeping {sleep_s} seconds'
                )
                time.sleep(sleep_s)
                waited_s += sleep_s

            if waited_s >= timout_s:
                raise TimeoutError(
                    f'Endpoint {ping_url} did not become read after {waited_s:,} seconds, exiting'
                )

    def __init__(self, model_cfg: Dict, tokenizer: AutoTokenizer):
        is_local = model_cfg.pop('local', False)
        api_key = model_cfg.pop('api_key', None)
        if is_local:
            base_url = os.environ.get('MOSAICML_MODEL_ENDPOINT',
                                      'http://0.0.0.0:8080/v2')
            model_cfg['base_url'] = base_url
            self.block_until_ready(base_url)

        if 'base_url' not in model_cfg:
            raise ValueError(
                'Must specify base_url or use local=True in model_cfg for FMAPIsEvalWrapper'
            )

        super().__init__(model_cfg, tokenizer, api_key)


class FMAPICasualLMEvalWrapper(FMAPIEvalInterface, OpenAICausalLMEvalWrapper):
    """Databricks Foundational Model API wrapper for causal LM models."""


class FMAPIChatAPIEvalWrapper(FMAPIEvalInterface, OpenAIChatAPIEvalWrapper):
    """Databricks Foundational Model API wrapper for chat models."""
