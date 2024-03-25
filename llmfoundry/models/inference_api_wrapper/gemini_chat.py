# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import google.generativeai as google_genai

from typing import Dict, List, Optional, Union
# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Implements a OpenAI chat and causal LM inference API wrappers."""

import logging
import os
import random
from time import sleep
from typing import  Any, Dict , Optional

from composer.core.types import Batch
from openai.types.chat.chat_completion import ChatCompletion

from transformers import AutoTokenizer

log = logging.getLogger(__name__)

from llmfoundry.models.inference_api_wrapper.interface import \
    InferenceAPIEvalWrapper

from transformers import AutoTokenizer
MAX_RETRIES = 3

__all__ = [
    'GeminiChatAPIEvalrapper',
]

log = logging.getLogger(__name__)



class GeminiChatAPIEvalrapper(InferenceAPIEvalWrapper):
    """Databricks Foundational Model API wrapper for causal LM models."""

    def __init__(self, model_cfg: Dict, tokenizer: AutoTokenizer) -> None:
        api_key = model_cfg.pop('api_key', None)
        if api_key is None:
            api_key = os.environ.get('GEMINI_API_KEY')
        google_genai.configure(api_key=api_key)
        super().__init__(model_cfg, tokenizer)
        self.model_cfg = model_cfg
        self.model = google_genai.GenerativeModel(model_cfg.get('version', ''))
        ignore = [
            google_genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            google_genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            google_genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            google_genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        ]
        self.safety_settings = {
            category: google_genai.types.HarmBlockThreshold.BLOCK_NONE
            for category in ignore
        }

    def generate_completion(
            self,
            prompt: str, #
            num_tokens: int,
            generation_kwargs: Optional[dict] = None) -> ChatCompletion:
        if generation_kwargs is None:
            generation_kwargs = {}
        if isinstance(prompt, str):
            generation_config=google_genai.types.GenerationConfig(
                candidate_count=1,
                max_output_tokens=num_tokens,
                temperature = generation_kwargs.get('temperature', 0)
            )
            response = self.model.generate_content(prompt,
                    safety_settings=self.safety_settings,
                    generation_config=generation_config)
            return response
        else:
            raise ValueError(f"Prompt must be str: {prompt}")

    def completion_to_string(self, completion: ChatCompletion):
        return [completion.text]

    def eval_forward(self, batch: Batch, outputs: Optional[Any] = None):
        # Override the base class because Chat's API always strips spacing from model outputs resulting in different tokens
        # than what the continuation would expect.
        # Get around this issue by retokenizing the batch to remove spacing from the continuation as well as
        # decoding the whole continuation at once.
        padding_tok = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else self.tokenizer.eos_token_id
        if batch.get('mode', '') == 'generate':
            outputs = []
            # generate-based implementation
            for tokens, _ in zip(batch['input_ids'], batch['labels']):

                tokens = tokens.tolist()
                tokens = [t for t in tokens if t != padding_tok]
                prompt = self.tokenizer.decode(tokens)

                if 'generation_length' in batch:
                    num_tokens = batch['generation_length']
                elif 'generation_kwargs' in batch:
                    num_tokens = batch['generation_kwargs'].get(
                        'max_new_tokens', 2)

                for _ in range(
                        0,
                        batch.get('generation_kwargs',
                                  {}).get('num_return_sequences', 1)):
                    api_output = self.try_generate_completion(  #
                        prompt,
                        num_tokens=num_tokens,
                        generation_kwargs=batch.get('generation_kwargs', {}))

                    assert api_output is not None
                    sample_output = self.completion_to_string(
                        api_output)[  # pyright: ignore
                            0]
                    outputs.append(sample_output)
            return outputs
        else:
            raise ValueError("Only 'generate' tasks are supported.")

    def try_generate_completion(
        self,
        prompt: Union[str, List],
        num_tokens: int,
        generation_kwargs: Optional[dict] = None
    ):
        if generation_kwargs is None:
            generation_kwargs = {}
       
        tries = 0
        completion = None
        delay = 1
        while tries < MAX_RETRIES:
            tries += 1
            try:
                completion = self.generate_completion(prompt, num_tokens,
                                                      generation_kwargs)                                     
                break
            except Exception as e:
                breakpoint()
                delay *= 2 * (1 + random.random())
                sleep(delay)
                continue
                

        return completion

