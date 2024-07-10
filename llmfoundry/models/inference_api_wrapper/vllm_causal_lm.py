# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Implements a VLLM based chat and causal LM inference API wrappers."""

import logging
from time import sleep
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import torch
from composer.core.types import Batch
from composer.utils.import_helpers import MissingConditionalImportError
from transformers import AutoTokenizer
import openai


log = logging.getLogger(__name__)

from llmfoundry.models.inference_api_wrapper.interface import \
    InferenceAPIEvalWrapper

__all__ = [
    'VLLMCausalLMEvalWrapper',
]


class VLLMEvalInterface(InferenceAPIEvalWrapper):

    def __init__(self, model_cfg: Dict, tokenizer: AutoTokenizer) -> None:
        super().__init__(model_cfg, tokenizer)
        self.base_url = model_cfg.get('base_url')
        self.model_name = model_cfg.get('model_name')
        self.client = openai.OpenAI(base_url=self.base_url, api_key="NONE")
        self.model_cfg = model_cfg


class VLLMCausalLMEvalWrapper(VLLMEvalInterface):

    def eval_forward(self, batch: Batch, outputs: Optional[Any] = None):
        padding_tok = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else self.tokenizer.eos_token_id

        # If the batch mode is generate, we will generate a requested number of tokens using the underlying
        # model's generate function. Extra generation kwargs can be passed in via the batch. Strings will
        # be returned from eval_forward

        if batch.get('mode', '') == 'generate':
            if 'generation_length' in batch:
                num_tokens = batch['generation_length']
            elif 'generation_kwargs' in batch:
                num_tokens = batch['generation_kwargs'].get(
                    'max_new_tokens', 2)
            num_sequences = batch.get('generation_kwargs', {}).get('num_return_sequences', 1)

            prompts = []
            for tokens, _ in zip(batch['input_ids'], batch['labels']):
                tokens = tokens.tolist()
                tokens = [t for t in tokens if t != padding_tok]
                prompts.append(tokens)

            results = self.client.completions.create(
                model=self.model_name,
                prompt=prompts,
                max_tokens=num_tokens,
                temperature=0.0)

            outputs = []
            for result in results.choices:
                outputs.append(result.text)
            return outputs

        else:
            assert 'continuation_indices' in batch
            output_logits_batch = []
            prompts = []
            max_length = 0
            vocab_size = len(self.tokenizer)
            for tokens, cont_idxs in zip(batch['input_ids'],
                                         batch['continuation_indices']):
                prompts.append(tokens[0:cont_idxs[-1] + 1].tolist())
                max_length = max(max_length, cont_idxs[-1])

            results = self.client.completions.create(
                model=self.model_name,
                prompt=prompts,
                max_tokens=1,
                temperature=0.0,
                echo=True,
                logprobs=5)

            for tokens, cont_idxs, result in zip(batch['input_ids'],
                                                 batch['continuation_indices'],
                                                 results.choices):
                assert len(result.logprobs.top_logprobs) == cont_idxs[-1] + 2
                logits = torch.full((max_length, vocab_size), float('-inf'), dtype=torch.float32)
                for i in cont_idxs:
                    for token, prob in result.logprobs.top_logprobs[i].items():
                        tokens = self.tokenizer.tokenize(token)
                        if len(tokens) > 1:
                            log.info(f"Got {len(tokens)} tokens but expected 1")
                        if tokens:
                            t = self.tokenizer.convert_tokens_to_ids(tokens)[-1]
                            logits[i-1, t] = prob

                output_logits_batch.append(logits)

            return torch.stack(output_logits_batch).to(batch['input_ids'].device)