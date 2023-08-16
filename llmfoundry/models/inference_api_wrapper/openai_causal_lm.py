# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Implements a Hugging Causal LM wrapped inside a :class:`.ComposerModel`."""

import os
from time import sleep
from typing import Any, Dict, List, Optional, Union

import openai
import tiktoken
import torch
from composer.core.types import Batch
from openai.error import RateLimitError
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from llmfoundry.models.inference_api_wrapper.interface import \
    InferenceAPIEvalWrapper

__all__ = [
    'OpenAICausalLMEvalWrapper', 'OpenAIChatAPIEvalWrapper',
    'OpenAITokenizerWrapper'
]

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
openai.api_key = os.getenv('OPENAI_API_KEY')

MAX_RETRIES = 100


class OpenAITokenizerWrapper:

    def __init__(self, name: str) -> None:
        self.tokenizer = tiktoken.encoding_for_model(name)

    def __call__(self, x: str, add_special_tokens: bool = False):
        return self.encode(x)

    def encode(self,
               x: Union[str, List[str]],
               add_special_tokens: bool = False):
        if isinstance(x, str):
            return {
                'input_ids':
                    self.tokenizer.encode(x, allowed_special={'<|endoftext|>'})
            }
        elif isinstance(x, list):
            return {
                'input_ids':
                    self.tokenizer.encode_batch(
                        x, allowed_special={'<|endoftext|>'})
            }
        else:
            raise ValueError(
                f'`encode` argument must be str or List[str], got: {type(x)}')

    def decode(self, x: List[int]):
        return self.tokenizer.decode(x)

    @property
    def pad_token_id(self):
        return self.tokenizer.eot_token

    @property
    def eos_token_id(self):
        return self.tokenizer.eot_token

    def construct_logit_tensor(self, logprobs: Dict[str, float]):
        tensor = torch.tensor([min(logprobs.values()) - 1] *
                              (self.pad_token_id + 1))
        for k in logprobs:
            encoding = self.encode(k)['input_ids']
            idx = encoding[0]
            tensor[idx] = logprobs[k]
        return tensor


class OpenAIChatAPIEvalWrapper(InferenceAPIEvalWrapper):

    def retokenize(self, tokens: List[int], cont_idxs: List[int]):
        original_len = len(tokens)
        retokenized_continuation = self.tokenizer.encode(
            self.tokenizer.decode(tokens[cont_idxs[0]:cont_idxs[-1] +
                                         1]).strip())['input_ids']
        tokens = tokens[:cont_idxs[0]] + retokenized_continuation + [
            tokens[-1]
        ] * (len(tokens) -
             len(tokens[:cont_idxs[0]] + retokenized_continuation))
        if len(tokens) > original_len:
            # this only happens if we were already at max seq len and the continuation got LARGER
            tokens = tokens[-original_len:]
            cont_idxs = list(
                range(original_len - len(retokenized_continuation),
                      original_len))
        else:
            cont_idxs = list(
                range(cont_idxs[0],
                      cont_idxs[0] + len(retokenized_continuation)))
        return torch.tensor(tokens), torch.tensor(cont_idxs)

    def rebatch(self, batch: Batch):
        new_batch: Dict[str, Union[List[torch.Tensor], torch.Tensor]] = {
            'input_ids': [],
            'continuation_indices': [],
            'labels': []
        }
        for tokens, cont_idxs in zip(batch['input_ids'],
                                     batch['continuation_indices']):
            tokens, cont_idxs = self.retokenize(tokens.tolist(),
                                                cont_idxs.tolist())

            assert isinstance(new_batch['input_ids'], list)
            new_batch['input_ids'].append(tokens)
            assert isinstance(new_batch['labels'], list)
            new_batch['labels'].append(tokens)
            assert isinstance(new_batch['continuation_indices'], list)
            new_batch['continuation_indices'].append(cont_idxs)

        new_batch.update({
            k: torch.stack(new_batch[k])  # pyright: ignore
            for k in ['input_ids', 'labels']
        })

        new_batch.update({k: v for k, v in batch.items() if k not in new_batch})

        return new_batch

    def eval_forward(self, batch: Batch, outputs: Optional[Any] = None):
        # If the batch mode is generate, we will generate a requested number of tokens using the underlying
        # model's generate function. Extra generation kwargs can be passed in via the batch. Strings will
        # be returned from eval_forward
        output_logits_batch = []
        batch = self.rebatch(batch)
        for tokens, cont_idxs in zip(batch['input_ids'],
                                     batch['continuation_indices']):

            seqlen = tokens.shape[0]
            tokens = tokens.tolist()
            cont_idxs = cont_idxs.tolist()
            expected_cont_tokens = tokens[cont_idxs[0]:cont_idxs[-1] + 1]
            output_logits = torch.nn.functional.one_hot(
                torch.tensor(tokens[1:cont_idxs[0]]),
                num_classes=self.tokenizer.pad_token_id + 1)

            prompt = self.tokenizer.decode(tokens[:cont_idxs[0]])
            next_logit_tensor = self.get_next_token_logit_tensor(
                prompt, num_tokens=len(expected_cont_tokens))

            if next_logit_tensor is not None:
                output_logits = torch.cat([output_logits, next_logit_tensor])
            padding = torch.nn.functional.one_hot(
                torch.full((seqlen - output_logits.shape[0],),
                           self.tokenizer.pad_token_id),
                num_classes=self.tokenizer.pad_token_id + 1)
            output_logits = torch.cat([output_logits, padding])
            output_logits_batch.append(output_logits)

        return torch.stack(output_logits_batch).to(batch['input_ids'].device)

    def get_next_token_logit_tensor(self, prompt: str, num_tokens: int = 1):
        tries = 0
        chat_completion = None
        while tries < MAX_RETRIES:
            tries += 1
            try:
                chat_completion = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=[{
                        'role': 'user',
                        'content': prompt
                    }],
                    max_tokens=num_tokens,
                    temperature=0.0)
                break
            except RateLimitError as e:
                if 'You exceeded your current quota' in str(e._message):
                    raise e
                sleep(60)
                continue
            except Exception as e:
                print(f'Caught exeception {str(e)}, continuing')
                continue

        if chat_completion is None:
            raise ValueError("Couldn't generate model output")

        assert isinstance(chat_completion, dict)
        if len(chat_completion['choices']) > 0:
            tensors = []
            for t in self.tokenizer.encode(chat_completion['choices'][0]
                                           ['message']['content'])['input_ids']:
                tensors.append(
                    self.tokenizer.construct_logit_tensor(
                        {self.tokenizer.decode([t]): 0.0}))

            if len(tensors) == 0:
                return None
            return torch.stack(tensors)
        else:
            # the model sometimes stops early even though we are still requesting tokens!
            # not sure if there's a fix
            return None


class OpenAICausalLMEvalWrapper(InferenceAPIEvalWrapper):

    def get_next_token_logit_tensor(self, prompt: str):
        tries = 0
        completion = None
        while tries < MAX_RETRIES:
            tries += 1
            try:
                completion = openai.Completion.create(engine=self.model_name,
                                                      prompt=prompt,
                                                      max_tokens=1,
                                                      logprobs=5,
                                                      temperature=0.0)
                break
            except RateLimitError as e:
                if 'You exceeded your current quota' in str(e._message):
                    raise e
                sleep(60)
                continue
            except Exception:
                continue

        if completion is None:
            raise ValueError("Couldn't generate model output")

        assert isinstance(completion, dict)
        if len(completion['choices'][0]['logprobs']['top_logprobs']) > 0:
            tensor = self.tokenizer.construct_logit_tensor(
                dict(completion['choices'][0]['logprobs']['top_logprobs'][0]))
            return tensor
        else:
            # the model sometimes stops early even though we are still requesting tokens!
            # not sure if there's a fix
            return None
