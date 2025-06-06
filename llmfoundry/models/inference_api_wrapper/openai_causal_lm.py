# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
"""Implements a OpenAI chat and causal LM inference API wrappers."""

import logging
import os
import random
from time import sleep
from typing import TYPE_CHECKING, Any, Optional, Union, cast

import torch
from composer.core.types import Batch
from composer.utils.import_helpers import MissingConditionalImportError
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase

log = logging.getLogger(__name__)

from llmfoundry.models.inference_api_wrapper.interface import \
    InferenceAPIEvalWrapper

__all__ = [
    'OpenAICausalLMEvalWrapper',
    'OpenAIChatAPIEvalWrapper',
    'OpenAIEvalInterface',
]

if TYPE_CHECKING:
    from openai.types.chat.chat_completion import ChatCompletion
    from openai.types.completion import Completion
    from openai.types.completion_choice import Logprobs

MAX_RETRIES = 10


class OpenAIEvalInterface(InferenceAPIEvalWrapper):

    def __init__(
        self,
        om_model_config: DictConfig,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        super().__init__(om_model_config, tokenizer)
        try:
            import openai
        except ImportError as e:
            raise MissingConditionalImportError(
                extra_deps_group='openai',
                conda_package='openai',
                conda_channel='conda-forge',
            ) from e

        api_key = os.environ.get('OPENAI_API_KEY')
        base_url = om_model_config.get('base_url')
        if base_url is None:
            # Using OpenAI default, where the API key is required
            if api_key is None:
                raise ValueError(
                    'No OpenAI API Key found. Ensure it is saved as an environmental variable called OPENAI_API_KEY.',
                )

        else:
            # Using a custom base URL, where the API key may not be required
            custom_url_tag = '' if api_key is not None else ' (no API key set)'
            log.info(
                f'Making request to custom base URL: {base_url}{custom_url_tag}',
            )
            api_key = 'placeholder'  # This cannot be None

        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
        if 'version' in om_model_config:
            self.model_name = om_model_config['version']
        else:
            self.model_name = om_model_config['name']

    def generate_completion(self, prompt: str, num_tokens: int):
        raise NotImplementedError()

    def process_result(self, completion):  # pyright: ignore
        raise NotImplementedError()

    def get_next_token_logit_tensor(self, prompt: str, num_tokens: int = 1):
        completion = self.try_generate_completion(prompt, num_tokens)
        return self.process_result(completion)

    def try_generate_completion(self, prompt: str, num_tokens: int):
        try:
            from openai import APITimeoutError, RateLimitError
        except ImportError as e:
            raise MissingConditionalImportError(
                extra_deps_group='openai',
                conda_package='openai',
                conda_channel='conda-forge',
            ) from e
        tries = 0
        completion = None
        delay = 1
        while tries < MAX_RETRIES:
            tries += 1
            try:
                completion = self.generate_completion(prompt, num_tokens)
                break
            except RateLimitError as e:
                if 'You exceeded your current quota' in str(
                    e.message,
                ):  # pyright: ignore
                    raise e
                delay *= 2 * (1 + random.random())
                sleep(delay)
                continue
            except APITimeoutError as e:
                delay *= 2 * (1 + random.random())
                sleep(delay)
                continue

        return completion


class OpenAIChatAPIEvalWrapper(OpenAIEvalInterface):

    def __init__(
        self,
        om_model_config: DictConfig,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        super().__init__(om_model_config, tokenizer)

        self.generate_completion = lambda prompt, num_tokens: self.client.chat.completions.create(
            model=self.model_name,
            messages=[{
                'role':
                    'system',
                'content':
                    om_model_config.get(
                        'system_role_prompt',
                        'Please complete the following text: ',
                    ),
            }, {
                'role': 'user',
                'content': prompt,
            }],
            max_tokens=num_tokens,
            temperature=0.0,
        )

    def retokenize(self, tokens: list[int], cont_idxs: list[int]):
        """Chat API will never respond with a word-initial space.

        If the continuation tokens begin with a word initial space, we need to
        re-tokenize with the space removed.
        """
        original_len = len(tokens)
        retokenized_continuation = cast(
            Any,
            self.tokenizer(
                self.tokenizer.decode(
                    tokens[cont_idxs[0]:cont_idxs[-1] + 1],
                ).strip(),
            )['input_ids'],
        )

        # replace the original continuation with the retokenized continuation + padding
        padding = [tokens[-1]] * (
            len(tokens) - len(tokens[:cont_idxs[0]] + retokenized_continuation)
        )
        tokens = tokens[:cont_idxs[0]] + retokenized_continuation + padding

        if len(tokens) > original_len:
            # this only happens if we were already at max seq len and the continuation got LARGER
            tokens = tokens[-original_len:]
            cont_idxs = list(
                range(
                    original_len - len(retokenized_continuation),
                    original_len,
                ),
            )
        else:
            cont_idxs = list(
                range(
                    cont_idxs[0],
                    cont_idxs[0] + len(retokenized_continuation),
                ),
            )
        return torch.tensor(tokens), torch.tensor(cont_idxs)

    def rebatch(self, batch: Batch):
        """Chat API tokenization has different behavior than GPT3.

        Model responses will never begin with spaces even if the continuation is
        expected to, so we need to retokenize the input to account for that.
        """
        new_batch: dict[str, Union[list[torch.Tensor], torch.Tensor]] = {
            'input_ids': [],
            'continuation_indices': [],
            'labels': [],
        }
        for tokens, cont_idxs in zip(
            batch['input_ids'],
            batch['continuation_indices'],
        ):
            tokens, cont_idxs = self.retokenize(
                tokens.tolist(),
                cont_idxs.tolist(),
            )

            assert isinstance(new_batch['input_ids'], list)
            new_batch['input_ids'].append(tokens)
            assert isinstance(new_batch['labels'], list)
            new_batch['labels'].append(tokens)
            assert isinstance(new_batch['continuation_indices'], list)
            new_batch['continuation_indices'].append(cont_idxs)

        new_batch.update({
            k:
                torch.stack(new_batch[k])  # pyright: ignore
            for k in ['input_ids', 'labels']
        })

        new_batch.update({k: v for k, v in batch.items() if k not in new_batch})

        return new_batch

    def eval_forward(self, batch: Batch, outputs: Optional[Any] = None):
        # Override the base class because Chat's API always strips spacing from model outputs resulting in different tokens
        # than what the continuation would expect.
        # Get around this issue by retokenizing the batch to remove spacing from the continuation as well as
        # decoding the whole continuation at once.
        padding_tok = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else self.tokenizer.eos_token_id
        assert isinstance(padding_tok, int)
        output_logits_batch = []
        batch = self.rebatch(batch)
        for tokens, cont_idxs in zip(
            batch['input_ids'],
            batch['continuation_indices'],
        ):

            seqlen = tokens.shape[0]
            tokens = tokens.tolist()
            cont_idxs = cont_idxs.tolist()
            expected_cont_tokens = tokens[cont_idxs[0]:cont_idxs[-1] + 1]
            output_logits = torch.nn.functional.one_hot(
                torch.tensor(tokens[1:cont_idxs[0]]),
                num_classes=len(self.tokenizer),
            )

            prompt = self.tokenizer.decode(tokens[:cont_idxs[0]])
            next_logit_tensor = self.get_next_token_logit_tensor(
                prompt,
                num_tokens=len(expected_cont_tokens),
            )

            if next_logit_tensor is not None:
                output_logits = torch.cat([output_logits, next_logit_tensor])
            padding = torch.nn.functional.one_hot(
                torch.full((seqlen - output_logits.shape[0],), padding_tok),
                num_classes=len(self.tokenizer),
            )
            output_logits = torch.cat([output_logits, padding])
            output_logits_batch.append(output_logits)

        return torch.stack(output_logits_batch).to(batch['input_ids'].device)

    def process_result(self, completion: Optional['ChatCompletion']):
        if completion is None:
            raise ValueError("Couldn't generate model output")

        if len(completion.choices) > 0:
            tensors = []
            for t in cast(
                Any,
                self.tokenizer(
                    completion.choices[0].message.content,
                )['input_ids'],
            ):
                # Not real logprobs
                tensor = torch.tensor([0] * (len(self.tokenizer)))
                tensor[t] = 1.0
                tensors.append(tensor)

            if len(tensors) == 0:
                return None
            return torch.stack(tensors)
        else:
            # the model sometimes stops early even though we are still requesting tokens!
            # not sure if there's a fix
            return None


class OpenAICausalLMEvalWrapper(OpenAIEvalInterface):

    def __init__(
        self,
        om_model_config: DictConfig,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        super().__init__(om_model_config, tokenizer)
        self.generate_completion = lambda prompt, num_tokens: self.client.completions.create(
            model=self.model_name,
            prompt=prompt,
            max_tokens=num_tokens,
            logprobs=5,
            temperature=0.0,
        )

    def process_result(self, completion: Optional['Completion']):
        if completion is None:
            raise ValueError("Couldn't generate model output")

        if TYPE_CHECKING:
            assert isinstance(completion, Completion)
            assert isinstance(completion.choices[0].logprobs, Logprobs)
            assert isinstance(completion.choices[0].logprobs.top_logprobs, list)

        if len(completion.choices[0].logprobs.top_logprobs[0]) > 0:
            # Construct tensor of shape (vocab_size,) with logprobs for each token
            tokenizer_logprobs = dict(
                completion.choices[0].logprobs.top_logprobs[0],
            )
            tensor = torch.tensor([min(tokenizer_logprobs.values()) - 1] *
                                  (len(self.tokenizer)))
            for k in tokenizer_logprobs:
                encoding = cast(Any, self.tokenizer(k)['input_ids'])
                tensor[encoding[0]] = tokenizer_logprobs[k]
            return tensor
        else:
            # the model sometimes stops early even though we are still requesting tokens!
            # not sure if there's a fix
            return None
