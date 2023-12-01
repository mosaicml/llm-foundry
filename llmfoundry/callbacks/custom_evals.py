# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Aggregate ICL evals into composite scores."""

import logging
import copy
import sys

from typing import Optional, Any

from composer.core import Callback, State, get_precision_context
from composer.loggers import Logger
from composer.models import HuggingFaceModel
from composer.utils import dist
from transformers import PreTrainedTokenizerBase

__all__ = ['InstructionFollowingEval']

log = logging.getLogger(__name__)

from contextlib import contextmanager

@contextmanager
def generate_mode(model: HuggingFaceModel):
    original_mode = model.training
    model.eval()

    tokenizer: PreTrainedTokenizerBase = model.tokenizer
    # Stash the original value of padding_side because generation requires left padding
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # dummy forward call needed for FSDP to work consistently
    model.dummy_forward_called = False

    yield

    tokenizer.padding_side = original_padding_side
    model.train(mode=original_mode)


class GenerateEval(Callback):

    def generate(self, state: State, prompts: list[str], batch_size: int = 1, generate_kws: Optional[dict[str, Any]] = None) -> list[str]:
        self.generate_kwargs = generate_kws or {}
        model = state.model.module if state.is_model_ddp else state.model
        tokenizer = model.tokenizer

        with generate_mode(model):
            tokenized_input = tokenizer(prompts, return_tensors='pt', padding=True)
            all_input_ids = tokenized_input['input_ids']
            all_attn_masks = tokenized_input['attention_mask']

            device = state.device

            output_token_ids = []
            # dummy forward call needed for FSDP to work consistently
            model.dummy_forward_called = False

            n_prompts = len(prompts)
            for start in range(0, n_prompts, batch_size):
                end = min(start +batch_size, n_prompts)
                input_ids = all_input_ids[start:end]
                attn_mask = all_attn_masks[start:end]

                # Move batch to device.
                input_ids = device.tensor_to_device(input_ids)
                attn_mask = device.tensor_to_device(attn_mask)
                with get_precision_context(state.precision):
                    output_token_ids.extend(
                        model.generate(  # type: ignore
                            input_ids=input_ids,
                            attention_mask=attn_mask,
                            synced_gpus=dist.get_world_size() > 1,
                            **self.generate_kwargs,
                        ))

            # Process prompts and outputs into a table.
            outputs = []
            input_tokens_len = all_input_ids.shape[1]
            for i, prompt in enumerate(prompts):
                output_tokens = output_token_ids[i][input_tokens_len:]
                output_text = tokenizer.decode(output_tokens, skip_special_tokens=True)
                outputs.append(output_text)
            return outputs


class InstructionFollowingEval(GenerateEval):


    def __init__(self, log_categories: bool=True):
        try:
            from instruction_following_eval import instruction_following_eval, default_examples
        except ImportError:
            print((
                "The instruction_following_eval repo must be installed to run this callbak\n"
                "Please install it with:\n"
                "  pip install git+https://github.com/josejg/instruction_following_eval.git"
            ), file=sys.stderr)
            sys.exit(1)

        self.log_categories = log_categories
        self.eval_fn = instruction_following_eval
        self.if_examples = default_examples()

        # TODO: grab from config
        self.batch_size = 48

    def fit_start(self, state: State, logger: Logger):
        self.run_eval(state, logger)

    def eval_after_all(self, state: State, logger: Logger):
        self.run_eval(state, logger)

    def run_eval(self, state: State, logger: Logger) -> dict[str, float]:

        examples = copy.deepcopy(self.if_examples)
        prompts = [example['prompt'] for example in examples]

        responses = self.generate(state, prompts, batch_size=self.batch_size)

        for example, response in zip(examples, responses):
            example['response'] = response

        acc_dict = self.eval_fn(examples)
        acc_dict = {
            f'metrics/instruction_following/0-shot/{category}': acc
            for category, acc in acc_dict.items()
        }

        if self.log_categories:
            logger.log_metrics(acc_dict)
        else:
            avg = 'metrics/instruction_following/0-shot/average'
            logger.log_metrics({avg: acc_dict[avg]})

        from pprint import pprint
        pprint(acc_dict)
        return acc_dict
