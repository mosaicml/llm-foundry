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

    def __init__(self, verbose: bool = False):
        self._verbose = verbose

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

            if self._verbose:
                from rich.console import Console
                c = Console()
                for i, (prompt, output) in enumerate(zip(prompts, outputs)):
                    c.print(f"# {i}/{len(prompts)}")
                    c.print(f'Prompt: {prompt}', style='green')
                    c.print(f'Reply:  {output}', style='yellow')

            return outputs

    def fit_start(self, state: State, logger: Logger):
        self.run_eval(state, logger)

    def epoch_end(self, state: State, logger: Logger):
        self.run_eval(state, logger)

    def run_eval(self, state: State, logger: Logger) -> dict[str, float]:
        raise NotImplementedError("Base Class")


class JSONExtractionEval(GenerateEval):

    def __init__(self, data_path: str, verbose: bool = False):
        super().__init__(verbose=verbose)
        from tunes.data.extraction import json_f1_score
        import datasets as hf_datasets

        self.batch_size = 1
 
        # dset = hf_datasets.load_from_disk(data_path)['test']
        from rclone_python import rclone
        rclone.copy(data_path, '/data/test')
        dset = hf_datasets.load_from_disk('/data/test')['test']

        self.prompts = dset['prompt']
        self.responses = dset['response']

    def run_eval(self, state: State, logger: Logger) -> dict[str, float]:
        responses = self.generate(state, self.prompts, batch_size=self.batch_size)
        f1_scores = [json_f1_score(gt, gen) for gt, gen in zip(self.responses, responses)]
        mean_f1_score = sum(f1_scores) / len(f1_scores)
        metrics = {
                'metrics/extraction/0-shot/MeanF1Score': mean_f1_score,
        }
        logger.log_metrics(metrics)
        return metrics


class InstructionFollowingEval(GenerateEval):


    def __init__(self, verbose: bool = False, log_categories: bool=True):
        super().__init__(verbose=verbose)
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


class MultiPLCodeEval(GenerateEval):

    # IMPORTANT: Temperature is a hyperparameter
    # https://github.com/mcarbin/composer/blob/9fe12627e178d34f1f599f383f9634fd24c4e99b/composer/datasets/in_context_learning_evaluation.py#L1062
