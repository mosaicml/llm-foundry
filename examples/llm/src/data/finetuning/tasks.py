# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""Code for task-specific seq-to-seq data formatting.

As explained in `README.md`, you can add to this file to define/register
tokenization functions for new seq-to-seq finetuning tasks. These tokenization
functions take individual examples that contain raw text and tokenize them in a
way that yields a consistent format.

When adding a new function, make sure to decorate it with:
    `@dataset_constructor.register()`,
where you can pass one or more names of datasets that should
use the decorated tokenization function.

Tokenization functions should take 2 arguments:
1. An input example
2. The tokenizer

Tokenization functions should end with:
    `return tokenizer(text=<prompt>, text_target=<response>)`,
where `<prompt>` is a placeholder for the prompt text string that you
extracted from the input example, and '<response>' is a placeholder for
the response text string.
You do not need to handle padding, truncation, etc. That will be handled
automatically elsewhere.

Just to be clear, "prompt" represents the text you would give the model
at inference time, and "response" represents the text you are training
it to produce given the prompt.
"""

import os
from functools import partial
from typing import Any, Callable, Dict, Optional, Union

import datasets
from omegaconf import DictConfig
from streaming import StreamingDataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

__all__ = ['dataset_constructor']

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


class StreamingFinetuningDataset(StreamingDataset):
    """Finetuning dataset with flexible tokenization using StreamingDataset.

    Args:
        local (str): Local dataset directory where shards are cached by split.
        tokenizer (Tokenizer): The name of the HuggingFace tokenizer to use to
            tokenize samples.
        tokenize_function (callable): A function that takes a text sample (dict) and a tokenizer,
            and returns the tokenized sample (dict). The tokenized sample must have `input_ids`
            and `labels`, with `input_ids` providing the input sequence and `labels` providing the
            (target) output sequence, in a sequence-to-sequence task.
        remote (str, optional): Download shards from this remote path or directory. If None, this
            rank and worker's partition of the dataset must all exist locally. Defaults to ``None``.
        split (str, optional): Which dataset split to use, if any. Defaults to ``None``.
        shuffle (bool): Whether to iterate over the samples in randomized order. Defaults to ``False``.
        predownload (int, optional): Target number of samples ahead to download the shards of while
            iterating. Defaults to ``100_000``.
        keep_zip (bool, optional): Whether to keep or delete the compressed file when
            decompressing downloaded shards. If set to None, keep if remote is local. Defaults to
            ``None``.
        download_retry (int): Number of download re-attempts before giving up. Defaults to ``2``.
        download_timeout (float): Number of seconds to wait for a shard to download before raising
            an exception. Defaults to ``60``.
        validate_hash (str, optional): Optional hash or checksum algorithm to use to validate
            shards. Defaults to ``None``.
        shuffle_seed (int): Seed for Deterministic data shuffling. Defaults to ``9176``.
        num_canonical_nodes (int, optional): Canonical number of nodes for shuffling with resumption.
            If ``None``, defaults to the number of nodes of the initial run. Defaults to 128.
        batch_size (int, optional): Batch size of its DataLoader, which affects how the dataset is
            partitioned over the workers. Defaults to ``None``.
    """

    def __init__(self,
                 local: str,
                 tokenizer: Tokenizer,
                 tokenize_function: Callable[[Dict, Tokenizer], Dict],
                 remote: Optional[str] = None,
                 split: Optional[str] = None,
                 shuffle: bool = False,
                 predownload: Optional[int] = 100_000,
                 keep_zip: Optional[bool] = None,
                 download_retry: int = 2,
                 download_timeout: float = 60,
                 validate_hash: Optional[str] = None,
                 shuffle_seed: int = 9176,
                 num_canonical_nodes: Optional[int] = 128,
                 batch_size: Optional[int] = None,
                 **kwargs: Any):

        if len(kwargs) > 0:
            raise ValueError(
                f'StreamingTextDataset() got an unexpected keyword argument: {kwargs}'
            )

        if remote is None or (local == remote):
            if os.path.isdir(local):
                contents = set(os.listdir(local))
                if split not in contents:
                    raise ValueError(
                        f'local directory {local} does not contain split {split}'
                    )

        # Build Dataset
        super().__init__(local=local,
                         remote=remote,
                         split=split,
                         shuffle=shuffle,
                         predownload=predownload,
                         keep_zip=keep_zip,
                         download_retry=download_retry,
                         download_timeout=download_timeout,
                         validate_hash=validate_hash,
                         shuffle_seed=shuffle_seed,
                         num_canonical_nodes=num_canonical_nodes,
                         batch_size=batch_size)

        self.tokenizer = tokenizer
        self.tokenize_function = tokenize_function

    # How to process a sample
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = super().__getitem__(idx)
        return self.tokenize_function(sample, self.tokenizer)


class DatasetConstructor:

    def __init__(self):
        self._task_tokenization_registry: Dict[str, Callable] = {}

    def register(self, *names: str):
        """Decorator for registering tokenization functions."""

        def _register_func(name: str, func: Callable) -> None:
            if name in self._task_tokenization_registry:
                raise ValueError(
                    f'A tokenization function has already been registered with {name=}.'
                )
            self._task_tokenization_registry[name] = func
            return

        def wrapper(func: Callable) -> Callable:
            for name in names:
                _register_func(name, func)
            return func

        return wrapper

    def build(self, cfg: DictConfig, tokenizer: Tokenizer):
        dataset_name = cfg.name
        split = cfg.split
        kwargs = cfg.get('kwargs', {})

        if dataset_name not in self._task_tokenization_registry:
            raise ValueError(
                f'{dataset_name} is not a registered dataset. ' +
                f'Available datasets: {self._task_tokenization_registry.keys()}'
            )

        dataset = datasets.load_dataset(dataset_name, split=split, **kwargs)

        tokenize_function = partial(
            self._task_tokenization_registry[dataset_name], tokenizer=tokenizer)

        columns_to_remove = list(dataset[0].keys())
        dataset = dataset.map(
            tokenize_function,
            batched=False,
            remove_columns=columns_to_remove,
        )
        return dataset

    def build_from_streaming(self, dataset_name: str, tokenizer: Tokenizer,
                             **kwargs: Any):

        tokenize_function = self._task_tokenization_registry[dataset_name]

        dataset = StreamingFinetuningDataset(
            tokenizer=tokenizer,
            tokenize_function=tokenize_function,
            **kwargs,
        )

        return dataset


dataset_constructor = DatasetConstructor()


@dataset_constructor.register('tatsu-lab/alpaca')
def alpaca_tokenize_function(inp: Dict, tokenizer: Tokenizer):
    """Split out prompt/response from text and tokenize."""
    try:
        prompt, response = inp['text'].split('### Response:')
    except Exception as e:
        raise ValueError(
            f"Unable to extract prompt/response from 'text'={inp['text']}"
        ) from e
    return tokenizer(
        text=prompt + '### Response:',
        text_target=response,
    )


@dataset_constructor.register('HuggingFaceH4/databricks_dolly_15k')
def dolly_tokenize_function(inp: Dict, tokenizer: Tokenizer):
    """Format the text string and tokenize."""
    PROMPT_FORMAT = 'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n'
    try:
        if inp['input'] != '':
            instruction = inp['instruction'] + '\n' + inp['input']
        else:
            instruction = inp['instruction']
        prompt = PROMPT_FORMAT.format(instruction=instruction)
        response = inp['output']
    except Exception as e:
        raise ValueError(
            f'Unable to extract prompt/response from {inp=}') from e
    return tokenizer(
        text=prompt,
        text_target=response,
    )


@dataset_constructor.register('sam-mosaic/full-hh-rlhf-chatml',
                              'sam-mosaic/vicuna_alpaca_hc3_chatml')
def simple_tokenize_function(inp: Dict, tokenizer: Tokenizer):
    """Already split, just tokenize."""
    return tokenizer(
        text=inp['prompt'],
        text_target=inp['response'],
    )


@dataset_constructor.register('bigscience/P3')
def p3_tokenize_function(inp: Dict, tokenizer: Tokenizer):
    """Format the already-split example and tokenize."""
    return tokenizer(
        text=inp['inputs'] + ':',
        text_target=inp['targets'],
    )


# Muennighoff's P3 and flan datasets share a similar convention
@dataset_constructor.register('Muennighoff/P3', 'Muennighoff/flan')
def muennighoff_tokenize_function(inp: Dict, tokenizer: Tokenizer):
    """Format the already-split example and tokenize."""
    try:
        prompt: str = inp['inputs']
        response: str = inp['targets']
        # Put a space before the response if needed
        transitions = (' ', '\n', '\t')
        if not (prompt.endswith(transitions) or
                response.startswith(transitions)):
            response = ' ' + response
    except Exception as e:
        raise ValueError(
            f'Unable to process prompt/response from {inp=}') from e
    return tokenizer(
        text=prompt,
        text_target=response,
    )
