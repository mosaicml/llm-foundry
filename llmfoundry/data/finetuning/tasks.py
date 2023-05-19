# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Includes code for task-specific seq-to-seq data formatting.

This file provides some templates/examples of preprocessing functions
that format examples for use in seq-to-seq finetuning tasks.
These preprocessing functions take individual examples that contain raw
text and process them into formatted examples.

These functions have this basic structure:

    def preprocessing_fn(example: Dict) -> Dict[str, str]:
        # code to extract prompt/response from `example`
        ...
        return {
            'prompt': <prompt>,
            'response': <response>,
        }

where `<prompt>` is a placeholder for the prompt text string that you
extracted from the input example, and '<response>' is a placeholder for
the response text string.

Just to be clear, "prompt" represents the text you would give the model
at inference time, and "response" represents the text you are training
it to produce given the prompt.

The key requirement of these functions is that they return a dictionary
with "prompt" and "response" keys, and that the values associated with
those keys are strings (i.e. text).
"""

import importlib
import os
from typing import Any, Callable, Dict, Optional, Union

import datasets as hf_datasets
from omegaconf import DictConfig
from streaming import StreamingDataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

__all__ = ['dataset_constructor']

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


def _tokenize_formatted_example(example: Dict[str, Any], tokenizer: Tokenizer):
    if ('prompt' not in example) or ('response' not in example):
        raise KeyError(
            'Unable to tokenize example because it has not been properly formatted. ' +\
            '"prompt" and "response" are required keys but at least one was missing ' +\
            f'from {example=}.'
        )
    return tokenizer(text=example['prompt'], text_target=example['response'])


class StreamingFinetuningDataset(StreamingDataset):
    """Finetuning dataset with flexible tokenization using StreamingDataset.

    Args:
        local (str): Local dataset directory where shards are cached by split.
        tokenizer (Tokenizer): The name of the HuggingFace tokenizer to use to
            tokenize samples.
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
                 remote: Optional[str] = None,
                 split: Optional[str] = None,
                 shuffle: bool = False,
                 predownload: Optional[int] = 100_000,
                 keep_zip: bool = False,
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

    # How to process a sample
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = super().__getitem__(idx)
        return _tokenize_formatted_example(sample, tokenizer=self.tokenizer)


class DatasetConstructor:

    def __init__(self):
        self._task_preprocessing_registry: Dict[str, Callable] = {}

    def register(self, *names: str):
        """Decorator for registering preprocessing functions."""

        def _register_func(name: str, func: Callable) -> None:
            if name in self._task_preprocessing_registry:
                raise ValueError(
                    f'A tokenization function has already been registered with {name=}.'
                )
            self._task_preprocessing_registry[name] = func
            return

        def wrapper(func: Callable) -> Callable:
            for name in names:
                _register_func(name, func)
            return func

        return wrapper

    def print_registered_tasks(self):
        tasks = sorted(self._task_preprocessing_registry.keys())
        print('\n'.join(tasks))

    def get_preprocessing_fn_from_str(self,
                                      preprocessor: Optional[str],
                                      dataset_name: Optional[str] = None,
                                      verbose: bool = False):
        """Get a preprocessing function from a string.

        String can be either a registered function or an import path.

        Args:
            preprocessor (Optional[str]): The name of the preprocessing function, or an import path.
            dataset_name (Optional[str]): The dataset name to look up in the registry.
            verbose (bool): Whether to print verbose messages or not.

        Returns:
            Callable: The preprocessing function or None if not found.

        Raises:
            ValueError: If the preprocessing function import from the provided string fails.
        """
        if preprocessor is None:
            if dataset_name is None:
                return None
            if dataset_name in self._task_preprocessing_registry:
                if verbose:
                    print(
                        f'Re-formatting dataset with "{dataset_name}" preprocessing function.'
                    )
                return self._task_preprocessing_registry[dataset_name]
            else:
                if verbose:
                    print(
                        'No preprocessor was supplied and no preprocessing function ' +\
                        f'is registered for dataset name "{dataset_name}". No additional ' +\
                        'preprocessing will be applied. If the dataset is already formatted ' +\
                        'correctly, you can ignore this message.'
                    )
                return None
        if preprocessor in self._task_preprocessing_registry:
            if verbose:
                print(
                    f'Re-formatting dataset with "{preprocessor}" preprocessing function.'
                )
            return self._task_preprocessing_registry[preprocessor]

        try:
            import_path, function_name = preprocessor.split(':', maxsplit=1)
            if verbose:
                print(
                    f'Importing preprocessing function via: `from {import_path} import {function_name}`'
                )
            module = importlib.import_module(import_path)
            preprocessing_fn = getattr(module, function_name)
        except Exception as e:
            raise ValueError(
                f'Failed to import preprocessing function from string = {preprocessor}.'
            ) from e

        return preprocessing_fn

    def build_from_hf(self, cfg: DictConfig, tokenizer: Tokenizer):
        """Load a HuggingFace Datasets, preprocess, and tokenize.

        Args:
            cfg (DictConfig): The dataset configuration.
            tokenizer (Tokenizer): The tokenizer to be used for tokenizing the dataset.

        Returns:
            Dataset: The tokenized dataset.
        """
        dataset_name = cfg.hf_name
        split = cfg.split
        kwargs = cfg.get('hf_kwargs', {})
        preprocessing_fn = self.get_preprocessing_fn_from_str(
            cfg.get('preprocessing_fn'), dataset_name, verbose=True)

        dataset = hf_datasets.load_dataset(dataset_name, split=split, **kwargs)

        def dataset_mapper(example: Dict):
            if preprocessing_fn is not None:
                example = preprocessing_fn(example)
            return _tokenize_formatted_example(example, tokenizer)

        columns_to_remove = list(dataset[0].keys())
        tokenized_dataset = dataset.map(
            dataset_mapper,
            batched=False,
            remove_columns=columns_to_remove,
        )

        return tokenized_dataset

    def build_from_streaming(self, *args: Any, **kwargs: Any):
        return StreamingFinetuningDataset(*args, **kwargs)


dataset_constructor = DatasetConstructor()


@dataset_constructor.register('tatsu-lab/alpaca')
def alpaca_preprocessing_function(inp: Dict):
    """Split out prompt/response from text."""
    try:
        prompt, response = inp['text'].split('### Response:')
        prompt += '### Response:'
    except Exception as e:
        raise ValueError(
            f"Unable to extract prompt/response from 'text'={inp['text']}"
        ) from e
    return {'prompt': prompt, 'response': response}


@dataset_constructor.register('HuggingFaceH4/databricks_dolly_15k')
def dolly_preprocessing_function(inp: Dict):
    """Format the text string."""
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
    return {'prompt': prompt, 'response': response}


@dataset_constructor.register('bigscience/P3')
def p3_preprocessing_function(inp: Dict):
    """Format the already-split example."""
    return {
        'prompt': inp['inputs'] + ':',
        'response': inp['targets'],
    }


# Muennighoff's P3 and flan datasets share a similar convention
@dataset_constructor.register('Muennighoff/P3', 'Muennighoff/flan')
def muennighoff_tokenize_function(inp: Dict):
    """Format the already-split example."""
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
    return {'prompt': prompt, 'response': response}
