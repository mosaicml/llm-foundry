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
import logging
import os
import warnings
from typing import Any, Callable, Dict, List, Optional, Union

import datasets as hf_datasets
from composer.utils import dist
from omegaconf import DictConfig
from streaming import StreamingDataset
from transformers import PreTrainedTokenizerBase

from llmfoundry.utils.logging_utils import SpecificWarningFilter

log = logging.getLogger(__name__)

__all__ = ['dataset_constructor']

_ALLOWED_RESPONSE_KEYS = {'response', 'completion'}
_ALLOWED_PROMPT_KEYS = {'prompt'}


def _tokenize_formatted_example(
        example: Dict[str, Any],
        tokenizer: PreTrainedTokenizerBase) -> Dict[str, List[int]]:
    """Tokenize a formatted example and validate expected keys."""
    example_keys = set(example.keys())
    prompt_keys = example_keys.intersection(_ALLOWED_PROMPT_KEYS)
    response_keys = example_keys.intersection(_ALLOWED_RESPONSE_KEYS)

    if len(prompt_keys) != 1:
        raise KeyError(
            f'Unable to tokenize example because {len(prompt_keys)} of the allowed prompt keys ' +\
            f'were present in {example_keys=}. Please specify exactly one. {_ALLOWED_PROMPT_KEYS=}'
        )

    if len(response_keys) != 1:
        raise KeyError(
            f'Unable to tokenize example because {len(response_keys)} of the allowed response keys ' +\
            f'were present in {example_keys=}. Please specify exactly one. {_ALLOWED_RESPONSE_KEYS=}'
        )

    prompt_key = prompt_keys.pop()
    response_key = response_keys.pop()
    prompt = example[prompt_key]
    response = example[response_key]

    if not isinstance(prompt, str):
        raise TypeError(
            f'Unable to tokenize example because {prompt_key} was not a string. {example=}'
        )

    if not isinstance(response, str):
        raise TypeError(
            f'Unable to tokenize example because {response_key} was not a string. {example=}'
        )

    return tokenizer(text=prompt, text_target=response)


class StreamingFinetuningDataset(StreamingDataset):
    """Finetuning dataset with flexible tokenization using StreamingDataset.

    Args:
        tokenizer (Tokenizer): The name of the HuggingFace tokenizer to use to
            tokenize samples.
        local (str): Local dataset directory where shards are cached by split.
        remote (str, optional): Remote path or directory to download the dataset from. If ``None``,
            its data must exist locally. StreamingDataset uses either ``streams`` or
            ``remote``/``local``. Defaults to ``None``.
        split (str, optional): Which dataset split to use, if any. If provided, we stream from/to
            the ``split`` subdirs of  ``remote`` and ``local``. Defaults to ``None``.
        download_retry (int): Number of download re-attempts before giving up. Defaults to ``2``.
        download_timeout (float): Number of seconds to wait for a shard to download before raising
            an exception. Defaults to ``60``.
        validate_hash (str, optional): Optional hash or checksum algorithm to use to validate
            shards. Defaults to ``None``.
        keep_zip (bool): Whether to keep or delete the compressed form when decompressing
            downloaded shards. If ``False``, keep iff remote is local or no remote. Defaults to
            `False``.
        epoch_size (Union[int, str], optional): Number of samples to draw per epoch balanced across all
            streams. If ``None``, takes its value from the total number of underlying samples.
            Provide this field if you are weighting streams relatively to target a larger or
            smaller epoch size. Defaults to ``None``.
        predownload (int, optional): Target number of samples ahead to download the shards of while
            iterating. If ``None``, its value is set to ``8 * batch_size``. Defaults to ``None``.
        cache_limit (Union[int, str], optional) - Maximum size in bytes of this StreamingDataset's
            shard cache. Before downloading a shard, the least recently used resident shard(s) may
            be evicted (deleted from the local cache) in order to stay under the limit. Set to None
            to disable shard eviction. Supports integer bytes as well as string human-readable
            bytes (e.g., 100b, 64kb, 77mb, and so on). Defaults to None.
        partition_algo (str): Which partitioning algorithm to use. Defaults to ``orig``.
        num_canonical_nodes (int, optional): Canonical number of nodes for shuffling with
            resumption. If ``None``, this is interpreted as 64 times the number of physical
            nodes of the initial run if ``shuffle_algo`` is ``py1s`` or ``py2s``, and simply the
            number of physical nodes of the initial run otherwise. Defaults to ``None``.
        batch_size (int, optional): Batch size of its DataLoader, which affects how the dataset is
            partitioned over the workers. Defaults to ``None``.
        shuffle (bool): Whether to iterate over the samples in randomized order. Defaults to
            ``False``.
        shuffle_algo (str): Which shuffling algorithm to use. Defaults to ``py1e``.
        shuffle_seed (int): Seed for Deterministic data shuffling. Defaults to ``9176``.
        shuffle_block_size (int): Unit of shuffle. If ``None``, its value is calculated as
            ``max(4_000_000 // num_canonical_nodes), 1 << 18)``. Defaults to ``None``.
        sampling_method (str): Which sampling method to use, either ``balanced`` or ``fixed``.
            Defaults to ``balanced``.
        sampling_granularity (int): When picking samples for a stream's final partial repeat,
            how many samples to pick from the same shard at a time (``1`` for evenly balanced
            across shards, ``1000`` to pick 1000 samples from the same shard at a time, etc).
            Defaults to ``1``.
        batching_method (str): Which batching method to use, either ``random``, ``stratified``, or
            ``per_stream``. Defaults to ``random``.
    """

    def __init__(self,
                 tokenizer: PreTrainedTokenizerBase,
                 local: str,
                 remote: Optional[str] = None,
                 split: Optional[str] = None,
                 download_retry: int = 2,
                 download_timeout: float = 60,
                 validate_hash: Optional[str] = None,
                 keep_zip: bool = False,
                 epoch_size: Optional[Union[int, str]] = None,
                 predownload: Optional[int] = None,
                 cache_limit: Optional[Union[int, str]] = None,
                 partition_algo: str = 'relaxed',
                 num_canonical_nodes: Optional[int] = None,
                 batch_size: Optional[int] = None,
                 shuffle: bool = False,
                 shuffle_algo: str = 'py1e',
                 shuffle_seed: int = 9176,
                 shuffle_block_size: Optional[int] = None,
                 sampling_method: str = 'balanced',
                 sampling_granularity: int = 1,
                 batching_method: str = 'random',
                 **kwargs: Any):

        if len(kwargs) > 0:
            raise ValueError(
                f'StreamingFinetuningDataset() got an unexpected keyword argument: {kwargs}'
            )

        if remote is None or (local == remote):
            if os.path.isdir(local):
                contents = set(os.listdir(local))
                if split not in contents:
                    raise ValueError(
                        f'local directory {local} does not contain split {split}'
                    )

        super().__init__(
            local=local,
            remote=remote,
            split=split,
            download_retry=download_retry,
            download_timeout=download_timeout,
            validate_hash=validate_hash,
            keep_zip=keep_zip,
            epoch_size=epoch_size,
            predownload=predownload,
            cache_limit=cache_limit,
            partition_algo=partition_algo,
            num_canonical_nodes=num_canonical_nodes,
            batch_size=batch_size,
            shuffle=shuffle,
            shuffle_algo=shuffle_algo,
            shuffle_seed=shuffle_seed,
            shuffle_block_size=shuffle_block_size,
            sampling_method=sampling_method,
            sampling_granularity=sampling_granularity,
            batching_method=batching_method,
        )

        self.tokenizer = tokenizer

    # How to process a sample
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = super().__getitem__(idx)
        return _tokenize_formatted_example(sample, tokenizer=self.tokenizer)


class DatasetConstructor:

    def __init__(self):
        self._task_preprocessing_registry: Dict[str, Callable] = {}

    def register(self, *names: str) -> Callable[[Callable], Callable]:
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

    def print_registered_tasks(self) -> None:
        tasks = sorted(self._task_preprocessing_registry.keys())
        log.info('\n'.join(tasks))

    def get_preprocessing_fn_from_dict(
        self, mapping: Union[Dict, DictConfig]
    ) -> Callable[[Dict[str, Any]], Dict[str, str]]:
        """Get a preprocessing function from a dictionary.

        The dictionary maps column names in the dataset to "prompt" and "response".
        For example,
            ```yaml
            preprocessing_fn:
                prompt: text
                response: summary
            ```
        would map the `text` column as to prompt and the `summary` column as the response.

        Args:
            mapping (dict): A dictionary mapping column names to "prompt" and "response".

        Returns:
            Callable: The preprocessing function.

        Raises:
            ValueError: If the mapping does not have keys "prompt" and "response".
        """

        def _preprocessor(example: Dict[str, Any]) -> Dict[str, str]:
            if list(mapping.keys()) != ['prompt', 'response']:
                raise ValueError(
                    f'Expected {mapping=} to have keys "prompt" and "response".'
                )
            return {
                'prompt': example[mapping['prompt']],
                'response': example[mapping['response']]
            }

        return _preprocessor

    def get_preprocessing_fn_from_str(
        self,
        preprocessor: Optional[str],
        dataset_name: Optional[str] = None
    ) -> Optional[Callable[[Dict[str, Any]], Dict[str, str]]]:
        """Get a preprocessing function from a string.

        String can be either a registered function or an import path.

        Args:
            preprocessor (Optional[str]): The name of the preprocessing function, or an import path.
            dataset_name (Optional[str]): The dataset name to look up in the registry.

        Returns:
            Callable: The preprocessing function or None if not found.

        Raises:
            ValueError: If the preprocessing function import from the provided string fails.
        """
        if preprocessor is None:
            if dataset_name is None:
                return None
            if dataset_name in self._task_preprocessing_registry:
                log.info(
                    f'Re-formatting dataset with "{dataset_name}" preprocessing function.'
                )
                return self._task_preprocessing_registry[dataset_name]
            else:
                log.info('No preprocessor was supplied and no preprocessing function ' +\
                        f'is registered for dataset name "{dataset_name}". No additional ' +\
                        'preprocessing will be applied. If the dataset is already formatted ' +\
                        'correctly, you can ignore this message.')
                return None
        if preprocessor in self._task_preprocessing_registry:
            log.info(
                f'Re-formatting dataset with "{preprocessor}" preprocessing function.'
            )
            return self._task_preprocessing_registry[preprocessor]

        try:
            import_path, function_name = preprocessor.split(':', maxsplit=1)
            module = importlib.import_module(import_path)
            preprocessing_fn = getattr(module, function_name)
        except Exception as e:
            raise ValueError(
                f'Failed to import preprocessing function from string = {preprocessor}.'
            ) from e

        return preprocessing_fn

    def build_from_hf(
        self, cfg: DictConfig, max_seq_len: int,
        tokenizer: PreTrainedTokenizerBase
    ) -> Union[hf_datasets.DatasetDict, hf_datasets.Dataset,
               hf_datasets.IterableDatasetDict, hf_datasets.IterableDataset]:
        """Load a HuggingFace Datasets, preprocess, and tokenize.

        Note: This function will drop examples where the prompt is longer than the max_seq_len

        Args:
            cfg (DictConfig): The dataset configuration.
            max_seq_len (int): The maximum sequence length. Examples with prompts longer than this will be dropped.
            tokenizer (Tokenizer): The tokenizer to be used for tokenizing the dataset.

        Returns:
            Dataset: The tokenized dataset.
        """
        dataset_name = cfg.hf_name
        # HF datasets does not support a split with dashes,so we replace split
        # dashes with underscore.
        split = cfg.split.replace('-', '_')
        kwargs = cfg.get('hf_kwargs', {})
        proto_preprocessing_fn = cfg.get('preprocessing_fn')
        if isinstance(proto_preprocessing_fn, dict) or isinstance(
                proto_preprocessing_fn, DictConfig):
            preprocessing_fn = self.get_preprocessing_fn_from_dict(
                proto_preprocessing_fn)
        else:
            preprocessing_fn = self.get_preprocessing_fn_from_str(
                proto_preprocessing_fn, dataset_name)

        signal_file_path = f'.node_{dist.get_node_rank()}_local_rank0_data_prep_completed'

        # Non local rank 0 ranks will wait here for local rank 0 to finish the data processing.
        # Once local rank 0 is done, the datasets are all cached on disk, and all other ranks
        # can just read them.
        if dist.get_local_rank() != 0:
            log.debug('Waiting for local_rank 0 to finish data prep')
            with dist.local_rank_zero_download_and_wait(signal_file_path):
                pass

        hf_tokenization_logger = logging.getLogger(
            'transformers.tokenization_utils_base')
        sequence_length_warning_filter = SpecificWarningFilter(
            'Token indices sequence length is longer than the specified maximum sequence length'
        )

        # We will trim examples later in the collate_fn, so we want to silence this warning from Hugging Face
        hf_tokenization_logger.addFilter(sequence_length_warning_filter)

        error: Optional[Exception] = None
        filtered_dataset = None
        try:
            dataset = hf_datasets.load_dataset(dataset_name,
                                               split=split,
                                               **kwargs)

            def dataset_mapper(example: Dict):
                if preprocessing_fn is not None:
                    example = preprocessing_fn(example)
                return _tokenize_formatted_example(example, tokenizer)

            detected_cpu_count = os.cpu_count() or 1
            detected_cpus_with_margin = detected_cpu_count - 8
            num_cpus_to_use = max(1, detected_cpus_with_margin)

            columns_to_remove = list(dataset[0].keys())
            tokenized_dataset = dataset.map(
                dataset_mapper,
                batched=False,
                remove_columns=columns_to_remove,
                num_proc=num_cpus_to_use,
                desc='Tokenizing dataset',
            )

            pad_token_id = tokenizer.pad_token_id

            def filter_long_or_empty_examples(example: Dict) -> bool:
                less_than_max_seq_len = len(example['input_ids']) < max_seq_len
                non_empty_input = len(example['input_ids']) > 0
                non_empty_labels = len(example['labels']) > 0
                non_padding_response = any(
                    token_id != pad_token_id for token_id in example['labels'])
                return (less_than_max_seq_len and non_empty_input and
                        non_empty_labels and non_padding_response)

            filtered_dataset = tokenized_dataset.filter(
                filter_long_or_empty_examples,
                num_proc=num_cpus_to_use,
                desc='Filtering out long prompts',
            )

            examples_removed = len(tokenized_dataset) - len(filtered_dataset)
            if examples_removed > 0:
                warnings.warn(
                    f'Dropped {examples_removed} examples where the prompt was longer than {max_seq_len}, '
                    +
                    'the prompt or response was empty, or the response was all padding tokens.'
                )
        except Exception as e:
            error = e
        # Now local rank 0 indicates to the other ranks that it is done
        if dist.get_local_rank() == 0:
            log.debug('Local rank 0 finished data prep')
            with open(signal_file_path, 'wb') as f:
                f.write(b'local_rank0_completed_data_prep')

        # All ranks sync up at this barrier, having completed data processing
        dist.barrier()

        # Last, local rank 0 cleans up the signal file
        if dist.get_local_rank() == 0:
            os.remove(signal_file_path)

        if error is not None:
            log.error('Error during data prep')
            raise error
        log.debug('All ranks finished data prep')

        hf_tokenization_logger.removeFilter(sequence_length_warning_filter)

        assert filtered_dataset is not None
        return filtered_dataset

    def build_from_streaming(self, *args: Any,
                             **kwargs: Any) -> StreamingFinetuningDataset:
        return StreamingFinetuningDataset(*args, **kwargs)


dataset_constructor = DatasetConstructor()


@dataset_constructor.register('tatsu-lab/alpaca')
def alpaca_preprocessing_function(inp: Dict) -> Dict[str, str]:
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
def dolly_preprocessing_function(inp: Dict) -> Dict[str, str]:
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
def p3_preprocessing_function(inp: Dict) -> Dict[str, str]:
    """Format the already-split example."""
    return {
        'prompt': inp['inputs'] + ':',
        'response': inp['targets'],
    }


# Muennighoff's P3 and flan datasets share a similar convention
@dataset_constructor.register('Muennighoff/P3', 'Muennighoff/flan')
def muennighoff_tokenize_function(inp: Dict) -> Dict[str, str]:
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
