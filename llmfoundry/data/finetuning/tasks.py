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
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import (
    Any,
    Callable,
    Literal,
    Optional,
    Sequence,
    Union,
    cast,
)

import datasets as hf_datasets
import datasets.exceptions as hf_exceptions
import huggingface_hub as hf_hub
import numpy as np
from composer.utils import dist
from streaming import Stream, StreamingDataset
from transformers import PreTrainedTokenizerBase

from llmfoundry.data import (
    SUPPORTED_MDS_ENCODING_TYPES,
    stream_remote_local_validate,
)
from llmfoundry.data.finetuning.collator import (
    stitch_turns_decoder_only,
    stitch_turns_encoder_decoder,
)
from llmfoundry.tokenizers import get_date_string
from llmfoundry.utils.consts import CROSS_ENTROPY_IGNORE_INDEX
# yapf: disable
from llmfoundry.utils.exceptions import (
    ALLOWED_MESSAGES_KEYS,
    ALLOWED_PROMPT_KEYS,
    ALLOWED_RESPONSE_KEYS,
    ChatTemplateError,
    ConsecutiveRepeatedChatRolesError,
    DatasetTooSmallError,
    IncorrectMessageKeyQuantityError,
    InvalidContentTypeError,
    InvalidConversationError,
    InvalidDatasetError,
    InvalidExampleTypeError,
    InvalidFileExtensionError,
    InvalidLastChatMessageRoleError,
    InvalidMessageTypeError,
    InvalidPromptResponseKeysError,
    InvalidPromptTypeError,
    InvalidResponseTypeError,
    InvalidRoleError,
    MisconfiguredHfDatasetError,
    NotEnoughChatDataError,
    UnableToProcessPromptResponseError,
    UnknownExampleTypeError,
)
#  yapf: enable
from llmfoundry.utils.file_utils import dist_mkdtemp
from llmfoundry.utils.logging_utils import SpecificWarningFilter

log = logging.getLogger(__name__)

__all__ = [
    'dataset_constructor',
    'tokenize_formatted_example',
    'is_valid_ift_example',
    'StreamingFinetuningDataset',
]

_ALLOWED_ROLE_KEYS = {'role'}
_ALLOWED_CONTENT_KEYS = {'content'}
_ALLOWED_ROLES = {'user', 'assistant', 'system', 'tool'}
_ALLOWED_LAST_MESSAGE_ROLES = {'assistant'}
SUPPORTED_EXTENSIONS = ['.csv', '.json', '.jsonl', '.parquet']
HUGGINGFACE_FOLDER_EXTENSIONS = ['.lock', '.metadata']
DEFAULT_TARGET_RESPONSES = 'last'
DEFAULT_TARGET_PROMPTS = 'none'

PromptResponseDict = Mapping[str, str]
ChatFormattedDict = Mapping[str, list[dict[str, str]]]
Example = Union[PromptResponseDict, ChatFormattedDict]
ExampleType = Literal['prompt_response', 'chat']
TokenizedExample = dict[str, list[dict[str, list[int]]]]

_DEFAULT_CHAT_TEMPLATE = (
    '{% for message in messages %}'
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    '{% endfor %}'
    '{% if add_generation_prompt %}'
    "{{ '<|im_start|>assistant\n' }}"
    '{% endif %}'
)


def _get_example_type(example: Example) -> ExampleType:
    """Determines the type of the input example.

    Args:
        example (Example): The input example, which can be a multi-way chat formatted conversation or an instruction-response pair.

    Returns:
        ExampleType: The type of the input example, which can be either 'chat' for multi-way chat formatted conversation or 'prompt_response' for instruction-response pair.

    Raises:
        KeyError: If the example type is unknown.
    """
    if not isinstance(example, Mapping):
        raise InvalidExampleTypeError(str(type(example)))
    if (
        len(example.keys()) == 1 and any(
            allowed_message_key in example
            for allowed_message_key in ALLOWED_MESSAGES_KEYS
        )
    ):
        return 'chat'
    elif (
        len(example.keys()) == 2 and
        any(p in example for p in ALLOWED_PROMPT_KEYS) and
        any(r in example for r in ALLOWED_RESPONSE_KEYS)
    ):
        return 'prompt_response'
    else:
        keys = str(set(example.keys()))
        raise UnknownExampleTypeError(keys)


def _is_empty_or_nonexistent(dirpath: str) -> bool:
    """Check if a directory is empty or non-existent.

    Args:
        dirpath (str): Directory path to check.

    Returns:
        True if directory is empty or non-existent. False otherwise.
    """
    return not os.path.isdir(dirpath) or len(os.listdir(dirpath)) == 0


def _get_key(dictionary: Mapping[str, Any], allowed_keys: set[str]):
    if not isinstance(dictionary, Mapping):
        raise InvalidExampleTypeError(str(type(dictionary)))
    desired_keys = allowed_keys.intersection(dictionary.keys())
    if len(desired_keys) == 0:
        raise UnknownExampleTypeError(str(set(dictionary.keys())))
    return list(desired_keys)[0]


def _validate_chat_formatted_example(example: ChatFormattedDict):
    if not isinstance(example, Mapping):
        raise InvalidExampleTypeError(str(type(example)))
    messages = example[_get_key(example, ALLOWED_MESSAGES_KEYS)]
    if not isinstance(messages, list):
        raise InvalidMessageTypeError(str(type(messages)))
    if len(messages) <= 1:
        raise NotEnoughChatDataError()

    last_message = messages[-1]
    role_key = _get_key(last_message, _ALLOWED_ROLE_KEYS)
    last_role = last_message[role_key]
    if last_role not in _ALLOWED_LAST_MESSAGE_ROLES:
        raise InvalidLastChatMessageRoleError(
            last_role,
            _ALLOWED_LAST_MESSAGE_ROLES,
        )

    last_message_role = None
    for message in messages:
        role_key, content_key = _get_key(message, _ALLOWED_ROLE_KEYS), _get_key(
            message,
            _ALLOWED_CONTENT_KEYS,
        )
        if len(message.keys()) != 2:
            raise IncorrectMessageKeyQuantityError(list(message.keys()))
        if message[role_key] not in _ALLOWED_ROLES:
            raise InvalidRoleError(message[role_key], _ALLOWED_ROLES)
        if not isinstance(message[content_key], (list, str)):
            # this validation check accepts text-only as well as multimodal messages for chat-enabled models
            raise InvalidContentTypeError(type(message[content_key]))
        if last_message_role is not None and last_message_role == message[
            role_key]:
            raise ConsecutiveRepeatedChatRolesError(last_message_role)
        last_message_role = message[role_key]


def _slice_chat_formatted_example(
    example: ChatFormattedDict,
    tokenizer: PreTrainedTokenizerBase,
) -> list[tuple[str, str]]:
    """Slices chat example into a list of templated prompt, response turns.

    Note: Assistant messages mark the end of chat turns. So there are as many turns as there are
        assistant messages in the chat example.

    Args:
        example (ChatFormattedDict): The chat example containing the messages.
        tokenizer (PreTrainedTokenizerBase): The tokenizer to apply the chat template.

    Returns:
        List[Tuple[str, str]]: A list of templated prompt and response string pairs, one pair per chat turn.

    Raises:
        ValueError: If any chat turn in the example has less than two messages or if the last message is not from the assistant.
        KeyError: If a message does not have a role or content.
    """
    _validate_chat_formatted_example(example)
    messages = example[_get_key(example, ALLOWED_MESSAGES_KEYS)]

    last_message = messages[-1]
    if last_message['role'] != 'assistant':
        raise InvalidLastChatMessageRoleError(
            last_message['role'],
            set('assistant'),
        )

    def slice_out_last_turn(
        messages_through_current_turn: list[dict[str, str]],
        conversation_through_previous_turn: str,
    ) -> tuple[str, str]:
        chat_template = None if tokenizer.chat_template is not None else _DEFAULT_CHAT_TEMPLATE

        try:
            full_conversation = tokenizer.apply_chat_template(
                messages_through_current_turn,
                tokenize=False,
                date_string=get_date_string(),
                chat_template=chat_template,
            )
            prompt_with_history = tokenizer.apply_chat_template(
                messages_through_current_turn[:-1],
                tokenize=False,
                add_generation_prompt=True,
                date_string=get_date_string(),
                chat_template=chat_template,
            )
        except Exception as e:
            raise ChatTemplateError(
                tokenizer.chat_template,  # type: ignore
                sample=messages_through_current_turn,
                inner_message=str(e),
            )
        if conversation_through_previous_turn != full_conversation[:len(  # type: ignore
            conversation_through_previous_turn,
        )]:
            raise InvalidConversationError(
                f'The full conversation must start with the conversation through the previous turn. {conversation_through_previous_turn=}, {full_conversation=}',
            )
        if conversation_through_previous_turn != prompt_with_history[:len(  # type: ignore
            conversation_through_previous_turn,
        )]:
            raise InvalidConversationError(
                f'The prompt_with_history must start with the conversation through the previous turn. {conversation_through_previous_turn=}, {prompt_with_history=}',
            )
        if prompt_with_history != full_conversation[:len(prompt_with_history)
                                                   ]:  # type: ignore
            raise InvalidConversationError(
                f'prompt_with_history must be the first part of the full conversation. {prompt_with_history=}, {full_conversation=}',
            )
        prompt = prompt_with_history[len(conversation_through_previous_turn):
                                    ]  # type: ignore
        response = full_conversation[len(prompt_with_history):]  # type: ignore
        return prompt, response  # type: ignore

    templated_prompt_response_turns: list[tuple[str, str]] = []
    conversation_through_previous_turn = ''
    for idx, message in enumerate(messages):
        if message['role'] == 'assistant':
            prompt, response = slice_out_last_turn(
                messages[:idx + 1],
                conversation_through_previous_turn,
            )
            templated_prompt_response_turns.append((prompt, response))
            conversation_through_previous_turn += prompt
            conversation_through_previous_turn += response

    return templated_prompt_response_turns


def _tokenize_with_bos_removal(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    text_target: str,
) -> dict[str, list[int]]:
    """Tokenizes the prompt and response using the provided tokenizer.

    Args:
        tokenizer (PreTrainedTokenizerBase): The tokenizer to use for tokenization.
        text (str): The prompt to tokenize.
        text_target (str): The response to tokenize.

    Returns:
        Dict[str, List[int]]: The tokenized text and text_target.
    """
    tokenized_sample = tokenizer(
        text=text,
        text_target=text_target,
        padding=False,
        truncation=False,
    )

    # Remove the BOS token from the start of the labels if it was automatically added
    # Unfortunately if the tokenizer is PretrainedTokenizerFast, as llama3 is, it does not provide
    # an add_bos_token attr that we can check explicitly, so instead we rely on checking if both the
    # text and the text_target start with bos_token_id to determine whether to strip bos.
    has_bos_token = hasattr(
        tokenizer,
        'bos_token_id',
    ) and tokenizer.bos_token_id is not None
    input_ids_starts_with_bos = False
    labels_starts_with_bos = False
    if has_bos_token and len(
        cast(Any, tokenized_sample['input_ids']),
    ) > 0 and len(cast(Any, tokenized_sample['labels'])) > 0:
        input_ids_starts_with_bos = cast(
            Any,
            tokenized_sample['input_ids'],
        )[0] == tokenizer.bos_token_id
        labels_starts_with_bos = cast(
            Any,
            tokenized_sample['labels'],
        )[0] == tokenizer.bos_token_id
    if input_ids_starts_with_bos and labels_starts_with_bos:
        tokenized_sample['labels'] = cast(Any, tokenized_sample['labels'])[1:]

    return tokenized_sample  # type: ignore


def _tokenize_chat_formatted_example(
    example: ChatFormattedDict,
    tokenizer: PreTrainedTokenizerBase,
) -> TokenizedExample:
    """Tokenizes a chat-formatted example using the provided tokenizer.

    Args:
        example (ChatFormattedDict): The chat-formatted example to tokenize.
        tokenizer (PreTrainedTokenizerBase): The tokenizer to use for tokenization.

    Returns:
        TokenizedExample: The tokenized example.
    """
    # Note: We do not add special tokens when tokenizing chat-formatted examples because
    # special tokens are expected to be added via the tokenizer's chat template. So,
    # we instead expect the prompt/response outputs of `_slice_chat_formatted_example`
    # (which calls `apply_chat_template`) to have the correct special tokens already.
    # We disable padding and truncation because those are handled in the collator, which needs to
    # be able to assume that none of the tokens are pad tokens.
    return {  # type: ignore
        'turns': [
            tokenizer(
                text=prompt,
                text_target=response,
                add_special_tokens=False,
                padding=False,
                truncation=False,
            ) for prompt, response in
            _slice_chat_formatted_example(example, tokenizer)
        ],
    }


def _tokenize_prompt_response_formatted_example(
    example: PromptResponseDict,
    tokenizer: PreTrainedTokenizerBase,
) -> TokenizedExample:
    """Tokenize a formatted example and validate expected keys."""
    example_keys = set(example.keys())
    prompt_keys = example_keys.intersection(ALLOWED_PROMPT_KEYS)
    response_keys = example_keys.intersection(ALLOWED_RESPONSE_KEYS)

    prompt_key = prompt_keys.pop()
    response_key = response_keys.pop()
    prompt = example[prompt_key]
    response = example[response_key]
    if not isinstance(prompt, str):
        raise InvalidPromptTypeError(type(prompt))

    if not isinstance(response, str):
        raise InvalidResponseTypeError(type(response))

    # Note: We default to the tokenizer's add_bos_token and add_eos_token behavior here
    # (which we do not do for chat-formatted examples). This is because chat examples specifically
    # go through the tokenizer's `apply_chat_template` method, which handles special tokens,
    # and these prompt-response-formatted examples do not.
    # We disable padding and truncation because those are handled in the collator, which needs to
    # be able to assume that none of the tokens are pad tokens.
    return {
        'turns': [
            _tokenize_with_bos_removal(
                tokenizer=tokenizer,
                text=prompt,
                text_target=response,
            ),
        ],
    }


def tokenize_formatted_example(
    example: Example,
    tokenizer: PreTrainedTokenizerBase,
) -> TokenizedExample:
    """Tokenizes a formatted example using the provided tokenizer.

    Args:
        example (Example): The input example to be tokenized.
        tokenizer (PreTrainedTokenizerBase): The tokenizer to be used for tokenization.

    Returns:
        TokenizedExample: The tokenized example.

    Raises:
        ValueError: If the example format is unknown.
    """
    example_format = _get_example_type(example)

    if example_format == 'chat':
        chat_example = cast(ChatFormattedDict, example)
        return _tokenize_chat_formatted_example(chat_example, tokenizer)
    elif example_format == 'prompt_response':
        prompt_response_example: PromptResponseDict = cast(
            PromptResponseDict,
            example,
        )
        return _tokenize_prompt_response_formatted_example(
            prompt_response_example,
            tokenizer,
        )
    else:
        raise NotImplementedError


def is_valid_ift_example(
    max_seq_len: int,
    target_prompts: str,
    target_responses: str,
    decoder_only_format: bool,
    example: TokenizedExample,
) -> bool:
    """Check if the example is a valid ift example.

    This function confirms that none of the ``input_ids`` and ``labels`` fields
    are empty in any of the turns within the example.

    This function also prepares the final input_ids and labels
    of the (potentially multi-turn) example, using the target settings
    and format, and checks whether they are suitable for training at max_seq_len.
    The example is not valid if (1) after truncation (if necessary),
    the training targets contain no loss-generating tokens, or (2) either the
    input_ids and labels are empty.

    The token sequences in ``example`` are assumed to not have had
    any padding or truncation applied already.

    Args:
        max_seq_len (int): Maximum sequence length.
        target_prompts (str): The prompts that are used as targets.
        target_responses (str): The responses that are used as targets.
        decoder_only_format (bool): Whether the data will be formatted
            for a decoder-only model.
        example (Dict): The input example after tokenization, which has
            a list of dicts, each with ``input_ids`` and ``labels`` fields.

    Returns:
        bool: Indicator of whether the input example is valid
    """
    for turn in example['turns']:
        if len(turn['input_ids']) == 0:
            return False
        if len(turn['labels']) == 0:
            return False

    if decoder_only_format:
        input_ids, labels = stitch_turns_decoder_only(
            example_turns=example['turns'],
            target_prompts=target_prompts,
            target_responses=target_responses,
        )

    else:
        input_ids, labels = stitch_turns_encoder_decoder(
            example_turns=example['turns'],
        )
    input_ids = input_ids[:max_seq_len]
    labels = labels[:max_seq_len]

    if len(input_ids) == 0:
        return False

    if len([label for label in labels if label != CROSS_ENTROPY_IGNORE_INDEX
           ],) == 0:
        return False

    return True


def _get_num_processes() -> int:
    """Get the number of processes to use for dataset processing."""
    detected_cpu_count = os.cpu_count() or 1
    detected_cpus_with_margin = detected_cpu_count - 8
    num_proc = max(1, detected_cpus_with_margin)

    # Check if the user has set the MAX_NUM_PROC environment variable
    # which caps the number of processes used for dataset processing.
    if 'MAX_NUM_PROC' in os.environ:
        max_num_proc_env = int(os.environ['MAX_NUM_PROC'])
        if max_num_proc_env < num_proc:
            num_proc = max_num_proc_env

    return num_proc


class StreamingFinetuningDataset(StreamingDataset):
    """Finetuning dataset with flexible tokenization using StreamingDataset.

    Args:
        tokenizer (Tokenizer): The name of the HuggingFace tokenizer to use to
            tokenize samples.
        token_encoding_type (str): The encoding type of the tokenized samples. This is only used
            for legacy datasets that have been written directly as 'bytes' instead of numpy
            arrays. Types are auto-inferred for numpy arrays. Defaults to 'int64'.
        streams (Sequence[Stream], optional): One or more Streams to stream/cache samples from,
            which may be upsampled or downsampled. StreamingDataset uses either ``streams`` or
            ``remote``/``local``. Defaults to ``None``.
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
        allow_unsafe_types (bool): If a shard contains Pickle, which allows arbitrary code
            execution during deserialization, whether to keep going if ``True`` or raise an error
            if ``False``. Defaults to ``False``.
        replication (int, optional): Determines how many consecutive devices will receive the same
            samples. Useful for training with tensor or sequence parallelism, where multiple
            devices need to see the same partition of the dataset. Defaults to ``None``.
        stream_name (str): The name of the Stream to use which is registered in
            streaming.base.stream.streams_registry. Defaults to ``stream``.
        stream_config (dict[str, Any]): Additional arguments to pass to the Stream constructor.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        token_encoding_type: str = 'int64',
        streams: Optional[Sequence[Stream]] = None,
        local: Optional[str] = None,
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
        max_seq_len: int = 2048,
        allow_unsafe_types: bool = False,
        replication: Optional[int] = None,
        packing_ratio: Optional[float] = None,
        stream_name: str = 'stream',
        stream_config: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ):

        if token_encoding_type not in SUPPORTED_MDS_ENCODING_TYPES:
            raise ValueError(
                f'The token_encoding_type must be one of {SUPPORTED_MDS_ENCODING_TYPES}, but got {token_encoding_type}',
            )
        self.token_encoding_type = token_encoding_type

        if streams is None:
            stream_remote_local_validate(remote, local, split)
        else:
            for stream in streams:
                stream_remote_local_validate(
                    stream.remote,
                    stream.local,
                    split,
                )

        super().__init__(
            streams=streams,
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
            allow_unsafe_types=allow_unsafe_types,
            replication=replication,
            stream_name=stream_name,
            stream_config=stream_config,
            **kwargs,
        )

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.packing_ratio = packing_ratio

    def tokenize_example(self, example: Example) -> TokenizedExample:
        return tokenize_formatted_example(example, self.tokenizer)

    # How to process a sample
    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = super().__getitem__(idx)
        if 'turns' in sample:
            # Already tokenized in latest format
            return sample
        if 'input_ids' in sample:
            # Already tokenized data (old format)
            if isinstance(sample['input_ids'], bytes):
                sample['input_ids'] = np.frombuffer(
                    sample['input_ids'],
                    dtype=getattr(np, self.token_encoding_type),
                )[:self.max_seq_len].tolist().copy()
                sample['labels'] = np.frombuffer(
                    sample['labels'],
                    dtype=getattr(np, self.token_encoding_type),
                )[:self.max_seq_len].tolist().copy()
            elif isinstance(sample['input_ids'], np.ndarray):
                sample['input_ids'] = sample['input_ids'][:self.max_seq_len
                                                         ].tolist().copy()
                sample['labels'] = sample['labels'][:self.max_seq_len].tolist(
                ).copy()
            else:
                input_ids_type = type(sample['input_ids'])
                raise ValueError(
                    f'Expect input_ids to be bytes or numpy.ndarray type, but got {input_ids_type}',
                )
            # Convert to latest format by wrapping sample as a "turn"
            return {'turns': [sample]}
        return self.tokenize_example(sample)

    def state_dict(self, num_samples: int,
                   from_beginning: bool) -> dict[str, Any]:
        if self.packing_ratio is not None:
            num_samples = int(self.packing_ratio * num_samples)

        return super().state_dict(
            num_samples=num_samples,
            from_beginning=from_beginning,
        )


class DatasetConstructor:

    def __init__(self):
        self._task_preprocessing_registry: dict[str, Callable] = {}

    def register(self, *names: str) -> Callable[[Callable], Callable]:
        """Decorator for registering preprocessing functions."""

        def _register_func(name: str, func: Callable) -> None:
            if name in self._task_preprocessing_registry:
                raise ValueError(
                    f'A tokenization function has already been registered with {name=}.',
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
        self,
        mapping: dict[str, str],
    ) -> Callable[[dict[str, Any]], Example]:
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

        def _preprocessor(example: dict[str, Any]) -> dict[str, str]:
            if list(mapping.keys()) != ['prompt', 'response']:
                raise InvalidPromptResponseKeysError(mapping, example)
            return {
                'prompt': example[mapping['prompt']],
                'response': example[mapping['response']],
            }

        return _preprocessor

    def get_preprocessing_fn_from_str(
        self,
        preprocessor: Optional[str],
        dataset_name: Optional[str] = None,
    ) -> Optional[Callable[[dict[str, Any]], Example]]:
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
                    f'Re-formatting dataset with "{dataset_name}" preprocessing function.',
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
                f'Re-formatting dataset with "{preprocessor}" preprocessing function.',
            )
            return self._task_preprocessing_registry[preprocessor]

        try:
            import_path, function_name = preprocessor.split(':', maxsplit=1)
            module = importlib.import_module(import_path)
            preprocessing_fn = getattr(module, function_name)
        except Exception as e:
            raise ValueError(
                f'Failed to import preprocessing function from string = {preprocessor}.',
            ) from e

        return preprocessing_fn

    def build_from_hf(
        self,
        dataset_name: str,
        split: str,
        safe_load: bool = False,
        max_seq_len: int = 2048,
        mapping_fn: Callable = tokenize_formatted_example,
        preprocessing_fn: Optional[Callable[[dict[str, Any]], Example]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        target_prompts: str = DEFAULT_TARGET_PROMPTS,
        target_responses: str = DEFAULT_TARGET_RESPONSES,
        decoder_only_format: bool = True,
        hf_kwargs: Optional[dict[str, Any]] = None,
    ) -> Union[hf_datasets.DatasetDict, hf_datasets.Dataset,
               hf_datasets.IterableDatasetDict, hf_datasets.IterableDataset]:
        """Load a HuggingFace Datasets, preprocess, and tokenize.

        Note: This function will drop examples where the prompt is longer than the max_seq_len

        Args:
            dataset_name (str): The name of the HuggingFace dataset
                to use. Can also be a remote http(s) directory or object store bucket
                containing the file {split}.jsonl in the format (prompt, response),
                in which case the builder will create a HuggingFace dataset.
            split (str): The split of the HuggingFace dataset.
            safe_load (bool, optional): Whether to enforce safe loading of the dataset.
                If `None`, will default to not applying any safe loading.
            max_seq_len (int): The maximum length of sequences
                in the batch. See :class:`Seq2SeqFinetuningCollator` docstring
                for details.
            mapping_fn (Callable): The mapping function to use for mapping the data
                examples.
            preprocessing_fn (Callable, optional): The preprocessing function to use for
                formatting the data examples.
            tokenizer (PreTrainedTokenizerBase): The tokenizer to be used for tokenizing
                the HuggingFace dataset.
            target_prompts (str): Which prompts are used as training targets.
                Defaults to "none", meaning prompts are never used as training targets.
                See :class:`Seq2SeqFinetuningCollator` docstring for details.
            target_responses (str): Which responses are used as training targets.
                Defaults to "last", meaning only the final response in multi-turn examples
                will serve as training targets. See :class:`Seq2SeqFinetuningCollator` docstring for
                details.
            decoder_only_format (bool): Whether to format the
                examples for a decoder-only model. See :class:`Seq2SeqFinetuningCollator`
                docstring for details.
            hf_kwargs (DictConfig, optional): Additional kwargs to
                pass to `datasets.load_dataset`, which can be used to load
                a dataset from local files.

        Returns:
            Dataset: The tokenized dataset.
        """
        if hf_kwargs is None:
            hf_kwargs = {}

        # None is checked in the function, because argument defaults were added after the function was written and we want
        # to preserve the ordering of the arguments for backwards compatibility.
        if tokenizer is None:
            raise ValueError('A tokenizer must be provided.')

        signal_file_path = dist.get_node_signal_file_name()

        download_folder = dist_mkdtemp()

        # Non local rank 0 ranks will wait here for local rank 0 to finish the data processing.
        # Once local rank 0 is done, the datasets are all cached on disk, and all other ranks
        # can just read them.
        if dist.get_local_rank() != 0:
            log.debug('Waiting for local_rank 0 to finish data prep')
            with dist.local_rank_zero_download_and_wait(signal_file_path):
                pass

        hf_tokenization_logger = logging.getLogger(
            'transformers.tokenization_utils_base',
        )
        sequence_length_warning_filter = SpecificWarningFilter(
            'Token indices sequence length is longer than the specified maximum sequence length',
        )

        # We will trim examples later in the collate_fn, so we want to silence this warning from Hugging Face
        hf_tokenization_logger.addFilter(sequence_length_warning_filter)

        error: Optional[Exception] = None
        filtered_dataset = None
        try:
            if safe_load:
                if not os.path.isdir(dataset_name):
                    # dataset_name is not a local dir path, download if needed.
                    local_dataset_dir = os.path.join(
                        download_folder,
                        dataset_name,
                    )

                    log.debug(
                        f'Downloading dataset {dataset_name} to {local_dataset_dir}.',
                    )

                    if _is_empty_or_nonexistent(dirpath=local_dataset_dir):
                        # Safely load a dataset from HF Hub with restricted file types.
                        hf_hub.snapshot_download(
                            dataset_name,
                            repo_type='dataset',
                            allow_patterns=[
                                '*' + ext for ext in SUPPORTED_EXTENSIONS
                            ],
                            token=hf_kwargs.get('token', None),
                            revision=hf_kwargs.get('revision', None),
                            local_dir_use_symlinks=False,
                            local_dir=local_dataset_dir,
                        )
                        if _is_empty_or_nonexistent(dirpath=local_dataset_dir):
                            raise InvalidFileExtensionError(
                                dataset_name,
                                SUPPORTED_EXTENSIONS,
                            )
                    # Set dataset_name to the downloaded location.
                    dataset_name = local_dataset_dir

                # dataset_name is a local dir path. Use the abspath to prevent confusion.
                dataset_name = os.path.abspath(dataset_name)

                # Ensure that the local dir contains only allowed file types.
                dataset_files = [
                    f for _, _, files in os.walk(dataset_name) for f in files
                ]
                if not all(
                    Path(f).suffix in SUPPORTED_EXTENSIONS +
                    HUGGINGFACE_FOLDER_EXTENSIONS or f == '.gitignore'
                    for f in dataset_files
                ):
                    raise InvalidFileExtensionError(
                        dataset_name,
                        SUPPORTED_EXTENSIONS,
                    )

            dataset = hf_datasets.load_dataset(
                dataset_name,
                split=split,
                **hf_kwargs,
            )

            def dataset_mapper(example: dict):
                if preprocessing_fn is not None:
                    return mapping_fn(
                        preprocessing_fn(example),
                        tokenizer,
                    )
                return mapping_fn(example, tokenizer)

            num_proc = _get_num_processes()
            if len(dataset) < num_proc:
                num_proc = 1

            columns_to_remove = list(dataset[0].keys())
            tokenized_dataset = dataset.map(
                dataset_mapper,
                batched=False,
                remove_columns=columns_to_remove,
                num_proc=num_proc,
                desc='Tokenizing dataset',
            )

            filtered_dataset = tokenized_dataset.filter(
                partial(
                    is_valid_ift_example,
                    max_seq_len,
                    target_prompts,
                    target_responses,
                    decoder_only_format,
                ),
                num_proc=num_proc,
                desc='Filtering out long prompts',
            )

            examples_removed = len(tokenized_dataset) - len(filtered_dataset)
            if examples_removed > 0:
                warnings.warn(
                    f'Dropped {examples_removed} examples where the prompt was longer than {max_seq_len}, '
                    +
                    'the prompt or response was empty, or the response was all padding tokens.',
                )
            if len(filtered_dataset) == 0:
                raise InvalidDatasetError(
                    f'No valid examples found after filtering out prompts longer than {max_seq_len}, '
                    +
                    'examples with empty prompts or responses, and examples with responses that are all padding tokens.',
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

        if isinstance(error, hf_exceptions.DatasetGenerationError):
            log.error('Huggingface DatasetGenerationError during data prep.')
            raise MisconfiguredHfDatasetError(
                dataset_name=dataset_name,
                split=split,
            ) from error
        if error is not None:
            log.error('Error during data prep')
            raise error
        log.debug('All ranks finished data prep')

        hf_tokenization_logger.removeFilter(sequence_length_warning_filter)

        assert filtered_dataset is not None
        return filtered_dataset

    @property
    def streaming_dataset_class(self) -> type[StreamingFinetuningDataset]:
        return StreamingFinetuningDataset

    def build_from_streaming(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> StreamingFinetuningDataset:
        dataset = self.streaming_dataset_class(*args, **kwargs)
        num_canonical_nodes = dataset.num_canonical_nodes
        num_samples = dataset.num_samples
        if num_canonical_nodes is None:
            num_physical_nodes = dist.get_world_size(
            ) // dist.get_local_world_size()
            if num_samples < num_physical_nodes:
                raise DatasetTooSmallError(
                    f'{num_samples=} is less than {dist.get_world_size() // dist.get_local_world_size()}, the number of physical nodes. ',
                )

        if num_canonical_nodes is not None and num_samples < num_canonical_nodes:
            raise DatasetTooSmallError(
                f'{num_samples=} is less than {num_canonical_nodes=}. ' +
                'Please check your index.json file and ensure that your dataset has been written out correctly.'
                + 'If this was intended, reduce num_canonical_nodes.',
            )
        return dataset


dataset_constructor = DatasetConstructor()


@dataset_constructor.register('tatsu-lab/alpaca')
def alpaca_preprocessing_function(inp: dict) -> PromptResponseDict:
    """Split out prompt/response from text."""
    try:
        prompt, response = inp['text'].split('### Response:')
        prompt += '### Response:'
    except Exception as e:
        raise UnableToProcessPromptResponseError(inp) from e

    return {'prompt': prompt, 'response': response}


@dataset_constructor.register('HuggingFaceH4/databricks_dolly_15k')
def dolly_preprocessing_function(inp: dict) -> PromptResponseDict:
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
        raise UnableToProcessPromptResponseError(inp) from e
    return {'prompt': prompt, 'response': response}


@dataset_constructor.register('bigscience/P3')
def p3_preprocessing_function(inp: dict) -> PromptResponseDict:
    """Format the already-split example."""
    return {
        'prompt': inp['inputs'] + ':',
        'response': inp['targets'],
    }


# Muennighoff's P3 and flan datasets share a similar convention
@dataset_constructor.register('Muennighoff/P3', 'Muennighoff/flan')
def muennighoff_tokenize_function(inp: dict) -> PromptResponseDict:
    """Format the already-split example."""
    try:
        prompt: str = inp['inputs']
        response: str = inp['targets']
        # Put a space before the response if needed
        transitions = (' ', '\n', '\t')
        if not (
            prompt.endswith(transitions) or response.startswith(transitions)
        ):
            response = ' ' + response
    except Exception as e:
        raise UnableToProcessPromptResponseError(inp) from e
    return {'prompt': prompt, 'response': response}


@dataset_constructor.register('teknium/OpenHermes-2.5')
def shareGPT_format_preprocessor(inp: dict) -> ChatFormattedDict:
    """Convert from ShareGPT format to our chat format."""
    role_map = {
        'human': 'user',
        'gpt': 'assistant',
    }
    try:
        conversation = inp['conversations']
        messages: list[dict[str, str]] = []
        for message in conversation:
            role: str = role_map.get(message['from'], message['from'])
            content: str = message['value']
            messages.append({'role': role, 'content': content})
    except Exception as e:
        raise UnableToProcessPromptResponseError(inp) from e
    return {'messages': messages}


@dataset_constructor.register('math-ai/StackMathQA')
def QA_format_preprocessor(inp: dict) -> ChatFormattedDict:
    """Convert from QA format to our chat format."""
    try:
        Q = inp['Q']
        A = inp['A']
        messages: list[dict[str, str]] = [{
            'role': 'user',
            'content': Q,
        }, {
            'role': 'assistant',
            'content': A,
        }]
    except Exception as e:
        raise UnableToProcessPromptResponseError(inp) from e
    return {'messages': messages}


@dataset_constructor.register('AI-MO/NuminaMath-CoT')
def messages_format_preprocessor(inp: dict) -> ChatFormattedDict:
    """Convert from QA format to our chat format."""
    try:
        messages = inp['messages']
    except Exception as e:
        raise UnableToProcessPromptResponseError(inp) from e
    return {'messages': messages}
