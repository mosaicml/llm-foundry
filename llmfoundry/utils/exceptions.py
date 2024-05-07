# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Custom exceptions for the LLMFoundry."""
from collections.abc import Mapping
from typing import Any, Dict, List, Literal, Optional

__all__ = [
    'ALLOWED_RESPONSE_KEYS',
    'ALLOWED_PROMPT_KEYS',
    'ALLOWED_MESSAGES_KEYS',
    'MissingHuggingFaceURLSplitError',
    'NotEnoughDatasetSamplesError',
    'UnknownExampleTypeError',
    'NotEnoughChatDataError',
    'ConsecutiveRepeatedChatRolesError',
    'InvalidLastChatMessageRoleError',
    'IncorrectMessageKeyQuantityError',
    'InvalidRoleError',
    'InvalidContentTypeError',
    'InvalidPromptTypeError',
    'InvalidResponseTypeError',
    'InvalidPromptResponseKeysError',
    'InvalidFileExtensionError',
    'UnableToProcessPromptResponseError',
    'ClusterDoesNotExistError',
    'FailedToCreateSQLConnectionError',
    'FailedToConnectToDatabricksError',
    'InputFolderMissingDataError',
    'OutputFolderNotEmptyError',
    'MisconfiguredHfDatasetError',
]

ALLOWED_RESPONSE_KEYS = {'response', 'completion'}
ALLOWED_PROMPT_KEYS = {'prompt'}
ALLOWED_MESSAGES_KEYS = {'messages'}

ErrorContext = Literal['TrainContext'] | Literal['EvalContext']


class ContextualError(BaseException):
    """Error thrown when an error occurs in the context of a specific task."""

    context: Optional[ErrorContext] = None


# Finetuning dataloader exceptions
class MissingHuggingFaceURLSplitError(ValueError, ContextualError):
    """Error thrown when there's no split used in HF dataset config."""

    def __init__(self) -> None:
        message = 'When using a HuggingFace dataset from a URL, you must set the ' + \
                    '`split` key in the dataset config.'
        super().__init__(message)


class NotEnoughDatasetSamplesError(ValueError, ContextualError):
    """Error thrown when there is not enough data to train a model."""

    def __init__(
        self,
        dataset_name: str,
        split: str,
        dataloader_batch_size: int,
        world_size: int,
        full_dataset_size: int,
        minimum_dataset_size: int,
    ) -> None:
        self.dataset_name = dataset_name
        self.split = split
        self.dataloader_batch_size = dataloader_batch_size
        self.world_size = world_size
        self.full_dataset_size = full_dataset_size
        self.minimum_dataset_size = minimum_dataset_size
        message = (
            f'Your dataset (name={dataset_name}, split={split}) ' +
            f'has {full_dataset_size} samples, but your minimum batch size ' +
            f'is {minimum_dataset_size} because you are running on {world_size} gpus and '
            +
            f'your per device batch size is {dataloader_batch_size}. Please increase the number '
            + f'of samples in your dataset to at least {minimum_dataset_size}.'
        )
        super().__init__(message)


## Tasks exceptions
class UnknownExampleTypeError(KeyError, ContextualError):
    """Error thrown when an unknown example type is used in a task."""

    def __init__(self, example: Mapping) -> None:
        self.example = example
        message = (
            f'Found keys {example.keys()} in dataset. Unknown example type. For prompt and response '
            f'finetuning, the valid prompt keys are {ALLOWED_PROMPT_KEYS} and the valid response keys are '
            f'{ALLOWED_RESPONSE_KEYS}. For chat finetuning, the allowed keys are {ALLOWED_MESSAGES_KEYS}'
        )
        super().__init__(message)


class NotEnoughChatDataError(ValueError, ContextualError):
    """Error thrown when there is not enough chat data to train a model."""

    def __init__(self) -> None:
        message = 'Chat example must have at least two messages'
        super().__init__(message)


class ConsecutiveRepeatedChatRolesError(ValueError, ContextualError):
    """Error thrown when there are consecutive repeated chat roles."""

    def __init__(self, repeated_role: str) -> None:
        self.repeated_role = repeated_role
        message = f'Conversation roles must alternate but found {repeated_role} repeated consecutively.'
        super().__init__(message)


class InvalidLastChatMessageRoleError(ValueError, ContextualError):
    """Error thrown when the last message role in a chat example is invalid."""

    def __init__(self, last_role: str, expected_roles: set[str]) -> None:
        self.last_role = last_role
        self.expected_roles = expected_roles
        message = f'Invalid last message role: {last_role}. Expected one of: {expected_roles}'
        super().__init__(message)


class IncorrectMessageKeyQuantityError(ValueError, ContextualError):
    """Error thrown when a message has an incorrect number of keys."""

    def __init__(self, keys: List[str]) -> None:
        self.keys = keys
        message = f'Expected 2 keys in message, but found {len(keys)}'
        super().__init__(message)


class InvalidRoleError(ValueError, ContextualError):
    """Error thrown when a role is invalid."""

    def __init__(self, role: str, valid_roles: set[str]) -> None:
        self.role = role
        self.valid_roles = valid_roles
        message = f'Expected role to be one of {valid_roles} but found: {role}'
        super().__init__(message)


class InvalidContentTypeError(TypeError, ContextualError):
    """Error thrown when the content type is invalid."""

    def __init__(self, content_type: type) -> None:
        self.content_type = content_type
        message = f'Expected content to be a string, but found {content_type}'
        super().__init__(message)


class InvalidPromptTypeError(TypeError, ContextualError):
    """Error thrown when the prompt type is invalid."""

    def __init__(self, prompt_type: type) -> None:
        self.prompt_type = prompt_type
        message = f'Expected prompt to be a string, but found {prompt_type}'
        super().__init__(message)


class InvalidResponseTypeError(TypeError, ContextualError):
    """Error thrown when the response type is invalid."""

    def __init__(self, response_type: type) -> None:
        self.response_type = response_type
        message = f'Expected response to be a string, but found {response_type}'
        super().__init__(message)


class InvalidPromptResponseKeysError(ValueError, ContextualError):
    """Error thrown when missing expected prompt and response keys."""

    def __init__(self, mapping: Dict[str, str], example: Dict[str, Any]):
        self.example = example
        message = f'Expected {mapping=} to have keys "prompt" and "response".'
        super().__init__(message)


class InvalidFileExtensionError(FileNotFoundError, ContextualError):
    """Error thrown when a file extension is not a safe extension."""

    def __init__(self, dataset_name: str, valid_extensions: List[str]) -> None:
        self.dataset_name = dataset_name
        self.valid_extensions = valid_extensions
        message = (
            f'safe_load is set to True. No data files with safe extensions {valid_extensions} '
            + f'found for dataset at local path {dataset_name}.'
        )
        super().__init__(message)


class UnableToProcessPromptResponseError(ValueError, ContextualError):
    """Error thrown when a prompt and response cannot be processed."""

    def __init__(self, input: Dict) -> None:
        self.input = input
        message = f'Unable to extract prompt/response from {input}'
        super().__init__(message)


## Convert Delta to JSON exceptions
class ClusterDoesNotExistError(ValueError, ContextualError):
    """Error thrown when the cluster does not exist."""

    def __init__(self, cluster_id: str) -> None:
        self.cluster_id = cluster_id
        message = f'Cluster with id {cluster_id} does not exist. Check cluster id and try again!'
        super().__init__(message)


class FailedToCreateSQLConnectionError(RuntimeError, ContextualError):
    """Error thrown when client can't sql connect to Databricks."""

    def __init__(self) -> None:
        message = 'Failed to create sql connection to db workspace. To use sql connect, you need to provide http_path and cluster_id!'
        super().__init__(message)


class FailedToConnectToDatabricksError(RuntimeError, ContextualError):
    """Error thrown when the client fails to connect to Databricks."""

    def __init__(self) -> None:
        message = 'Failed to create databricks connection. Check hostname and access token!'
        super().__init__(message)


## Convert Text to MDS exceptions
class InputFolderMissingDataError(ValueError, ContextualError):
    """Error thrown when the input folder is missing data."""

    def __init__(self, input_folder: str) -> None:
        self.input_folder = input_folder
        message = f'No text files were found at {input_folder}.'
        super().__init__(message)


class OutputFolderNotEmptyError(FileExistsError, ContextualError):
    """Error thrown when the output folder is not empty."""

    def __init__(self, output_folder: str) -> None:
        self.output_folder = output_folder
        message = f'{output_folder} is not empty. Please remove or empty it and retry.'
        super().__init__(message)


class MisconfiguredHfDatasetError(ValueError, ContextualError):
    """Error thrown when a HuggingFace dataset is misconfigured."""

    def __init__(self, dataset_name: str, split: str) -> None:
        self.dataset_name = dataset_name
        self.split = split
        message = f'Your dataset (name={dataset_name}, split={split}) is misconfigured. Please check your dataset format and make sure you can load your dataset locally.'
        super().__init__(message)
