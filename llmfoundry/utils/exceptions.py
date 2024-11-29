# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Custom exceptions for the LLMFoundry."""
from typing import Any, Literal, Optional, Union

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
    'DatasetTooSmallError',
    'RunTimeoutError',
    'StoragePermissionError',
    'UCNotEnabledError',
    'DeltaTableNotFoundError',
]

ALLOWED_RESPONSE_KEYS = {'response', 'completion'}
ALLOWED_PROMPT_KEYS = {'prompt'}
ALLOWED_MESSAGES_KEYS = {'messages'}

FailureLocation = Union[Literal['TrainDataloader'], Literal['EvalDataloader']]
FailureAttribution = Union[Literal['UserError'], Literal['InternalError'],
                           Literal['NetworkError']]
TrainDataLoaderLocation = 'TrainDataloader'
EvalDataLoaderLocation = 'EvalDataloader'


class BaseContextualError(Exception):
    """Error thrown when an error occurs in the context of a specific task."""

    location: Optional[FailureLocation] = None
    error_attribution: Optional[FailureAttribution] = None

    def __init__(self, message: str, **kwargs: Any) -> None:
        self.error = message
        self.serializable_attributes = []

        for key, value in kwargs.items():
            setattr(self, key, value)
            self.serializable_attributes.append(key)

        super().__init__(message)

    def __reduce__(self):
        """Adjust the reduce behavior for pickling.

        Because we have custom exception subclasses with constructor args, we
        need to adjust the reduce behavior to ensure that the exception can be
        pickled. This allows error propagation across processes in
        multiprocessing.
        """
        if self.__class__ == BaseContextualError:
            raise NotImplementedError(
                'BaseContextualError is a base class and cannot be pickled.',
            )
        tuple_of_args = tuple([
            getattr(self, key) for key in self.serializable_attributes
        ])
        return (self.__class__, tuple_of_args)


class UserError(BaseContextualError):
    """Error thrown when an error is caused by user input."""

    error_attribution = 'UserError'

    def __reduce__(self):
        if self.__class__ == UserError:
            raise NotImplementedError(
                'UserError is a base class and cannot be pickled.',
            )

        return super().__reduce__()


class NetworkError(BaseContextualError):
    """Error thrown when an error is caused by a network issue."""

    error_attribution = 'NetworkError'

    def __reduce__(self):
        if self.__class__ == NetworkError:
            raise NotImplementedError(
                'NetworkError is a base class and cannot be pickled.',
            )

        return super().__reduce__()


class InternalError(BaseContextualError):
    """Error thrown when an error is caused by an internal issue."""

    error_attribution = 'InternalError'

    def __reduce__(self):
        if self.__class__ == InternalError:
            raise NotImplementedError(
                'InternalError is a base class and cannot be pickled.',
            )

        return super().__reduce__()


# Finetuning dataloader exceptions
class MissingHuggingFaceURLSplitError(UserError):
    """Error thrown when there's no split used in HF dataset config."""

    def __init__(self) -> None:
        message = 'When using a HuggingFace dataset from a URL, you must set the ' + \
                    '`split` key in the dataset config.'
        super().__init__(message)


class NotEnoughDatasetSamplesError(UserError):
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
        message = (
            f'Your dataset (name={dataset_name}, split={split}) ' +
            f'has {full_dataset_size} samples, but your minimum batch size ' +
            f'is {minimum_dataset_size} because you are running on {world_size} gpus and '
            +
            f'your per device batch size is {dataloader_batch_size}. Please increase the number '
            + f'of samples in your dataset to at least {minimum_dataset_size}.'
        )
        super().__init__(
            message,
            dataset_name=dataset_name,
            split=split,
            dataloader_batch_size=dataloader_batch_size,
            world_size=world_size,
            full_dataset_size=full_dataset_size,
            minimum_dataset_size=minimum_dataset_size,
        )


## Tasks exceptions
class InvalidExampleTypeError(UserError):
    """Error thrown when a message type is not a `Mapping`."""

    def __init__(self, example_type: str) -> None:
        message = f'Expected example to be a `Mapping`, but found type {example_type}'
        super().__init__(message, example_type=example_type)


class InvalidMessageTypeError(UserError):
    """Error thrown when a message type is not an `Iterable`."""

    def __init__(self, message_type: str) -> None:
        message = f'Expected message to be an `Iterable`, but found type {message_type}'
        super().__init__(message, message_type=message_type)


class UnknownExampleTypeError(UserError):
    """Error thrown when an unknown example type is used in a task."""

    def __init__(self, example_keys: str) -> None:
        message = (
            f'Found keys {example_keys} in dataset. Unknown example type. For prompt and response '
            f'finetuning, the valid prompt keys are {ALLOWED_PROMPT_KEYS} and the valid response keys are '
            f'{ALLOWED_RESPONSE_KEYS}. For chat finetuning, the allowed keys are {ALLOWED_MESSAGES_KEYS}'
        )

        super().__init__(message, example_keys=example_keys)


class NotEnoughChatDataError(UserError):
    """Error thrown when there is not enough chat data to train a model."""

    def __init__(self) -> None:
        message = 'Chat example must have at least two messages'
        super().__init__(message)


class ConsecutiveRepeatedChatRolesError(UserError):
    """Error thrown when there are consecutive repeated chat roles."""

    def __init__(self, repeated_role: str) -> None:
        message = f'Conversation roles must alternate but found {repeated_role} repeated consecutively.'
        super().__init__(message, repeated_role=repeated_role)


class ChatTemplateError(UserError):
    """Error thrown when a chat template fails to process a sample."""

    def __init__(
        self,
        template: str,
        sample: list[dict[str, Any]],
        inner_message: str,
    ) -> None:
        message = f'Failed to process sample {sample} with template {template}. {inner_message}'
        super().__init__(
            message,
            template=template,
            sample=sample,
            inner_message=inner_message,
        )


class InvalidLastChatMessageRoleError(UserError):
    """Error thrown when the last message role in a chat example is invalid."""

    def __init__(self, last_role: str, expected_roles: set[str]) -> None:
        message = f'Invalid last message role: {last_role}. Expected one of: {expected_roles}'
        super().__init__(
            message,
            last_role=last_role,
            expected_roles=expected_roles,
        )


class IncorrectMessageKeyQuantityError(UserError):
    """Error thrown when a message has an incorrect number of keys."""

    def __init__(self, keys: list[str]) -> None:
        message = f'Expected 2 keys in message, but found {len(keys)}'
        super().__init__(message, keys=keys)


class InvalidRoleError(UserError):
    """Error thrown when a role is invalid."""

    def __init__(self, role: str, valid_roles: set[str]) -> None:
        message = f'Expected role to be one of {valid_roles} but found: {role}'
        super().__init__(message, role=role, valid_roles=valid_roles)


class InvalidContentTypeError(UserError):
    """Error thrown when the content type is invalid."""

    def __init__(self, content_type: type) -> None:
        message = f'Expected content to be a string, but found {content_type}'
        super().__init__(message, content_type=content_type)


class InvalidPromptTypeError(UserError):
    """Error thrown when the prompt type is invalid."""

    def __init__(self, prompt_type: type) -> None:
        message = f'Expected prompt to be a string, but found {prompt_type}'
        super().__init__(message, prompt_type=prompt_type)


class InvalidResponseTypeError(UserError):
    """Error thrown when the response type is invalid."""

    def __init__(self, response_type: type) -> None:
        message = f'Expected response to be a string, but found {response_type}'
        super().__init__(message, response_type=response_type)


class InvalidPromptResponseKeysError(UserError):
    """Error thrown when missing expected prompt and response keys."""

    def __init__(self, mapping: dict[str, str], example: dict[str, Any]):
        message = f'Expected {mapping=} to have keys "prompt" and "response".'
        super().__init__(message, mapping=mapping, example=example)


class InvalidFileExtensionError(UserError):
    """Error thrown when a file extension is not a safe extension."""

    def __init__(self, dataset_name: str, valid_extensions: list[str]) -> None:
        message = (
            f'safe_load is set to True. No data files with safe extensions {valid_extensions} '
            + f'found for dataset at local path {dataset_name}.'
        )
        super().__init__(
            message,
            dataset_name=dataset_name,
            valid_extensions=valid_extensions,
        )


class UnableToProcessPromptResponseError(
    UserError,
):
    """Error thrown when a prompt and response cannot be processed."""

    def __init__(self, input: dict) -> None:
        message = f'Unable to extract prompt/response from {input}'
        super().__init__(message, input=input)


## Convert Delta to JSON exceptions
class MalformedUCTableError(UserError):
    """Error thrown when the UC table has the wrong columns."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)

    def __reduce__(self):
        # Return a tuple of class, a tuple of arguments, and optionally state
        return (MalformedUCTableError, (self.message,))

    def __str__(self):
        return self.message


class ClusterDoesNotExistError(UserError):
    """Error thrown when the cluster does not exist."""

    def __init__(self, cluster_id: str) -> None:
        message = f'Cluster with id {cluster_id} does not exist. Check cluster id and try again!'
        super().__init__(message, cluster_id=cluster_id)


class ClusterInvalidAccessMode(UserError):
    """Error thrown when the cluster does not exist."""

    def __init__(self, cluster_id: str, access_mode: str) -> None:
        message = f'The cluster you have provided: {cluster_id} does not have data governance enabled.' + \
                  'Please use a cluster with a data security mode other than NONE.'
        super().__init__(
            message,
            cluster_id=cluster_id,
            access_mode=access_mode,
        )


class FailedToCreateSQLConnectionError(
    NetworkError,
):
    """Error thrown when client can't sql connect to Databricks."""

    def __init__(self) -> None:
        message = 'Failed to create sql connection to db workspace. ' + \
            'To use sql connect, you need to provide http_path and cluster_id!'
        super().__init__(message)


class FailedToConnectToDatabricksError(
    NetworkError,
):
    """Error thrown when the client fails to connect to Databricks."""

    def __init__(self) -> None:
        message = 'Failed to create databricks connection. Check hostname and access token!'
        super().__init__(message)


## Convert Text to MDS exceptions
class InputFolderMissingDataError(UserError):
    """Error thrown when the input folder is missing data."""

    def __init__(self, input_folder: str) -> None:
        message = f'No text files were found at {input_folder}.'
        super().__init__(message, input_folder=input_folder)


class InputFolderNotFound(UserError):
    """Error thrown when the a folder is not found."""

    def __init__(self, folder_that_was_not_found: str) -> None:
        message = f'{folder_that_was_not_found} not found.'
        super().__init__(
            message,
            folder_that_was_not_found=folder_that_was_not_found,
        )


class CannotUnicodeDecodeFile(UserError):
    """Error thrown when the input folder is missing data."""

    def __init__(self, text_file: str) -> None:
        message = f'Text file {text_file} contains chars that cannot be utf-8 decoded. Please remove or replace these chars.'
        super().__init__(message, text_file=text_file)


class OutputFolderNotEmptyError(UserError):
    """Error thrown when the output folder is not empty."""

    def __init__(self, output_folder: str) -> None:
        message = f'{output_folder} is not empty. Please remove or empty it and retry.'
        super().__init__(message, output_folder=output_folder)


class MisconfiguredHfDatasetError(UserError):
    """Error thrown when a HuggingFace dataset is misconfigured."""

    def __init__(self, dataset_name: str, split: Optional[str] = None) -> None:
        message = f'Your dataset (name={dataset_name}, split={split}) is misconfigured. ' + \
            'Please check your dataset format and make sure you can load your dataset locally.' \
            if split is not None else f'Your dataset (name={dataset_name}) is misconfigured. ' + \
            'Please check your dataset format and make sure you can load your dataset locally.'
        super().__init__(message, dataset_name=dataset_name, split=split)


class InvalidDatasetError(UserError):
    """Error thrown when a dataset contains no valid samples for training."""

    def __init__(self, reason: str) -> None:
        message = f'Dataset contains no valid samples for training. {reason}'
        super().__init__(message, reason=reason)


class DatasetTooSmallError(UserError):
    """Error thrown when the dataset is too small to be processed."""

    def __init__(self, reason: str) -> None:
        message = f'Your dataset is too small and produced no complete samples or too few samples. Please provide more data. {reason}'
        super().__init__(message, reason=reason)


class RunTimeoutError(InternalError):
    """Error thrown when a run times out."""

    def __init__(self, timeout: int) -> None:
        message = f'Run timed out after {timeout} seconds.'
        super().__init__(message, timeout=timeout)


class LossSpikeError(UserError):
    """Error thrown if a severe loss spike occurs."""

    def __init__(
        self,
        outlier_multiplier: float,
        running_loss_avg: float,
        outlier_counter: int,
        loss_window: list[float],
    ) -> None:
        message = f'Training stopped due to a loss spike. The training loss was more than {outlier_multiplier} times greater than the running average loss (approx. {running_loss_avg}) over {outlier_counter} consecutive training steps. Please try submitting the run again with a lower learning rate.'

        super().__init__(
            message,
            outlier_multiplier=outlier_multiplier,
            running_loss_avg=running_loss_avg,
            outlier_counter=outlier_counter,
            loss_window=loss_window,
        )


class HighLossError(UserError):
    """Error thrown if training loss plateaus or is unstable at a high level."""

    def __init__(
        self,
        loss_cap: float,
        window_size: int,
        loss_window: list[float],
    ) -> None:
        message = f'Training stopped due to consistently high losses. The training loss exceeded the threshold of {loss_cap} for more than half of the {window_size} most recent training steps. Please try submitting the run again with a lower learning rate.'

        super().__init__(
            message,
            loss_cap=loss_cap,
            window_size=window_size,
            loss_window=loss_window,
        )


class InsufficientPermissionsError(UserError):
    """Error thrown when the user does not have sufficient permissions."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)

    def __reduce__(self):
        # Return a tuple of class, a tuple of arguments, and optionally state
        return (InsufficientPermissionsError, (self.message,))

    def __str__(self):
        return self.message


class FaultyDataPrepCluster(UserError):
    """Error thrown when the user uses faulty data prep cluster."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)

    def __reduce__(self):
        # Return a tuple of class, a tuple of arguments, and optionally state
        return (FaultyDataPrepCluster, (self.message,))

    def __str__(self):
        return self.message


class FinetuningFileNotFoundError(UserError):
    """Error thrown when a file can't be found with any supported extension."""

    def __init__(
        self,
        files_searched: list[str],
        supported_extensions: list[str],
    ) -> None:
        message = (
            f'Could not find a file with any of ' + \
            f'the supported extensions: {supported_extensions}\n' + \
            f'at {files_searched}'
        )
        super().__init__(
            message,
            files_searched=files_searched,
            supported_extensions=supported_extensions,
        )


class InvalidConversationError(UserError):
    """Error thrown when the conversation is invalid."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)

    def __reduce__(self):
        # Return a tuple of class, a tuple of arguments, and optionally state
        return (InvalidConversationError, (self.message,))

    def __str__(self):
        return self.message


class StoragePermissionError(UserError):
    """Error thrown due to invalid permissions accessing blob storage."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)

    def __reduce__(self):
        # Return a tuple of class, a tuple of arguments, and optionally state
        return (StoragePermissionError, (self.message,))

    def __str__(self):
        return self.message


class UCNotEnabledError(UserError):
    """Error thrown when user does not have UC enabled on their cluster."""

    def __init__(self) -> None:
        message = 'Unity Catalog is not enabled on your cluster.'
        super().__init__(message)


class DeltaTableNotFoundError(UserError):
    """Error thrown when the delta table passed in training doesn't exist."""

    def __init__(
        self,
        catalog_name: str,
        volume_name: str,
        table_name: str,
    ) -> None:
        message = f'Your data path {catalog_name}.{volume_name}.{table_name} does not exist. Please double check your delta table name'
        super().__init__(
            message=message,
            catalog_name=catalog_name,
            volume_name=volume_name,
            table_name=table_name,
        )
