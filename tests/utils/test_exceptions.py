# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import json
import pickle
from typing import Dict

from llmfoundry.utils.exceptions import (
    ClusterDoesNotExistError,
    ConsecutiveRepeatedChatRolesError,
    FailedToConnectToDatabricksError,
    FailedToCreateSQLConnectionError,
    IncorrectMessageKeyQuantityError,
    InputFolderMissingDataError,
    InvalidContentTypeError,
    InvalidFileExtensionError,
    InvalidLastChatMessageRoleError,
    InvalidPromptResponseKeysError,
    InvalidPromptTypeError,
    InvalidResponseTypeError,
    InvalidRoleError,
    MisconfiguredHfDatasetError,
    MissingHuggingFaceURLSplitError,
    NotEnoughChatDataError,
    NotEnoughDatasetSamplesError,
    OutputFolderNotEmptyError,
    RunTimeoutError,
    UnableToProcessPromptResponseError,
    UnknownExampleTypeError,
)


def test_exception_serialization():
    exceptions = [
        MissingHuggingFaceURLSplitError(),
        NotEnoughDatasetSamplesError('ds_name', 'split', 1, 2, 3, 4),
        UnknownExampleTypeError('my_keys'),
        NotEnoughChatDataError(),
        ConsecutiveRepeatedChatRolesError('role'),
        InvalidLastChatMessageRoleError('role', {'other_role'}),
        IncorrectMessageKeyQuantityError(['key', 'key2']),
        InvalidRoleError('role', {'other_role'}),
        InvalidContentTypeError(Dict),
        InvalidPromptTypeError(Dict),
        InvalidResponseTypeError(Dict),
        InvalidPromptResponseKeysError({'prompt': 'response'},
                                       {'response': 'prompt'}),
        InvalidFileExtensionError('dsname', ['ext1', 'ext2']),
        UnableToProcessPromptResponseError({'prompt': 'response'}),
        ClusterDoesNotExistError('cluster_name'),
        FailedToCreateSQLConnectionError(),
        FailedToConnectToDatabricksError(),
        InputFolderMissingDataError('folder'),
        OutputFolderNotEmptyError('folder'),
        MisconfiguredHfDatasetError('dataset_name', 'split'),
        RunTimeoutError(100),
    ]

    failed_exceptions = {}

    for exception in exceptions:
        exc_str = str(exception)
        pkl = pickle.dumps(exception)
        try:
            unpickled_exc = pickle.loads(pkl)
            unpickled_exc_str = str(unpickled_exc)
            assert exc_str == unpickled_exc_str
        except Exception as e:
            failed_exceptions[exception.__class__.__name__] = str(e)

    if failed_exceptions:
        raise AssertionError(
            f'Failed to serialize/deserialize the following exceptions: {failed_exceptions.keys()}\n\n'
            + json.dumps(failed_exceptions, indent=2),
        )
