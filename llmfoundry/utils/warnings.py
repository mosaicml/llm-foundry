# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0


class VersionedDeprecationWarning(DeprecationWarning):
    """A custom deprecation warning class that includes version information.

    Attributes:
        message (str): The deprecation message describing why the feature is deprecated.
        after_version (str): The version after which the feature will be deprecated.
            It will be removed after two releases.

    Example:
        >>> def deprecated_function():
        ...     warnings.warn(
        ...         VersionedDeprecationWarning(
        ...             "Function XYZ is deprecated.",
        ...             after_version="2.0.0"
        ...         )
        ...     )
        ...
        >>> deprecated_function()
        DeprecationWarning: After version 2.0.0: Function XYZ is deprecated.
    """

    def __init__(self, message: str, after_version: str) -> None:
        super().__init__(f'After version {after_version}:' + message)
