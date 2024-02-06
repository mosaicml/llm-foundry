# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0


class VersionedDeprecationWarning(DeprecationWarning):
    """A custom deprecation warning class that includes version information.

    Attributes:
        message (str): The deprecation message describing why the feature is deprecated.
        remove_version (str): The version in which the feature will be removed.

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
        DeprecationWarning: Function XYZ is deprecated. It will be removed in version 2.0.0.
    """

    def __init__(self, message: str, remove_version: str) -> None:
        super().__init__(message +
                         f' It will be removed in version {remove_version}.')
