# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import importlib.metadata
from typing import Optional


def clear_entrypoints(entry_point_groups: Optional[list[str]] = None):
    """Clears specified entry point registries.

    Clears all if none specified.
    """
    entry_points = importlib.metadata.entry_points()

    if entry_point_groups is None:
        # Filter entry points to only those that start with 'llmfoundry_'
        entry_point_groups = [group for group in entry_points.groups if group.startswith('llmfoundry_')]

    for group in entry_point_groups:
        if group in entry_points.groups:
            try:
                del importlib.metadata.entry_points()[group]
                print(f"Cleared entry point group: {group}")
            except KeyError:
                print(f"Entry point group {group} not found.")
        else:
            print(f"Entry point group {group} not found in distribution.")

    print("Specified entry point registries have been cleared.")