import importlib.metadata
from typing import Optional

def clear_entrypoints(entry_point_groups: Optional[list[str]]=None):
    """Clears specified entry point registries. Clears all if none specified."""
    entry_points = importlib.metadata.entry_points()

    if entry_point_groups is None:
        entry_point_groups = entry_points.groups

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
