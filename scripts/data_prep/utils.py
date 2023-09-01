# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import platform
from typing import Dict, Iterable, Optional

import psutil
from torch.utils.data import DataLoader, Dataset


def build_dataloader(dataset: Dataset,
                     batch_size: int,
                     num_workers: Optional[int] = None) -> DataLoader:
    if num_workers is None:
        # Multiple workers is only supported on linux machines
        if 'linux' in platform.platform().lower():
            num_workers = max(1, psutil.cpu_count())  # type: ignore
        else:
            num_workers = 0

    # If using multiple workers, configure each worker to prefetch as many samples as it can, up to
    # the aggregate device batch size
    # If not using workers, the torch DataLoader expects the default value for prefetch_factor,
    # which non-intuitively must be 2.
    # If on macOS, PyTorch requires prefetch_factor set to None since num_workers is always zero
    if 'macos' in platform.platform().lower() and num_workers == 0:
        prefetch_factor = None
    else:
        prefetch_factor = max(1, 2 * batch_size //
                              num_workers) if num_workers > 0 else 2
        
    return DataLoader(
        dataset=dataset,
        sampler=None,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )


def generate_samples(
        loader: DataLoader,
        truncate_num_samples: Optional[int] = None
) -> Iterable[Dict[str, bytes]]:
    """Generator over samples of a dataloader.

    Args:
       loader (DataLoader): A dataloader emitting batches like {key: [sample0_bytes, sample1_bytes, sample2_bytes, ...]}
       truncate_num_samples (Optional[int]): An optional # of samples to stop at.

    Yields:
        Sample dicts.
    """
    n_samples = 0
    for batch in loader:
        keys = list(batch.keys())
        current_bs = len(batch[keys[0]])
        for idx in range(current_bs):
            if truncate_num_samples is not None and n_samples == truncate_num_samples:
                return
            n_samples += 1
            yield {k: v[idx] for k, v in batch.items()}
