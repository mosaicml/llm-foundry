# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import platform
from glob import glob
from typing import Dict, Iterable, List, Optional

import psutil
from composer.utils import ObjectStore
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
    if 'macos' in platform.platform().lower() or num_workers == 0:
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


def with_id(basename: str, shard_id: int) -> str:
    """Get a new basename with the given shard_id.

    From https://github.com/mosaicml/streaming/blob/main/examples/multiprocess_dataset_conversion.ipynb.

    Args:
        basename (str): Old basename of file.
        shard_id (int): New shard ID.

    Returns:
        str: New basename of file.
    """
    parts = basename.split('.')
    parts[1] = f'{shard_id:05}'
    return '.'.join(parts)


def merge_shard_groups(root: str) -> None:
    """Merge ephemeral sub-datasets created in parallel into one dataset.

    From https://github.com/mosaicml/streaming/blob/main/examples/multiprocess_dataset
    _conversion.ipynb.

    Args:
        root (str): Root directory.
    """
    pattern = os.path.join(root, '*')
    subdirs = sorted(glob(pattern))
    shard_id = 0
    infos = []
    for subdir in subdirs:
        index_filename = os.path.join(subdir, 'index.json')
        with open(index_filename) as index_file:
            obj = json.load(index_file)
        for info in obj['shards']:
            old_basename = info['raw_data']['basename']
            new_basename = with_id(old_basename, shard_id)
            info['raw_data']['basename'] = new_basename

            if info['zip_data'] is not None:
                old_basename = info['zip_data']['basename']
                new_basename = with_id(old_basename, shard_id)
                info['zip_data']['basename'] = new_basename

            old_filename = os.path.join(subdir, old_basename)
            new_filename = os.path.join(root, new_basename)
            os.rename(old_filename, new_filename)

            shard_id += 1
            infos.append(info)

        os.remove(index_filename)
        os.rmdir(subdir)

    index_filename = os.path.join(root, 'index.json')
    obj = {
        'version': 2,
        'shards': infos,
    }
    text = json.dumps(obj, sort_keys=True)
    with open(index_filename, 'w') as out:
        out.write(text)


class DownloadingIterable:

    def __init__(
        self,
        object_names: List[str],
        output_folder: str,
        object_store: Optional[ObjectStore],
    ):
        """Iterable that downloads files from an object store before yielding.

        If object_store is None, input_folder_prefix is treated as a local path.

        Args:
            object_names (List[str]): Names of objects to download
            output_folder (str): Local folder to write downloaded files to
            object_store (Optiona[ObjectStore]): Object store to download from
        """
        self.object_names = object_names
        self.object_store = object_store
        self.output_folder = output_folder

    def __iter__(self):
        for object_name in self.object_names:
            object_name = object_name.strip('/')
            output_filename = os.path.join(self.output_folder, object_name)
            if self.object_store is not None:
                self.object_store.download_object(object_name=object_name,
                                                  filename=output_filename,
                                                  overwrite=True)
            else:
                # Inputs are local so we do not need to download them.
                output_filename = object_name

            with open(output_filename) as _txt_file:
                txt = _txt_file.read()
            yield {'text': txt}
