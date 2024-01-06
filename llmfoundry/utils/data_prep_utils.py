# Copyright 2022-2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
from glob import glob
from typing import List, Optional

from composer.utils import ObjectStore


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
            # Default output_filename, used for local paths.
            output_filename = object_name

            # Download objects if remote path.
            if self.object_store is not None:
                output_filename = os.path.join(self.output_folder,
                                               object_name.strip('/'))
                self.object_store.download_object(object_name=object_name,
                                                  filename=output_filename,
                                                  overwrite=True)

            with open(output_filename) as _txt_file:
                txt = _txt_file.read()
            yield {'text': txt}
