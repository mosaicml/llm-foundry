# Copyright 2022 MosaicML Benchmarks authors
# SPDX-License-Identifier: Apache-2.0

"""C4 streaming dataset conversion script."""

import os
import platform
from argparse import ArgumentParser, Namespace
from typing import Dict, Iterable

import datasets as hf_datasets
from streaming import MDSWriter
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from tqdm import tqdm


def parse_args() -> Namespace:
    """Parse commandline arguments."""
    args = ArgumentParser()
    args.add_argument('--out_root', type=str, required=True)
    args.add_argument('--splits', nargs='+', default=['train', 'val'])

    return args.parse_args()


def build_hf_c4_dataset(split: str) -> IterableDataset:
    """Collect the samples for this dataset split.

    Args:
        split (str): Split name.

    Returns:
        An IterableDataset.
    """

    class ShardedC4(IterableDataset):

        def __init__(self):
            self.dataset = hf_datasets.load_dataset(path='c4',
                                                    name='en',
                                                    split=split,
                                                    streaming=True)

        def num_shards(self):
            it = self.dataset._ex_iterable  # type: ignore
            return len(it.kwargs['filepaths'])  # type: ignore

        def __iter__(self):
            worker_info = get_worker_info()
            if worker_info:
                num_workers = worker_info.num_workers
                worker_id = worker_info.id
                kwargs = self.dataset._ex_iterable.kwargs  # type: ignore
                shards = kwargs['filepaths']  # type: ignore
                assert len(shards) % num_workers == 0
                kwargs['filepaths'] = shards[
                    worker_id::num_workers]  # type: ignore  # noqa
            return iter(self.dataset)

    return ShardedC4()


def generate_samples(dataset: IterableDataset) -> Iterable[Dict[str, bytes]]:
    """Generator over each dataset sample.

    Args:
        samples (IterableDataset): An iterable dataset that is multi-worker compatible

    Yields:
        Sample dicts.
    """
    # Multiple workers is only supported on linux machines
    if 'linux' in platform.platform().lower():
        num_workers = min(64, dataset.num_shards())  # type: ignore
    else:
        num_workers = 0
    batch_size = 512
    # If using multiple workers, configure each worker to prefetch as many samples as it can, up to the aggregate device batch size
    # If not using workers, the torch DataLoader expects the default value for prefetch_factor, which non-intuitively must be 2.
    prefetch_factor = max(1, 2 * batch_size //
                          num_workers) if num_workers > 0 else 2

    loader = DataLoader(
        dataset=dataset,
        sampler=None,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
    for batch in loader:
        keys = list(batch.keys())
        current_bs = len(batch[keys[0]])
        for idx in range(current_bs):
            yield {
                key: batch_values[idx].encode('utf-8')
                for key, batch_values in batch.items()
            }


def main(args: Namespace) -> None:
    """Main: create C4 streaming dataset.

    Args:
        args (Namespace): Commandline arguments.
    """
    columns = {'text': 'str', 'timestamp': 'str', 'url': 'str'}

    for (split, split_new_name, expected_num_samples) in [
        ('train', 'train', 364868892),
        ('validation', 'val', 364608),
    ]:
        # Only generate the splits requested
        if split_new_name not in args.splits:
            continue

        # Get samples
        dataset = build_hf_c4_dataset(split=split)
        samples = generate_samples(dataset)

        # Write samples
        with MDSWriter(dirname=os.path.join(args.out_root, split_new_name),
                       columns=columns) as out:
            for sample in tqdm(samples,
                               desc=split_new_name,
                               total=expected_num_samples):
                out.write(sample)


if __name__ == '__main__':
    main(parse_args())
