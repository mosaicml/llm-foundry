# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import os
from argparse import ArgumentParser, Namespace
from typing import Dict, Iterable, List, Union

import datasets as hf_datasets
from streaming import MDSWriter
from torch.utils.data import IterableDataset
from tqdm import tqdm

from examples.common.convert_dataset import build_dataloader, generate_samples


def parse_args() -> Namespace:
    """Parse commandline arguments."""
    parser = ArgumentParser(description='Convert dataset into MDS format.')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--data_subset',
                        type=str,
                        default=None,
                        help='E.g. "all" or "en"')
    parser.add_argument('--splits', nargs='+', default=['train', 'validation'])
    parser.add_argument('--columns', nargs='+', required=True)
    parser.add_argument('--out_root', type=str, required=True)
    parser.add_argument('--compression', type=str, default=None)
    parser.add_argument('--local', type=str, default=None)

    parsed = parser.parse_args()

    if os.path.isdir(parsed.out_root) and len(
            set(os.listdir(parsed.out_root)).intersection(set(
                parsed.splits))) > 0:
        raise ValueError(
            f'--out_root={parsed.out_root} contains {os.listdir(parsed.out_root)} which cannot overlap with the requested splits {parsed.splits}.'
        )

    return parsed


class SimpleDataset(IterableDataset):
    """An IterableDataset that returns text samples for MDSWriter.

    Returns dicts of {'key': bytes} for each 'key' in `columns`
    """

    def __init__(self, dataset_name: str, data_subset: Union[str, None],
                 split: str, columns: List[str]):
        self.hf_dataset = hf_datasets.load_dataset(path=dataset_name,
                                                   name=data_subset,
                                                   split=split,
                                                   streaming=True)
        self.columns = columns

    def __iter__(self) -> Iterable[Dict[str, bytes]]:
        for sample in self.hf_dataset:
            # convert to bytes to store in MDS binary format
            yield {key: sample[key].encode('utf-8') for key in self.columns}


def main(args: Namespace) -> None:
    """Main: create a streaming dataset.

    Args:
        args (Namespace): Commandline arguments.
    """
    for split_name in args.splits:
        dataset = SimpleDataset(args.dataset,
                                data_subset=args.data_subset,
                                split=split_name,
                                columns=args.columns)
        loader = build_dataloader(dataset=dataset, batch_size=512)
        samples = generate_samples(loader)

        # Write samples
        print(f'Converting {split_name} to MDS format...')
        out = os.path.join(args.out_root, split_name)
        if args.local is not None:
            out = (os.path.join(args.local, split_name), out)
            keep_local = True
        else:
            keep_local = False
        with MDSWriter(columns={key: 'str' for key in args.columns},
                       out=out,
                       compression=args.compression,
                       keep_local=keep_local) as out:
            for sample in tqdm(samples, desc=split_name):
                out.write(sample)


if __name__ == '__main__':
    """Example for converting Muennighoff/P3:

        >>> python convert_finetuning_dataset.py --dataset "Muennighoff/P3" --splits train validation --columns inputs targets --out_root s3://<bucket>/muennighoff-p3
    """
    main(parse_args())
