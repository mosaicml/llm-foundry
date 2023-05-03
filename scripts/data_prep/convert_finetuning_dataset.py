# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import os
import platform
from argparse import ArgumentParser, Namespace
from typing import Dict, Iterable, List, Optional, Union

import datasets as hf_datasets
from streaming import MDSWriter
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from llmfoundry.data.finetuning.tasks import dataset_constructor


def parse_args() -> Namespace:
    """Parse commandline arguments."""
    parser = ArgumentParser(description='Convert dataset into MDS format.')
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help=
        'Name/path of the dataset (e.g., first argument to `datasets.load_dataset`)'
    )
    parser.add_argument('--data_subset',
                        type=str,
                        default=None,
                        help='(Optional) subset of data to use.')
    parser.add_argument('--splits',
                        nargs='+',
                        default=['train', 'validation'],
                        help='Which splits of the dataset to convert.')
    parser.add_argument('--preprocessor',
                        type=str,
                        default=None,
                        help='Name or import path of function used to preprocess (reformat) the dataset. ' +\
                             'See README for additional details.')
    parser.add_argument(
        '--skip-preprocessing',
        action='store_true',
        help=
        'Whether to skip preprocesing (e.g., if the dataset is already formatted correctly)'
    )
    parser.add_argument(
        '--out_root',
        type=str,
        required=True,
        help=
        'Root path of output directory where MDS shards will be stored. Can be a remote URI.'
    )
    parser.add_argument(
        '--local',
        type=str,
        default=None,
        help=
        '(Optional) root path of local directory if you want to keep a local copy when out_root is remote.'
    )
    parser.add_argument('--compression',
                        type=str,
                        default=None,
                        help='(Optional) name of compression algorithm to use.')

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


def build_dataloader(dataset: SimpleDataset, batch_size: int) -> DataLoader:
    # Multiple workers is only supported on linux machines
    if 'linux' in platform.platform().lower():
        num_workers = min(64, dataset.hf_dataset.n_shards)  # type: ignore
    else:
        num_workers = 0

    # If using multiple workers, configure each worker to prefetch as many samples as it can, up to
    # the aggregate device batch size
    # If not using workers, the torch DataLoader expects the default value for prefetch_factor,
    # which non-intuitively must be 2.
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


def main(args: Namespace) -> None:
    """Main: create a streaming dataset.

    Args:
        args (Namespace): Commandline arguments.
    """
    if args.skip_preprocessing:
        preprocessing_fn = lambda x: x  # Just an identity function
    else:
        preprocessor_str = args.preprocessor
        preprocessing_fn = dataset_constructor.get_preprocessing_fn_from_str(
            preprocessor=preprocessor_str,
            dataset_name=args.dataset,
            verbose=True)
        if preprocessing_fn is None:
            raise ValueError(
                '`args.preprocessor` was not set and no preprocessing function ' +\
                'has been registered for `args.dataset`. If this was intentional ' +\
                '(e.g., because your dataset is already correctly formatted), ' +\
                'include the "--skip-preprocessing" flag to avoid this error.'
            )

    columns = ['prompt', 'response']

    for split_name in args.splits:
        dataset = hf_datasets.load_dataset(path=args.dataset,
                                           name=args.data_subset,
                                           split=split_name,
                                           streaming=True)
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
        with MDSWriter(columns={key: 'str' for key in columns},
                       out=out,
                       compression=args.compression,
                       keep_local=keep_local) as out:
            for sample in tqdm(samples, desc=split_name):
                formatted_sample = preprocessing_fn(sample)
                if ('prompt'
                        not in formatted_sample) or ('response'
                                                     not in formatted_sample):
                    raise KeyError(
                        'Unable to tokenize example because it has not been properly formatted. ' +\
                        '"prompt" and "response" are required keys but at least one was missing ' +\
                        f'from {formatted_sample=}.'
                    )
                encoded_sample = {
                    key: formatted_sample[key].encode('utf-8')
                    for key in columns
                }
                out.write(encoded_sample)


if __name__ == '__main__':
    """Example for converting Muennighoff/P3:

    >>> python convert_finetuning_dataset.py \
    >>>    --dataset "Muennighoff/P3" \
    >>>    --splits train validation \
    >>>    --preprocessor llmfoundry.data.finetuning.tasks:p3_preprocessing_function \
    >>>    --out_root s3://<bucket>/muennighoff-p3
    """
    main(parse_args())
