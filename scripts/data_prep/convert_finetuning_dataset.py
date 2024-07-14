# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from argparse import ArgumentParser, Namespace
from typing import Union

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict

from llmfoundry.data_prep import convert_finetuning_dataset_from_args

HFDataset = Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]


def parse_args() -> Namespace:
    """Parse commandline arguments."""
    parser = ArgumentParser(description='Convert dataset into MDS format.')
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help=
        'Name of the dataset (e.g., first argument to `datasets.load_dataset`, for jsonl data format, it is `json`)',
    )
    parser.add_argument(
        '--data_subset',
        type=str,
        default=None,
        help='(Optional) subset of data to use.',
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'validation'],
        help='Which splits of the dataset to convert.',
    )
    parser.add_argument('--preprocessor',
                        type=str,
                        default=None,
                        help='Name or import path of function used to preprocess (reformat) the dataset. ' +\
                             'See README for additional details.')
    parser.add_argument(
        '--data_files',
        nargs='+',
        default=[],
        help=
        'Data file for each split. If set, its length should be exact same as len(splits)',
    )
    parser.add_argument(
        '--skip-preprocessing',
        action='store_true',
        help=
        'Whether to skip preprocessing (e.g., if the dataset is already formatted correctly)',
    )
    parser.add_argument(
        '--out_root',
        type=str,
        required=True,
        help=
        'Root path of output directory where MDS shards will be stored. Can be a remote URI.',
    )
    parser.add_argument(
        '--local',
        type=str,
        default=None,
        help=
        '(Optional) root path of local directory if you want to keep a local copy when out_root is remote.',
    )
    parser.add_argument(
        '--compression',
        type=str,
        default=None,
        help='(Optional) name of compression algorithm to use.',
    )
    parser.add_argument('--num_workers', type=int, required=False, default=None)
    parser.add_argument('--tokenizer', type=str, required=False, default=None)
    parser.add_argument('--tokenizer_kwargs', type=str, required=False)

    ##### Used to determine whether an example needs to be filtered out ####
    parser.add_argument('--max_seq_len', type=int, default=2048)
    parser.add_argument(
        '--target_prompts',
        type=str,
        default='none',
        help='Used to determine which samples are valid at max_seq_len. ' +\
             'This is the policy for when to use prompts as training targets. Default "none" means prompts are never used as training targets.',
    )
    parser.add_argument(
        '--target_responses',
        type=str,
        default='last',
        help='Used to determine which samples are valid at max_seq_len. ' +\
             'This is the policy for which responses to treat as training targets. Default "last" means the only the final response (if multi-turn) is used.',
    )
    parser.add_argument(
        '--encoder_decoder',
        action='store_true',
        help='Used to determine which samples are valid at max_seq_len. ' +\
             'Set this flag if the data are intended to be used to train an encoder-decoder model. If so, you must use the default ' +\
            '``target_prompts`` and ``target_responses`` settings of "none" and "last", respectively.',
    )

    parsed = parser.parse_args()
    return parsed


if __name__ == '__main__':
    """Example for converting Muennighoff/P3:

    >>> python convert_finetuning_dataset.py \
    >>>    --dataset "Muennighoff/P3" \
    >>>    --splits train validation \
    >>>    --preprocessor llmfoundry.data.finetuning.tasks:p3_preprocessing_function \
    >>>    --out_root s3://<bucket>/muennighoff-p3
    """
    args = parse_args()
    convert_finetuning_dataset_from_args(
        dataset=args.dataset,
        data_subset=args.data_subset,
        splits=args.splits,
        preprocessor=args.preprocessor,
        data_files=args.data_files,
        skip_preprocessing=args.skip_preprocessing,
        out_root=args.out_root,
        local=args.local,
        compression=args.compression,
        num_workers=args.num_workers,
        tokenizer=args.tokenizer,
        tokenizer_kwargs=args.tokenizer_kwargs,
        max_seq_len=args.max_seq_len,
        target_prompts=args.target_prompts,
        target_responses=args.target_responses,
        encoder_decoder=args.encoder_decoder,
    )
