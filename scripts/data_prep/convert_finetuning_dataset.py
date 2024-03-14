# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import platform
import warnings
from argparse import ArgumentParser, Namespace
from typing import Callable, Dict, Iterable, Optional, Union

import datasets as hf_datasets
import psutil
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from streaming import MDSWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from llmfoundry.data.finetuning.collator import validate_target_settings
from llmfoundry.data.finetuning.tasks import (_get_example_type,
                                              dataset_constructor,
                                              is_valid_ift_example,
                                              tokenize_formatted_example)
from llmfoundry.utils.builders import build_tokenizer

HFDataset = Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]


def parse_args() -> Namespace:
    """Parse commandline arguments."""
    parser = ArgumentParser(description='Convert dataset into MDS format.')
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help=
        'Name of the dataset (e.g., first argument to `datasets.load_dataset`, for jsonl data format, it is `json`)'
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
        '--data_files',
        nargs='+',
        default=[],
        help=
        'Data file for each split. If set, its length should be exact same as len(splits)'
    )
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
             'This is the policy for when to use prompts as training targets. Default "none" means prompts are never used as training targets.'
    )
    parser.add_argument(
        '--target_responses',
        type=str,
        default='last',
        help='Used to determine which samples are valid at max_seq_len. ' +\
             'This is the policy for which responses to treat as training targets. Default "last" means the only the final response (if multi-turn) is used.'
    )
    parser.add_argument(
        '--encoder_decoder',
        action='store_true',
        help='Used to determine which samples are valid at max_seq_len. ' +\
             'Set this flag if the data are intended to be used to train an encoder-decoder model. If so, you must use the default ' +\
            '``target_prompts`` and ``target_responses`` settings of "none" and "last", respectively.'
    )

    parsed = parser.parse_args()

    if os.path.isdir(parsed.out_root) and len(
            set(os.listdir(parsed.out_root)).intersection(set(
                parsed.splits))) > 0:
        raise ValueError(
            f'--out_root={parsed.out_root} contains {os.listdir(parsed.out_root)} which cannot overlap with the requested splits {parsed.splits}.'
        )

    if parsed.tokenizer_kwargs is not None:
        parsed.tokenizer_kwargs = json.loads(parsed.tokenizer_kwargs)
    else:
        parsed.tokenizer_kwargs = {}

    if len(parsed.data_files) > 0 and len(parsed.data_files) != len(
            parsed.splits):
        raise ValueError(
            f'If data_files is set, data_files and splits must have the same length. Got {len(parsed.data_files)=} while {len(parsed.splits)=}'
        )

    return parsed


def build_dataloader(dataset: HFDataset,
                     batch_size: int,
                     num_workers: Optional[int] = None) -> DataLoader:
    if num_workers is None:
        # Multiple workers is only supported on linux machines
        if 'linux' in platform.platform().lower():
            num_workers = max(1, psutil.cpu_count())
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


def get_columns_and_format(dataset: HFDataset, tokenizing: bool,
                           preprocessing_fn: Callable):
    ex = preprocessing_fn(next(iter(dataset)))
    example_type = _get_example_type(ex)
    if tokenizing:
        return {'turns': 'json'}, example_type
    if example_type == 'chat':
        # Chat format
        return {'messages': 'json'}, example_type
    else:
        # Prompt-response format
        return {'prompt': 'str', 'response': 'str'}, example_type


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
            preprocessor=preprocessor_str, dataset_name=args.dataset)
        if preprocessing_fn is None:
            raise ValueError(
                '`args.preprocessor` was not set and no preprocessing function ' +\
                'has been registered for `args.dataset`. If this was intentional ' +\
                '(e.g., because your dataset is already correctly formatted), ' +\
                'include the "--skip-preprocessing" flag to avoid this error.'
            )

    # Make sure the target settings are valid
    validate_target_settings(
        target_prompts=args.target_prompts,
        target_responses=args.target_responses,
        decoder_only_format=not args.encoder_decoder,
    )

    tokenizer = None
    tokenizer_kwargs = args.tokenizer_kwargs
    tokenizer_kwargs.update({'model_max_length': args.max_seq_len})
    if args.tokenizer:
        tokenizer = build_tokenizer(args.tokenizer, tokenizer_kwargs)

    for i, split_name in enumerate(args.splits):
        data_file = None
        if len(args.data_files) > 0:
            data_file = args.data_files[i]
        dataset = hf_datasets.load_dataset(path=args.dataset,
                                           name=args.data_subset,
                                           split=split_name,
                                           data_files=data_file,
                                           streaming=True)
        # Determine the output columns
        columns, example_type = get_columns_and_format(
            dataset=dataset,
            tokenizing=tokenizer is not None,
            preprocessing_fn=preprocessing_fn)
        # Prepare the iterables
        if example_type == 'chat':
            samples = iter(dataset)
        else:
            loader = build_dataloader(dataset=dataset,
                                      batch_size=512,
                                      num_workers=args.num_workers)
            samples = generate_samples(loader)

        # Write samples
        print(f'Converting {split_name} to MDS format...')
        out = os.path.join(args.out_root, split_name)
        if args.local is not None:
            out = (os.path.join(args.local, split_name), out)
            keep_local = True
        else:
            keep_local = False
        with MDSWriter(columns=columns,
                       out=out,
                       compression=args.compression,
                       keep_local=keep_local) as out:
            examples_removed = 0
            for sample in tqdm(samples, desc=split_name):
                formatted_sample = preprocessing_fn(sample)

                # Use the _get_example_type utility to confirm that the formatted sample
                # can be interpreted by the tokenization code
                try:
                    example_type = _get_example_type(formatted_sample)
                except Exception as e:
                    raise ValueError(
                        'Encountered an error when checking example for proper formatting. ' +\
                        f'example={formatted_sample}'
                    ) from e
                if tokenizer is not None:
                    sample = tokenize_formatted_example(formatted_sample,
                                                        tokenizer=tokenizer)
                    if not is_valid_ift_example(
                            args.max_seq_len,
                            target_prompts=args.target_prompts,
                            target_responses=args.target_responses,
                            decoder_only_format=not args.encoder_decoder,
                            example=sample):
                        examples_removed += 1
                        continue

                    sample_to_write = {'turns': []}
                    for turn in sample['turns']:
                        turn_to_write = {}
                        for key in ['input_ids', 'labels']:
                            turn_to_write[key] = list(turn[key])
                        sample_to_write['turns'].append(turn_to_write)
                    out.write(sample_to_write)
                else:
                    if example_type == 'prompt_response':
                        encoded_sample = {
                            key: formatted_sample[key].encode('utf-8')
                            for key in ['prompt', 'response']
                        }
                    else:
                        encoded_sample = formatted_sample
                    out.write(encoded_sample)
        if tokenizer is not None and examples_removed > 0:
            warnings.warn(
                f'Dropped {examples_removed} examples where the prompt was longer than {args.max_seq_len}, '
                +
                'the prompt or response was empty, or the response was all padding tokens.'
            )


if __name__ == '__main__':
    """Example for converting Muennighoff/P3:

    >>> python convert_finetuning_dataset.py \
    >>>    --dataset "Muennighoff/P3" \
    >>>    --splits train validation \
    >>>    --preprocessor llmfoundry.data.finetuning.tasks:p3_preprocessing_function \
    >>>    --out_root s3://<bucket>/muennighoff-p3
    """
    main(parse_args())
