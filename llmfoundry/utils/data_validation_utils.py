# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import re
import tempfile
from argparse import ArgumentParser, Namespace
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from typing import Callable, Mapping, cast

import torch
import datasets
import numpy as np
import pandas as pd
from composer.utils import (ObjectStore, maybe_create_object_store_from_uri,
                            parse_uri)
from datasets import get_dataset_split_names
from huggingface_hub import dataset_info
from omegaconf import OmegaConf as om
from streaming.base.storage.download import download_file
from streaming.base.storage.upload import CloudUploader

from llmfoundry.data import ConcatTokensDataset
from llmfoundry.utils import build_tokenizer
from composer.utils import dist, get_file, parse_uri

from llmfoundry.data.finetuning.tasks import (DOWNLOADED_FT_DATASETS_DIRPATH,
                                              SUPPORTED_EXTENSIONS,
                                              dataset_constructor)


def get_import_exception_message(package_name: str, extra_deps: str) -> str:
    """Get import exception message.

    Args:
        package_name (str): Package name.

    Returns:
        str: Exception message.
    """
    return f'llm-foundry was installed without {extra_deps} support. ' + \
            f'To use {extra_deps} related packages with llm-foundry, run ' + \
            f'`pip install \'llm-foundry[{extra_deps}]\'`.'


def create_om_cfg(FT_API_args: Namespace):
    task_type = FT_API_args.task_type

    train_data_path = FT_API_args.train_data_path
    split = 'train'

    if is_hf_dataset_path(FT_API_args.train_data_path):
        train_data_path, split = '/'.join(
            FT_API_args.train_data_path.split('/')
            [:2]), FT_API_args.train_data_path.split('/')[-1]

    model = FT_API_args.model
    max_seq_len = FT_API_args.context_length
    detected_cpu_count = os.cpu_count() or 1

    common_args = {
        'drop_last': False,
        'num_workers': detected_cpu_count,
        'prefetch_factor': 2,
        'pin_memory': False,
        'persistent_workers': False,
        'timeout': 0
    }
    if task_type == 'INSTRUCTION_FINETUNE':
        cfg = om.create({
            'dataset': {
                'hf_name': train_data_path,
                'split': split,
                'max_seq_len': max_seq_len,
                'decoder_only_format': True,
                'allow_pad_trimming': False,
                'shuffle': True,
            },
            **common_args
        })

    else:
        cfg = om.create({
            'name': 'finetuning',
            'dataset': {
                'remote': train_data_path,
                'local': train_data_path,
                'split': split,
                'max_seq_len': max_seq_len,
                'decoder_only_format': True,
                'allow_pad_trimming': False,
                'packing_ratio': None,
                'shuffle': True,
            },
            **common_args
        })

    tokenizer = build_tokenizer(
        tokenizer_name=model,
        tokenizer_kwargs={'model_max_length': max_seq_len},
    )

    return cfg, tokenizer


def token_counts_and_validation(FT_API_args):
    from llmfoundry.data.finetuning import build_finetuning_dataloader

    cfg, tokenizer = create_om_cfg(FT_API_args)

    device_batch_size = 1
    dataspec = build_finetuning_dataloader(cfg, tokenizer, device_batch_size)
    dataloader = dataspec.dataloader
    token_counting_func = dataspec.get_num_tokens_in_batch

    total_tokens = []
    for batch in tqdm(dataloader):
        n_batch_tokens = token_counting_func(batch)
        if n_batch_tokens == 0:
            raise ValueError('Empty train sample')
        total_tokens.append(n_batch_tokens)
    return total_tokens


def get_num_samples_in_batch(batch: dict) -> int:
    decoder_only = True

    if not isinstance(batch, Mapping) or ('attention_mask' not in batch and
                                          'input_ids' not in batch):
        raise ValueError(
            'get_tokens_per_batch_func() requires a batch with an attention_mask key or an input_ids key'
        )

    if not decoder_only and 'decoder_attention_mask' not in batch:
        raise ValueError(
            'get_tokens_per_batch_func() for encoder decoder requires a batch with a decoder_attention_mask key'
        )

    # Count number of non padding tokens in batch
    if 'attention_mask' in batch:
        input_ids_tokens = int(sum(batch['attention_mask']))
    else:
        input_ids_tokens = batch['input_ids'].numel()

    # For encoder decoder models only
    decoder_input_ids_tokens = 0
    if not decoder_only:
        decoder_input_ids_tokens = int(
            torch.sum(batch['decoder_attention_mask']).item())

    response_tokens = len(batch['labels']) if 'labels' in batch else 0

    return {
        'ntokens': input_ids_tokens + decoder_input_ids_tokens + response_tokens
    }


def token_counts(FT_API_args):
    from llmfoundry.data.finetuning import build_finetuning_dataloader

    cfg, tokenizer = create_om_cfg(FT_API_args)

    device_batch_size = 1
    dataspec = build_finetuning_dataloader(cfg, tokenizer, device_batch_size)
    dataloader = dataspec.dataloader

    detected_cpu_count = os.cpu_count() or 1
    num_cpus_to_use = max(1, detected_cpu_count)

    token_lens = dataloader.dataset.map(
        get_num_samples_in_batch,
        batched=False,
        num_proc=num_cpus_to_use,
        desc='List of Token length',
    )

    return token_lens


def check_HF_datasets(dataset_names_with_splits: list):
    token = os.environ.get('HUGGING_FACE_HUB_TOKEN')
    for dataset_name_with_split in dataset_names_with_splits:
        dataset_name, split = os.path.split(dataset_name_with_split)
        # make sure we have a dataset and split
        if not dataset_name or not split:
            return False, f"Failed to load Hugging Face dataset {dataset_name_with_split}. Please ensure that you include the split name (e.g. 'mosaicml/dolly_hhrlhf/train')."
        # check user access to the dataset
        try:
            _ = dataset_info(dataset_name)
        except:
            token_warning = ''
            if not token:
                token_warning = ' If this is a private dataset, please set your HUGGING_FACE_HUB_TOKEN using: mcli create secret hf.'
            return False, f"Failed to load Hugging Face dataset {dataset_name_with_split}. Please ensure that the dataset exists and that you have access to it. Remember to include the split name (e.g. 'mosaicml/dolly_hhrlhf/train')." + token_warning
        # check that split exists
        try:
            splits = get_dataset_split_names(dataset_name)
        except:  # error raised in the case of multiple subsets
            return False, f'Failed to load Hugging Face dataset {dataset_name_with_split}. Please make sure that the split is valid and that your dataset does not have subsets.'
        if split not in splits:
            return False, f'Failed to load Hugging Face dataset {dataset_name_with_split}. Split not found.'
    return True, ''


def is_hf_dataset_path(path: str):
    """Check if a given string is a dataset path used by Hugging Face.

    Args:
        path (str): The string to be checked.

    Returns:
        bool: True if the string is a dataset path, False otherwise.
    """
    # Regular expression to match the dataset path pattern
    pattern = r'^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+/?(train|validation|test)?/?$'

    return bool(re.match(pattern, path))


def is_uc_delta_table(name: str):
    """name is in the form of catalog.scheme.tablename.

    Args:
        name (str): a string folder/file/table path
    Return:
        (bool): True if name is valid UC delta table format
    """
    return '.' in name and '/' not in name and '\\' not in name and len(
        name.split('.')) == 3


def integrity_check(out: Union[str, Tuple[str, str]]):
    """Check if the index file has integrity.

       If index is a cloud url, first download it to a temp local file.

    Args:
        out (Union[str, Tuple[str,str]]): MDS dataset path
    """

    def count_shards(mds_root: str):
        n_shard_files = 0
        cu = CloudUploader.get(mds_root, exist_ok=True, keep_local=True)
        for o in cu.list_objects():
            if o.endswith('.mds'):
                n_shard_files += 1
        return n_shard_files

    cu = CloudUploader.get(out, keep_local=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        if cu.remote:
            download_file(os.path.join(cu.remote, 'index.json'),
                          os.path.join(temp_dir, 'index.json'),
                          timeout=60)
            actual_n_shard_files = count_shards(cu.remote)
            local_merged_index_path = os.path.join(temp_dir, 'index.json')
        else:
            local_merged_index_path = os.path.join(cu.local, 'index.json')
            actual_n_shard_files = count_shards(cu.local)

        merged_index = json.load(open(local_merged_index_path, 'r'))
        n_shard_files = len(
            {b['raw_data']['basename'] for b in merged_index['shards']})
        return n_shard_files == actual_n_shard_files


def parse_args(tokenizer,
               concat_tokens,
               output_folder,
               input_folder,
               compression='zstd',
               bos_text='',
               eos_text='',
               no_wrap=False,
               processes=32,
               reprocess=True,
               skip_mdswrite=False) -> Namespace:
    parsed = Namespace(tokenizer=tokenizer,
                       concat_tokens=concat_tokens,
                       output_folder=output_folder,
                       input_folder=input_folder,
                       eos_text=eos_text,
                       bos_text=bos_text,
                       no_wrap=no_wrap,
                       compression=compression,
                       processes=processes,
                       reprocess=reprocess,
                       skip_mdswrite=skip_mdswrite)
    # Make sure we have needed concat options
    if (parsed.concat_tokens is not None and
            isinstance(parsed.concat_tokens, int) and parsed.tokenizer is None):
        parser.error(
            'When setting --concat_tokens, you must specify a --tokenizer')
    # now that we have validated them, change BOS/EOS to strings
    if parsed.bos_text is None:
        parsed.bos_text = ''
    if parsed.eos_text is None:
        parsed.eos_text = ''
    return parsed

def cpt_token_counts(args: Namespace) -> int:
    """Count tokens from the concatenated dataset"""

    from llmfoundry.utils.data_prep_utils import convert_text_to_mds, _args_str
    n_samples = convert_text_to_mds(tokenizer_name=args.tokenizer,
                                    output_folder=args.output_folder,
                                    input_folder=args.input_folder,
                                    concat_tokens=args.concat_tokens,
                                    eos_text=args.eos_text,
                                    bos_text=args.bos_text,
                                    no_wrap=args.no_wrap,
                                    compression=args.compression,
                                    processes=args.processes,
                                    reprocess=True, # overwrite args.reprocess,
                                    args_str=_args_str(args),
                                    skip_mdswrite = True) # overwrite args.skip_mdswrite

    return n_samples


def plot_hist(data, save_plot_path=None):
    """A helper function to draw the frequency of counts of tokens in a dataset"""

    import matplotlib.pyplot as plt
    import pandas as pd

    # Figure and Axis Setup
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # Histogram Plotting
    data.hist(bins=100, edgecolor='black', color='skyblue', alpha=0.7, ax=ax)

    # Aesthetics
    plt.title('Histogram of Token Counts')
    plt.xlabel('Number of Tokens per Sample')
    plt.ylabel('Count of Frequency')

    # Grid and Layout
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()

    # Statistical Information (optional)
    mean_val = data.mean()
    median_val = data.median()
    plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=1)
    plt.axvline(median_val, color='green', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(mean_val * 1.1, max_ylim * 0.9, f'Mean: {mean_val:.2f}')
    plt.text(median_val * 1.1, max_ylim * 0.8, f'Median: {median_val:.2f}')

    if save_plot_path is not None:
        plt.savefig(save_plot_path)

    # Show the Plot
    plt.show()



