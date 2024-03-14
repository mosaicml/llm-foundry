# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import re
import tempfile
from argparse import Namespace
from typing import Mapping, Optional, Tuple, Union
from urllib.parse import urlparse

import pandas as pd
import torch
from datasets import get_dataset_split_names
from huggingface_hub import dataset_info
from omegaconf import OmegaConf as om
from streaming.base.storage.download import download_file
from streaming.base.storage.upload import CloudUploader
from tqdm import tqdm

from llmfoundry.data.finetuning import build_finetuning_dataloader
from llmfoundry.utils import build_tokenizer


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


def token_counts_with_collate(FT_API_args):
    from llmfoundry.data.finetuning import build_finetuning_dataloader, _build_collate_fn

    cfg, tokenizer = create_om_cfg(FT_API_args)
    detected_cpu_count = os.cpu_count() or 1
    num_cpus_to_use = max(1, detected_cpu_count)
    cfg.num_workers = num_cpus_to_use

    device_batch_size = 1
    dataspec = build_finetuning_dataloader(cfg, tokenizer, device_batch_size)
    dataloader = dataspec.dataloader

    collate_fn, dataloader_batch_size = _build_collate_fn(
        cfg, tokenizer, device_batch_size)

    def mapper(example: dict):
        batch = collate_fn([example])
        return get_num_samples_in_batch(batch)

    token_lens = dataloader.dataset.map(
        mapper,
        batched=False,
        num_proc=num_cpus_to_use,
        desc='List of Token length',
    )

    return token_lens


def get_num_samples_in_batch(batch: dict) -> Mapping:
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


def token_counts(FT_API_args: Namespace):
    cfg, tokenizer = create_om_cfg(FT_API_args)

    device_batch_size = 1
    dataspec = build_finetuning_dataloader(cfg, tokenizer, device_batch_size)
    dataloader = dataspec.dataloader

    detected_cpu_count = os.cpu_count() or 1
    num_cpus_to_use = max(1, detected_cpu_count)

    token_lens = dataloader.dataset.map(  # pyright: ignore
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


def parse_hf_dataset_url(url: str) -> Tuple[str, str]:
    """Parses a Hugging Face dataset URL to extract the prefix and submixes.

    This function assumes a specific URL format used by Hugging Face datasets.
    It splits the URL path to extract the prefix (if any) and the submixes name.
    The expected URL format is 'https://huggingface.co/datasets/{prefix}/{submixes}'.
    If the URL does not include a prefix, it defaults to an empty string.

    Args:
        url (str): The Hugging Face dataset URL to be parsed.

    Returns:
        tuple: A tuple containing two elements:
               - prefix (str): The extracted prefix from the URL. If no prefix is present, returns an empty string.
               - submixes (str): The name of the submixes extracted from the URL.

    Raises:
        ValueError: If the URL does not conform to the expected format.

    Examples:
    >>> parse_hf_dataset_url("https://huggingface.co/datasets/test_prefix/test_submix")
    ('test_prefix', 'test_submix')

    >>> parse_hf_dataset_url("https://huggingface.co/datasets/test_submix")
    ('', 'test_submix')
    """
    parsed_url = urlparse(url.rstrip('/'))
    path_parts = parsed_url.path.split('/')

    # Assuming the URL format is: https://huggingface.co/datasets/{prefix}/{submixes}
    if len(path_parts) >= 3 and path_parts[1] == 'datasets':
        prefix = '' if len(path_parts) == 3 else path_parts[2]
        submixes = path_parts[2] if len(path_parts) == 3 else path_parts[3]
        return prefix, submixes
    else:
        raise ValueError('Invalid Hugging Face dataset URL format')


def is_hf_dataset_path(path: str,
                       check_split: bool = False) -> Tuple[bool, bool]:
    """Check if a given string is a dataset path used by Hugging Face.

    Args:
        path (str): The string to be checked.
        check_split (bool): Whether insist path ends with a split component

    Returns:
        (bool,bool): First indicates if it is a dataset id, second indicates if it is a dataset repo id
    """
    if check_split:
        pattern = r'^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+/(train|validation|test)/?$'
    else:
        pattern = r'^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+(/train|/validation|/test)?/?$'

    is_dataset_id = bool(re.match(pattern, path))

    try:
        _, _ = parse_hf_dataset_url(path)
        is_repo_id = True
    except ValueError as _:
        is_repo_id = False

    return is_dataset_id, is_repo_id


def is_uc_delta_table(name: str):
    """Name is in the form of catalog.scheme.tablename.

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


def parse_args(tokenizer: str,
               concat_tokens: int,
               output_folder: str,
               input_folder: str,
               compression: str = 'zstd',
               bos_text: str = '',
               eos_text: str = '',
               no_wrap: bool = False,
               processes: int = 32,
               reprocess: bool = True,
               skip_mdswrite: bool = False) -> Namespace:
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
        raise ValueError(
            'When setting --concat_tokens, you must specify a --tokenizer')
    # now that we have validated them, change BOS/EOS to strings
    if parsed.bos_text is None:
        parsed.bos_text = ''
    if parsed.eos_text is None:
        parsed.eos_text = ''
    return parsed


def cpt_token_counts(args: Namespace) -> int:
    """Wrapper of convert_text_to_mds.

    Count tokens from the concatenated dataset. Skip MDS write because it
    creates some issues in spark environment.
    """
    from llmfoundry.utils.data_prep_utils import convert_text_to_mds

    def _args_str(original_args: Namespace) -> str:
        """Create a string from the args to determine whether to reprocess.

        Args:
            original_args (Namespace): Arguments to main function.
        """
        # Take the arguments that influence the final result.
        # reprocess and max_mds_writer_workers are not taken.
        args = Namespace(
            tokenizer_name=original_args.tokenizer,
            output_folder=original_args.output_folder,
            input_folder=original_args.input_folder,
            concat_tokens=original_args.concat_tokens,
            eos_text=original_args.eos_text,
            bos_text=original_args.bos_text,
            no_wrap=original_args.no_wrap,
            compression=original_args.compression,
            processes=original_args.processes,
            skip_mdswrite=original_args.skip_mdswrite,
        )

        return str(args)

    n_samples = convert_text_to_mds(
        tokenizer_name=args.tokenizer,
        output_folder=args.output_folder,
        input_folder=args.input_folder,
        concat_tokens=args.concat_tokens,
        eos_text=args.eos_text,
        bos_text=args.bos_text,
        no_wrap=args.no_wrap,
        compression=args.compression,
        processes=args.processes,
        reprocess=True,  # overwrite args.reprocess,
        args_str=_args_str(args),
        skip_mdswrite=True)  # overwrite args.skip_mdswrite

    return n_samples


def plot_hist(data: pd.DataFrame, save_plot_path: Optional[str] = None):
    """Helper function draw frequency of token counts in a dataset."""
    import matplotlib.pyplot as plt

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
    _, max_ylim = plt.ylim()
    plt.text(mean_val * 1.1, max_ylim * 0.9, f'Mean: {mean_val:.2f}')
    plt.text(median_val * 1.1, max_ylim * 0.8, f'Median: {median_val:.2f}')

    if save_plot_path is not None:
        plt.savefig(save_plot_path)

    # Show the Plot
    plt.show()
