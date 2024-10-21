# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import re
import tempfile
from argparse import Namespace
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Mapping, Optional, Tuple, Union

import torch
from composer.utils import (
    maybe_create_object_store_from_uri,
)
from datasets import get_dataset_split_names
from huggingface_hub import dataset_info
from omegaconf import OmegaConf as om
from streaming.base.storage.download import download_file
from streaming.base.storage.upload import CloudUploader
from transformers import AutoTokenizer

from llmfoundry.command_utils.data_prep.convert_text_to_mds import (
    ConcatTokensFromFilesDataset,
    get_object_names,
    get_task_args,
    is_already_processed,
    is_remote_path,
)
from llmfoundry.utils import build_tokenizer
from llmfoundry.utils.data_prep_utils import (
    DownloadingIterable,
    merge_shard_groups,
)
from llmfoundry.utils.exceptions import (
    InputFolderMissingDataError,
    OutputFolderNotEmptyError,
)

log = logging.getLogger(__name__)


def create_om_cfg(FT_API_args: Namespace):
    task_type = FT_API_args.task_type

    train_data_path = FT_API_args.train_data_path
    split = 'train'

    if is_hf_dataset_path(FT_API_args.train_data_path):
        train_data_path, split = '/'.join(
            FT_API_args.train_data_path.split('/')[:2],
        ), FT_API_args.train_data_path.split('/')[-1]

    model = FT_API_args.model
    max_seq_len = FT_API_args.context_length
    detected_cpu_count = os.cpu_count() or 1

    common_args = {
        'drop_last': False,
        'num_workers': detected_cpu_count,
        'prefetch_factor': 2,
        'pin_memory': False,
        'persistent_workers': False,
        'timeout': 0,
    }
    if task_type == 'INSTRUCTION_FINETUNE' or task_type == 'CHAT_COMPLETION':
        cfg = om.create({
            'dataset': {
                'hf_name': train_data_path,
                'split': split,
                'max_seq_len': max_seq_len,
                'decoder_only_format': True,
                'allow_pad_trimming': False,
                'shuffle': True,
            },
            **common_args,
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
            **common_args,
        })

    tokenizer = build_tokenizer(
        tokenizer_name=model,
        tokenizer_kwargs={'model_max_length': max_seq_len},
    )

    return cfg, tokenizer


def token_counts_with_collate(FT_API_args: Namespace):
    from llmfoundry import registry
    from llmfoundry.data.finetuning import build_finetuning_dataloader
    from llmfoundry.utils.registry_utils import construct_from_registry

    cfg, tokenizer = create_om_cfg(FT_API_args)
    dataloader = build_finetuning_dataloader(
        **cfg,
        tokenizer=tokenizer,
        device_batch_size=1,
    ).dataloader

    detected_cpu_count = os.cpu_count() or 1
    num_cpus_to_use = max(1, detected_cpu_count)
    cfg.num_workers = num_cpus_to_use

    dataloader_cfg = {
        'name': 'finetuning',
        'dataset': cfg.dataset,
        'drop_last': cfg.drop_last,
        'num_workers': cfg.num_workers,
        'pin_memory': cfg.pin_memory,
        'prefetch_factor': cfg.prefetch_factor,
        'persistent_workers': cfg.persistent_workers,
        'timeout': cfg.timeout,
    }
    collate_fn, _ = construct_from_registry(
        name='finetuning_collator',
        registry=registry.collators,
        partial_function=False,
        kwargs={
            'dataloader_cfg': dataloader_cfg,
            'tokenizer': tokenizer,
            'dataset_batch_size': 1,
        },
    )

    def mapper(example: dict):
        batch = collate_fn([example])
        return get_num_samples_in_batch(batch)

    token_lens = dataloader.dataset.map( # pyright: ignore
        mapper,
        batched=False,
        num_proc=num_cpus_to_use,
        desc='List of Token length',
    )

    return token_lens


def get_num_samples_in_batch(batch: dict) -> dict[str, int]:
    decoder_only = True

    if not isinstance(batch, Mapping) or (
        'attention_mask' not in batch and 'input_ids' not in batch
    ):
        raise ValueError(
            'get_tokens_per_batch_func() requires a batch with an attention_mask key or an input_ids key',
        )

    if not decoder_only and 'decoder_attention_mask' not in batch:
        raise ValueError(
            'get_tokens_per_batch_func() for encoder decoder requires a batch with a decoder_attention_mask key',
        )

    # Count number of non padding tokens in batch
    if 'attention_mask' in batch:
        input_ids_tokens = int(torch.sum(batch['attention_mask']).item())
    else:
        input_ids_tokens = batch['input_ids'].numel()

    # For encoder decoder models only
    decoder_input_ids_tokens = 0
    if not decoder_only:
        decoder_input_ids_tokens = int(
            torch.sum(batch['decoder_attention_mask']).item(),
        )

    response_tokens = len(batch['labels']) if 'labels' in batch else 0

    return {
        'ntokens':
            input_ids_tokens + decoder_input_ids_tokens + response_tokens,
    }


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
    """Name is in the form of catalog.scheme.tablename.

    Args:
        name (str): a string folder/file/table path
    Return:
        (bool): True if name is valid UC delta table format
    """
    return '.' in name and '/' not in name and '\\' not in name and len(
        name.split('.'),
    ) == 3


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
            download_file(
                os.path.join(cu.remote, 'index.json'),
                os.path.join(temp_dir, 'index.json'),
                timeout=60,
            )
            actual_n_shard_files = count_shards(cu.remote)
            local_merged_index_path = os.path.join(temp_dir, 'index.json')
        else:
            local_merged_index_path = os.path.join(cu.local, 'index.json')
            actual_n_shard_files = count_shards(cu.local)

        merged_index = json.load(open(local_merged_index_path, 'r'))
        n_shard_files = len({
            b['raw_data']['basename'] for b in merged_index['shards']
        })
        return n_shard_files == actual_n_shard_files


def parse_args(
    tokenizer: str,
    concat_tokens: int,
    output_folder: str,
    input_folder: str,
    compression: str = 'zstd',
    bos_text: str = '',
    eos_text: str = '',
    no_wrap: bool = False,
    processes: int = 32,
    reprocess: bool = True,
) -> Namespace:
    parsed = Namespace(
        tokenizer=tokenizer,
        concat_tokens=concat_tokens,
        output_folder=output_folder,
        input_folder=input_folder,
        eos_text=eos_text,
        bos_text=bos_text,
        no_wrap=no_wrap,
        compression=compression,
        processes=processes,
        reprocess=reprocess,
    )
    # Make sure we have needed concat options
    if (
        parsed.concat_tokens is not None and
        isinstance(parsed.concat_tokens, int) and parsed.tokenizer is None
    ):
        raise ValueError(
            'When setting --concat_tokens, you must specify a --tokenizer',
        )
    # now that we have validated them, change BOS/EOS to strings
    if parsed.bos_text is None:
        parsed.bos_text = ''
    if parsed.eos_text is None:
        parsed.eos_text = ''
    return parsed


def download_and_convert_starargs(args: Tuple):
    """Helper function to call download_and_convert with star args.

    This helps us use download_and_convert with mutiprocessing.
    """
    return download_and_convert(*args)


def download_and_convert(
    file_names: list[str],
    output_folder: str,
    input_folder: str,
    tokenizer_name: str,
    concat_tokens: int,
    eos_text: str,
    bos_text: str,
    no_wrap: bool,
    compression: str,
    trust_remote_code: bool,
):
    """Downloads and converts text files to MDS format.

    Args:
        file_names (list[str]): Files to process
        output_folder (str): Folder to write MDS shards to
        input_folder (str): Folder of text files to process
        tokenizer_name (str): Name of tokenizer to use
        concat_tokens (int): Concatenate up to this many tokens
        eos_text (str): Text to append to each example to separate concatenated samples
        bos_text (str): Text to prepend to each example to separate concatenated samples
        no_wrap: (bool): Whether to let text examples wrap across multiple training examples
        compression (str): The compression algorithm to use for MDS writing
        trust_remote_code (bool): If true, allows custom code to be executed to load the tokenizer
    """
    log.info(f'Starting download and conversion for {len(file_names)} files')

    object_store = maybe_create_object_store_from_uri(input_folder)

    # Download file_names
    with tempfile.TemporaryDirectory() as tmp_dir:
        log.info(f'Created temporary directory: {tmp_dir}')
        downloading_iter = DownloadingIterable(
            object_names=file_names,
            output_folder=tmp_dir,
            object_store=object_store,
        )
        log.info(f'Initializing tokenizer: {tokenizer_name}')
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=trust_remote_code,
        )
        tokenizer.model_max_length = 5000000000  # Hack to prevent warnings from HuggingFace

        # Use the ConcatTokensDataset from LLM-foundry to concatenate sequences of tokens up
        # to the maximum sequence length
        dataset = ConcatTokensFromFilesDataset(
            files=downloading_iter,
            max_length=concat_tokens,
            tokenizer=tokenizer,
            eos_text=eos_text,
            bos_text=bos_text,
            no_wrap=no_wrap,
        )

        num_samples = sum([1 for _ in dataset])  # pyright: ignore
    return num_samples


def convert_text_to_mds(
    tokenizer_name: str,
    output_folder: str,
    input_folder: str,
    concat_tokens: int,
    eos_text: str,
    bos_text: str,
    no_wrap: bool,
    compression: str,
    processes: int,
    args_str: str,
    reprocess: bool,
    trust_remote_code: bool,
):
    """Convert a folder of text files to MDS format.

    Args:
        tokenizer_name (str): Name of tokenizer to use
        output_folder (str): Folder to write MDS shards to
        input_folder (str): Folder of text files to process
        concat_tokens (int): Concatenate up to this many tokens
        eos_text (str): Text to append to each example to separate concatenated samples
        bos_text (str): Text to prepend to each example to separate concatenated samples
        no_wrap: (bool): Whether to let text examples wrap across multiple training examples
        compression (str): The compression algorithm to use for MDS writing
        processes (int): The number of processes to use.
        args_str (str): String representation of the arguments
        reprocess (bool): Whether to always reprocess the given folder of text files
        trust_remote_code (bool): If true, allows custom code to be executed to load the tokenizer
    """
    # Load the tokenizer once on the main process so that the files are cached to avoid race conditions
    # in the Hugging Face load code
    AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=trust_remote_code,
    )

    is_remote_output = is_remote_path(output_folder)
    log.info(f'Output is remote: {is_remote_output}')

    object_names = get_object_names(input_folder)
    if len(object_names) == 0:
        log.error(f'No text files found in input folder: {input_folder}')
        raise InputFolderMissingDataError(input_folder)

    # Check if the text files in the bucket have already been processed.
    if not reprocess and is_already_processed(
        output_folder,
        args_str,
        object_names,
    ):
        log.info(
            f'Input folder {input_folder} is already processed at {output_folder} and '
            +
            'reprocess is set to False. Set reprocess to True if you would like to force reprocessing.',
        )
        return

    # Use a temporary local directory if the output is remote and there are more than 1 processes
    local_output_folder = tempfile.TemporaryDirectory(
    ).name if is_remote_output else output_folder
    log.info(f'Using local output folder: {local_output_folder}')

    if os.path.isdir(output_folder) and len(os.listdir(output_folder)) > 0:
        log.error(f'Output folder is not empty: {output_folder}')
        raise OutputFolderNotEmptyError(output_folder)

    if processes > 1:
        log.info(f'Using multiprocessing with {processes} processes')
        # Download and convert the text files in parallel
        args = get_task_args(
            object_names,
            local_output_folder,
            input_folder,
            processes,
            tokenizer_name,
            concat_tokens,
            eos_text,
            bos_text,
            no_wrap,
            compression,
            trust_remote_code,
        )
        with ProcessPoolExecutor(max_workers=processes) as executor:
            pool = list(executor.map(download_and_convert_starargs, args))
            total_tokens = sum(pool)

        log.info('Merging MDS shards from each process')
        # Merge the mds shards from each of the processes into a single folder
        merge_shard_groups(local_output_folder)
    else:
        log.info('Using single process for download and conversion')
        total_tokens = download_and_convert(
            object_names,
            local_output_folder,
            input_folder,
            tokenizer_name,
            concat_tokens,
            eos_text,
            bos_text,
            no_wrap,
            compression,
            trust_remote_code,
        )

    return total_tokens


def plot_hist(data: Any, save_plot_path: Optional[bool] = None):
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
