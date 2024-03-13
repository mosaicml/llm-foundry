# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import re
import tempfile
from argparse import ArgumentParser, Namespace
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

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


from typing import Any, Callable, Dict, List, Mapping, Optional, Union, cast

import torch


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


import logging
import math
import os
import tempfile
from argparse import ArgumentParser, Namespace
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from typing import Iterable, List, Tuple, cast

import datasets as hf_datasets
import psutil
from composer.utils import (ObjectStore, maybe_create_object_store_from_uri,
                            parse_uri)
from streaming import MDSWriter
from tqdm import tqdm
from transformers import AutoTokenizer

from llmfoundry.data import ConcatTokensDataset
from llmfoundry.utils.data_prep_utils import (DownloadingIterable,
                                              merge_shard_groups)

log = logging.getLogger(__name__)
DONE_FILENAME = '.text_to_mds_conversion_done'


def parse_args(tokenizer,
               concat_tokens,
               output_folder,
               input_folder,
               compression='zstd',
               bos_text='',
               eos_text='',
               no_wrap=False,
               processes=32,
               reprocess=True) -> Namespace:
    parsed = Namespace(tokenizer=tokenizer,
                       concat_tokens=concat_tokens,
                       output_folder=output_folder,
                       input_folder=input_folder,
                       eos_text=eos_text,
                       bos_text=bos_text,
                       no_wrap=no_wrap,
                       compression=compression,
                       processes=processes,
                       reprocess=reprocess)
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


def original_parse_args() -> Namespace:
    """Parse commandline arguments."""
    parser = ArgumentParser(
        description=
        'Convert text files into MDS format, optionally concatenating and tokenizing',
    )
    parser.add_argument(
        '--output_folder',
        type=str,
        required=True,
        help='The folder to write output to',
    )
    parser.add_argument(
        '--input_folder',
        type=str,
        required=True,
        help='The folder with text files to convert to mds',
    )
    parser.add_argument(
        '--compression',
        type=str,
        default='zstd',
        help='The compression algorithm to use for MDS writing',
    )

    parser.add_argument(
        '--concat_tokens',
        type=int,
        help='Convert text to tokens and concatenate up to this many tokens',
    )

    parser.add_argument(
        '--tokenizer',
        type=str,
        help='The name of the tokenizer to use',
    )
    parser.add_argument(
        '--bos_text',
        type=str,
        required=False,
        default=None,
        help=
        'The text to prepend to each example to separate concatenated examples',
    )
    parser.add_argument(
        '--eos_text',
        type=str,
        required=False,
        default=None,
        help=
        'The text to append to each example to separate concatenated examples',
    )
    parser.add_argument(
        '--no_wrap',
        default=False,
        action='store_true',
        help=
        'Whether to let text examples wrap across multiple training examples',
    )
    parser.add_argument(
        '--processes',
        type=int,
        required=False,
        default=min(max(psutil.cpu_count() - 2, 1), 32),
        help=
        'The number of processes to use to download and convert the dataset',
    )
    parser.add_argument(
        '--reprocess',
        type=bool,
        required=False,
        default=False,
        help='If true, reprocess the input_folder to mds format. Otherwise, ' +
        'only reprocess upon changes to the input folder or dataset creation parameters.',
    )

    parsed = parser.parse_args()

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


def get_object_names(input_folder: str) -> List[str]:
    """Get object names from a local or remote folder.

    Args:
        input_folder (str): local or remote folder path.
    """
    object_store = maybe_create_object_store_from_uri(input_folder)
    if object_store is not None:
        _, _, folder_prefix = parse_uri(input_folder)
        names = [
            name for name in object_store.list_objects(folder_prefix)
            if name.endswith('.txt')
        ]
    else:
        # input_folder is a local folder
        names = [
            text_file for dirpath, _, _ in os.walk(input_folder)
            for text_file in glob(os.path.join(dirpath, '*.txt'))
        ]
    # return names, sizes
    log.info(f'Found {len(names)} text files at {input_folder}')

    return names


def get_task_args(
    object_names: List[str],
    output_root: str,
    input_folder: str,
    n_groups: int,
    tokenizer_name: str,
    concat_tokens: int,
    eos_text: str,
    bos_text: str,
    no_wrap: bool,
    compression: str,
) -> Iterable:
    """Get download_and_convert arguments split across n_groups.

    Each group handles a portion of object_names.

    Args:
        object_names (List[str]): Names of objects to process
        output_root (str): Folder to write MDS shards to
        input_folder (str): Folder of text files to process
        n_groups (int): Number of groups to split the object names into
        tokenizer_name (str): Name of tokenizer to use
        concat_tokens (int): Concantenate up to this many tokens
        eos_text (str): Textend to append to each example to separate concatenated samples
        bos_text (str): Text to prepend to each example to separate concatenated samples
        no_wrap: (bool): Whether to let text examples wrap across multiple training examples
        compression (str): The compression algorithm to use for MDS writing
    """
    num_objects = len(object_names)
    objs_per_group = math.ceil(num_objects / n_groups)
    for group, i in enumerate(range(0, num_objects, objs_per_group)):
        output_subdir = os.path.join(output_root, str(group))
        yield (
            object_names[i:min(i + objs_per_group, num_objects)],
            output_subdir,
            input_folder,
            tokenizer_name,
            concat_tokens,
            eos_text,
            bos_text,
            no_wrap,
            compression,
        )


def download_and_convert_starargs(args: Tuple):
    """Helper function to call download_and_convert with star args.

    This helps us use download_and_convert with mutiprocessing.
    """
    return download_and_convert(*args)


def download_and_convert(
    file_names: List[str],
    output_folder: str,
    input_folder: str,
    tokenizer_name: str,
    concat_tokens: int,
    eos_text: str,
    bos_text: str,
    no_wrap: bool,
    compression: str,
):
    """Downloads and converts text fies to MDS format.

    Args:
        file_names (List[str]): Files to process
        output_folder (str): Folder to write MDS shards to
        input_folder (str): Folder of text files to process
        tokenizer_name (str): Name of tokenizer to use
        concat_tokens (int): Concantenate up to this many tokens
        eos_text (str): Textend to append to each example to separate concatenated samples
        bos_text (str): Text to prepend to each example to separate concatenated samples
        no_wrap: (bool): Whether to let text examples wrap across multiple training examples
        compression (str): The compression algorithm to use for MDS writing
    Returns:
        (int): token count of the current group
    """
    object_store = maybe_create_object_store_from_uri(input_folder)

    # Download file_names
    with tempfile.TemporaryDirectory() as tmp_dir:
        downloading_iter = DownloadingIterable(object_names=file_names,
                                               output_folder=tmp_dir,
                                               object_store=object_store)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenizer.model_max_length = 5000000000  # Hack to prevent warnings from HuggingFace

        # Use the ConcatTokensDataset from LLM-foundry to concatenate sequences of tokens up
        # to the maximum sequence length
        dataset = ConcatTokensDataset(
            hf_dataset=downloading_iter,
            max_length=concat_tokens,
            tokenizer=tokenizer,
            eos_text=eos_text,
            bos_text=bos_text,
            no_wrap=no_wrap,
        )

        token_count = sum([ 1 for _ in dataset])

        # columns = {'tokens': 'bytes'}

        # log.info('Converting to MDS format...')
        # with MDSWriter(out=output_folder,
        #                columns=columns,
        #                compression=compression) as out:
        #     for sample in tqdm(dataset):
        #         out.write(sample)

    return token_count


def is_remote_path(path: str) -> bool:
    """Checks whether a path is a remote path.

    Args:
        path (str): path to check
    """
    backend, _, _ = parse_uri(path)
    return backend != ''


def is_already_processed(output_root: str, args_str: str,
                         object_names: List[str]) -> bool:
    """Determines whether a group of text files has already been processed.

    Checks the done fie at output root to determine this.

    Args:
        output_root (str): Output folder where a done file may exist
        args_str (str): String representation of the arguments
        object_names (List[str]): Names of objects to convert to MDS format
    """
    # Retrieve the done file contents
    output_object_store = maybe_create_object_store_from_uri(output_root)
    if output_object_store is not None:
        # Download and read the done file from the remote object store
        _, _, output_folder_prefix = parse_uri(output_root)
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                done_file = os.path.join(tmp_dir, DONE_FILENAME)
                output_object_store.download_object(
                    os.path.join(output_folder_prefix, DONE_FILENAME),
                    done_file)
                with open(done_file) as df:
                    done_file_contents = df.read().splitlines()
        except FileNotFoundError:
            return False
    else:
        # Read the local done file
        done_file = os.path.join(output_root, DONE_FILENAME)
        if not os.path.isfile(done_file):
            return False
        with open(done_file) as df:
            done_file_contents = df.read().splitlines()
    # Compare the arguments
    prev_args_str = done_file_contents[0]
    if prev_args_str != args_str:
        return False

    # Compare file names
    prev_names = done_file_contents[1:]
    if len(prev_names) != len(object_names):
        return False
    for idx, prev_name in enumerate(prev_names):
        if object_names[idx] != prev_name:
            return False
    return True


def write_done_file(folder: str, args_str: str, object_names: List[str]):
    """Write a file to signify completion.

    This the done file includes the arguments to processing and
    a list of objects that were processed.

    Args:
        folder (str): Folder to write the done file to
        args_str (str): String representation of arguments
        object_names (List[str]): List of objects to convert to MDS format
    """
    with open(os.path.join(folder, DONE_FILENAME), 'w') as done_file:
        done_file.write('\n'.join([args_str] + object_names) + '\n')


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
)->int:
    """Convert a folder of text files to MDS format.

    Args:
        tokenizer_name (str): Name of tokenizer to use
        output_folder (str): Folder to write MDS shards to
        input_folder (str): Folder of text files to process
        concat_tokens (int): Concantenate up to this many tokens
        eos_text (str): Textend to append to each example to separate concatenated samples
        bos_text (str): Text to prepend to each example to separate concatenated samples
        no_wrap: (bool): Whether to let text examples wrap across multiple training examples
        compression (str): The compression algorithm to use for MDS writing
        processes (int): The number of processes to use.
        args_str (str): String representation of the arguments
        reprocess (bool): Whether to always reprocess the given folder of text files
    Returns:
        (int): total tokens of the dataset
    """
    is_remote_output = is_remote_path(output_folder)

    object_names = get_object_names(input_folder)
    if len(object_names) == 0:
        raise ValueError(f'No text files were found at {input_folder}.')

    # Check if the text files in the bucket have already been processed.
    if not reprocess and is_already_processed(output_folder, args_str,
                                              object_names):
        log.info(
            f'Input folder {input_folder} is already processed at {output_folder} and '
            +
            'reprocess is set to False. Set reprocess to True if you would like to force reprocessing.'
        )
        return

    # Use a temporary local directory if the output is remote and there are more than 1 processes
    local_output_folder = tempfile.TemporaryDirectory(
    ).name if is_remote_output else output_folder

    if processes > 1:
        # Download and convert the text files in parallel
        args = get_task_args(object_names, local_output_folder, input_folder,
                             processes, tokenizer_name, concat_tokens, eos_text,
                             bos_text, no_wrap, compression)
        with ProcessPoolExecutor(max_workers=processes) as executor:
            pool = list(executor.map(download_and_convert_starargs, args))

        # Merge the mds shards from each of the processes into a single folder
        # merge_shard_groups(local_output_folder)
        total_tokens = sum(pool)
    else:
        total_tokens = download_and_convert(object_names, local_output_folder, input_folder,
                             tokenizer_name, concat_tokens, eos_text, bos_text,
                             no_wrap, compression)

    # Write a done file with the args and object names
    write_done_file(local_output_folder, args_str, object_names)

    if is_remote_output:
        # Upload the local output to the remote location
        output_object_store = cast(
            ObjectStore, maybe_create_object_store_from_uri(output_folder))
        _, _, output_folder_prefix = parse_uri(output_folder)
        files_to_upload = os.listdir(local_output_folder)

        for file in files_to_upload:
            assert not os.path.isdir(file)
            remote_path = os.path.join(output_folder_prefix, file)
            output_object_store.upload_object(
                remote_path, os.path.join(local_output_folder, file))

    return total_tokens


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
    )

    return str(args)


from composer.utils import dist, get_file, parse_uri

from llmfoundry.data.finetuning.tasks import (DOWNLOADED_FT_DATASETS_DIRPATH,
                                              SUPPORTED_EXTENSIONS,
                                              dataset_constructor)


def _download_remote_hf_dataset(remote_path: str, split: str) -> str:
    """Downloads a dataset from a remote object store.

    This function supports 'jsonl', 'csv', and 'parquet' file formats for the dataset. It will attempt to download
    the dataset, then once it is downloaded, convert it into HuggingFace ``datasets`` format, and then return this
    dataset.

    The function also ensures synchronicity across multiple processes during the file download. It creates a signal
    file that is used to synchronize the start of the download across different processes. Once the download is
    completed, the function removes the signal file.

    Args:
        hf_name (str): The path of the HuggingFace dataset to download.
        split (str): The dataset split to download (e.g., 'train', 'validation', 'test').

    Returns:
        A local directory path where the dataset files are stored.

    Raises:
        FileNotFoundError: Raised if the dataset file cannot be found with any of the supported extensions.
    """
    finetune_dir = os.path.join(
        DOWNLOADED_FT_DATASETS_DIRPATH,
        split if split != 'data' else 'data_not',
    )
    os.makedirs(finetune_dir, exist_ok=True)
    for extension in SUPPORTED_EXTENSIONS:
        name = f'{remote_path.strip("/")}/{split}{extension}'
        destination = str(
            os.path.abspath(
                os.path.join(finetune_dir, 'data',
                             f'{split}-00000-of-00001{extension}')))

        # Since we don't know exactly what the extension will be, since it is one of a list
        # use a signal file to wait for instead of the desired file
        signal_file_path = os.path.join(
            finetune_dir, f'.node_{dist.get_node_rank()}_local_rank0_completed')
        if dist.get_local_rank() == 0:
            try:
                get_file(path=name, destination=destination, overwrite=True)
            except FileNotFoundError as e:
                if extension == SUPPORTED_EXTENSIONS[-1]:
                    files_searched = [
                        f'{cfg.dataset.hf_name}/{cfg.dataset.split}{ext}'
                        for ext in SUPPORTED_EXTENSIONS
                    ]
                    raise FileNotFoundError(
                        f'Could not find a file with any of ' + \
                        f'the supported extensions: {SUPPORTED_EXTENSIONS}\n' + \
                        f'at {files_searched}'
                    ) from e
                else:
                    log.debug(
                        f'Could not find {name}, looking for another extension')
                continue

            os.makedirs(os.path.dirname(signal_file_path), exist_ok=True)
            with open(signal_file_path, 'wb') as f:
                f.write(b'local_rank0_completed_download')

        # Avoid the collective call until the local rank zero has finished trying to download the dataset
        # so that we don't timeout for large downloads. This syncs all processes on the node
        with dist.local_rank_zero_download_and_wait(signal_file_path):
            # Then, wait to ensure every node has finished trying to download the dataset
            dist.barrier()

        # clean up signal file
        if dist.get_local_rank() == 0:
            os.remove(signal_file_path)
        dist.barrier()
        break
    return finetune_dir


def plot_hist(data, save_plot_path=None):
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


def get_import_exception_message(package_name: str, extra_deps: str) -> str:
    """Get import exception message.

    Args:
        package_name (str): Package name.

    Returns:
        str: Exception message.
    """
    return f'BYOD was installed without {extra_deps} support. ' + \
            f'To use {extra_deps} related packages with BYOD, run ' + \
            f'`pip install \'mosaicml-byod[{extra_deps}]\'`.'


def pandas_processing_fn(df: pd.DataFrame,
                         **args: Any) -> Iterable[Dict[str, bytes]]:
    """Tokenize helper function for dataframe_to_mds.

    Args:
        df (pandas.DataFrame): The input pandas DataFrame that needs to be processed.
        **args : Additional arguments to be passed to the 'process_some_data' function during processing.

    Returns:
        iterable obj
    """
    import datasets as hf_datasets
    from transformers import AutoTokenizer

    hf_dataset = hf_datasets.Dataset.from_pandas(df=df)
    tokenizer = AutoTokenizer.from_pretrained(args['tokenizer'])
    tokenizer.model_max_length = 5000000000  # Hack to prevent warnings from HuggingFace

    if bos_text + eos_text == '':
        test_tokens = tokenizer('test')
        if test_tokens['input_ids'][
                0] != tokenizer.bos_token_id and test_tokens['input_ids'][
                    -1] != tokenizer.eos_token_id:
            tok_error_msg = 'This tokenizer does not insert an EOS nor BOS token. '
            tok_error_msg += 'Concatenating with this tokenizer will result in sequences being '
            tok_error_msg += 'attached without a separating token. Please use another tokenizer, '
            tok_error_msg += 'such as facebook/opt-125m, or specify EOS/BOS text with e.g. '
            tok_error_msg += '--bos_text=<|endoftext|>.'
            raise ValueError(tok_error_msg)

    dataset = ConcatTokensDataset(
        hf_dataset=hf_dataset,
        max_length=args.get('concat_tokens', None),
        tokenizer=tokenizer,
        eos_text=args.get('eos_text', None),
        bos_text=args.get('bos_text', None),
        no_wrap=args.get('no_wrap', None),
    )

    for sample in dataset:  # pyright: ignore
        yield sample


# Copyright 2023 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""A utility to convert spark dataframe to MDS."""

import logging
import os
import shutil
from collections.abc import Iterable
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import pandas as pd

try:
    from pyspark import TaskContext
    from pyspark.sql.connect.dataframe import DataFrame as SparkConnDataFrame
    from pyspark.sql.dataframe import DataFrame as SparkSqlDataFrame
    from pyspark.sql.types import (ArrayType, BinaryType, BooleanType, ByteType,
                                   DateType, DayTimeIntervalType, DecimalType,
                                   DoubleType, FloatType, IntegerType, LongType,
                                   MapType, ShortType, StringType, StructField,
                                   StructType, TimestampNTZType, TimestampType)
except ImportError as e:
    e.msg = get_import_exception_message(e.name,
                                         extra_deps='spark')  # pyright: ignore
    #raise e

try:
    from dask.dataframe import DataFrame as DaskDataFrame
    from dask.distributed import Client, LocalCluster
except ImportError as e:
    e.msg = get_import_exception_message(e.name,
                                         extra_deps='dask')  # pyright: ignore
    #raise e
    DaskDataFrame = None

try:
    from streaming import MDSWriter
    from streaming.base.format.index import get_index_basename
    from streaming.base.format.mds.encodings import _encodings
    from streaming.base.storage.upload import CloudUploader
    from streaming.base.util import merge_index as do_merge_index
except ImportError as e:
    e.msg = get_import_exception_message(
        e.name, extra_deps='streaming')  # pyright: ignore
    #raise e

logger = logging.getLogger(__name__)

MAPPING_SPARK_TO_MDS = {
    ByteType: 'uint8',
    ShortType: 'uint16',
    IntegerType: 'int',
    LongType: 'int64',
    FloatType: 'float32',
    DoubleType: 'float64',
    DecimalType: 'str_decimal',
    StringType: 'str',
    BinaryType: 'bytes',
    BooleanType: None,
    TimestampType: None,
    TimestampNTZType: None,
    DateType: None,
    DayTimeIntervalType: None,
    ArrayType: None,
    MapType: None,
    StructType: None,
    StructField: None
}

MAPPING_DASK_TO_MDS = {'object': 'str', 'int64': 'int64', 'string': 'str'}


def isSparkDataFrame(dataframe: Union[SparkSqlDataFrame, SparkConnDataFrame,
                                      DaskDataFrame]):
    return isinstance(dataframe, SparkSqlDataFrame) or isinstance(
        dataframe, SparkConnDataFrame)


def infer_dataframe_schema(
        dataframe: Union[SparkSqlDataFrame, SparkConnDataFrame, DaskDataFrame],
        user_defined_cols: Optional[Dict[str, Any]] = None) -> Optional[Dict]:
    """Retrieve schema to construct a dictionary or do sanity check for
    MDSWriter.

    Args:
        dataframe (spark dataframe): dataframe to inspect schema
        user_defined_cols (Optional[Dict[str, Any]]): user specified schema for MDSWriter

    Returns:
        If user_defined_cols is None, return schema_dict (dict): column name and dtypes that are
        supported by MDSWriter, else None

    Raises:
        ValueError if any of the datatypes are unsupported by MDSWriter.
    """

    def map_spark_dtype(spark_data_type: Any) -> str:
        """Map spark data type to mds supported types.

        Args:
            spark_data_type: https://spark.apache.org/docs/latest/sql-ref-datatypes.html

        Returns:
            str: corresponding mds datatype for input.

        Raises:
            raise ValueError if no mds datatype is found for input type
        """
        mds_type = MAPPING_SPARK_TO_MDS.get(type(spark_data_type), None)
        if mds_type is None:
            raise ValueError(f'{spark_data_type} is not supported by MDSWriter')
        return mds_type

    def map_dask_dtype(dask_data_type: Any) -> str:
        """Map dask/pandas data type to mds supported types."""
        mds_type = MAPPING_DASK_TO_MDS.get(str(dask_data_type), None)
        if mds_type not in mds_supported_dtypes:
            raise ValueError(f'{dask_data_type} is not supported by MDSWriter')
        return mds_type

    mds_supported_dtypes = {
        mds_type for mds_type in MAPPING_SPARK_TO_MDS.values()
        if mds_type is not None
    }

    # user has provided schema, we just check if mds supports the dtype
    if user_defined_cols is not None:
        for col_name, user_dtype in user_defined_cols.items():
            if col_name not in dataframe.columns:
                raise ValueError(
                    f'{col_name} is not a column of input dataframe: {dataframe.columns}'
                )
            if user_dtype not in mds_supported_dtypes:
                raise ValueError(f'{user_dtype} is not supported by MDSWriter')

            if isSparkDataFrame(dataframe):
                actual_spark_dtype = dataframe.schema[col_name].dataType
                mapped_mds_dtype = map_spark_dtype(actual_spark_dtype)
            else:
                actual_dask_dtype = dataframe.dtypes.to_dict()[col_name]
                mapped_mds_dtype = map_dask_dtype(actual_dask_dtype)

            if user_dtype != mapped_mds_dtype:
                raise ValueError(
                    f'Mismatched types: column name `{col_name}` is `{mapped_mds_dtype}` in '
                    + f'DataFrame but `{user_dtype}` in user_defined_cols')
        return None

    schema_dict = {}

    if isSparkDataFrame(dataframe):
        schema = dataframe.schema
        for field in schema:
            dtype = map_spark_dtype(field.dataType)
            if dtype in _encodings:
                schema_dict[field.name] = dtype
            else:
                raise ValueError(f'{dtype} is not supported by MDSWriter')
    else:
        schema_dict = dataframe.dtypes.to_dict()
        for k, v in schema_dict.items():
            schema_dict[k] = map_dask_dtype(v)

    return schema_dict


def dataframeToMDS(
        dataframe: Union[SparkSqlDataFrame, SparkConnDataFrame, DaskDataFrame],
        merge_index: bool = True,
        mds_kwargs: Optional[Dict[str, Any]] = None,
        udf_iterable: Optional[Callable] = None,
        udf_kwargs: Optional[Dict[str, Any]] = None) -> Tuple[Any, int]:
    """Deprecated API Signature.

    To be replaced by dataframe_to_mds
    """
    logger.warning(
        'The DataframeToMDS signature has been deprecated and will be removed in Streaming 0.8. '
        + 'Use dataframe_to_mds with the same arguments going forward')
    return dataframe_to_mds(dataframe, merge_index, mds_kwargs, udf_iterable,
                            udf_kwargs)


def dataframe_to_mds(
        dataframe: Union[SparkSqlDataFrame, SparkConnDataFrame, DaskDataFrame],
        merge_index: bool = True,
        mds_kwargs: Optional[Dict[str, Any]] = None,
        udf_iterable: Optional[Callable] = None,
        udf_kwargs: Optional[Dict[str, Any]] = None) -> Tuple[Any, int]:
    """Execute a spark dataframe to MDS conversion process.

    This method orchestrates the conversion of a spark dataframe into MDS format by processing the
    input data, applying a user-defined iterable function if provided, and writing the results to
    an MDS-compatible format. The converted data is saved to mds_path.

    Args:
        dataframe (pyspark.sql.DataFrame or dask.dataframe): A DataFrame containing Delta Lake data.
        merge_index (bool): Whether to merge MDS index files. Defaults to ``True``.
        mds_kwargs (dict): Refer to https://docs.mosaicml.com/projects/streaming/en/stable/
            api_reference/generated/streaming.MDSWriter.html
        udf_iterable (Callable or None): A user-defined function that returns an iterable over the
            dataframe. udf_kwargs is the k-v args for the method. Defaults to ``None``.
        udf_kwargs (Dict): Additional keyword arguments to pass to the pandas processing
            function if provided. Defaults to an empty dictionary.

    Returns:
        mds_path (str or (str,str)): actual local and remote path were used
        fail_count (int): number of records failed to be converted

    Notes:
        - The method creates a SparkSession if not already available.
        - The 'udf_kwargs' dictionaries can be used to pass additional
          keyword arguments to the udf_iterable.
        - If udf_iterable is set, schema check will be skipped because the user defined iterable
          can create new columns. User must make sure they provide correct mds_kwargs[columns]
    """

    def write_mds_dask(pdf: pd.DataFrame, partition_info=None):

        fid = partition_info['number']  # pdf.index[0]
        if mds_path[1] == '':  # only local
            output = os.path.join(mds_path[0], f'{fid}')
            partition_path = (output, '')
        else:
            output = (os.path.join(mds_path[0], f'{fid}'),
                      os.path.join(mds_path[1], f'{fid}'))
            partition_path = output

        if mds_kwargs:
            kwargs = mds_kwargs.copy()
            kwargs['out'] = output
        else:
            kwargs = {}

        if merge_index:
            kwargs[
                'keep_local'] = True  # need to keep workers' locals to do merge

        if udf_iterable is not None:
            records = udf_iterable(pdf, **udf_kwargs or {})
        else:
            records = pdf.to_dict('records')
        assert isinstance(records, Iterable), (
            f'pandas_processing_fn needs to return an iterable instead of a ' +
            f'{type(records)}')

        with MDSWriter(**kwargs) as mds_writer:
            for sample in records:
                try:
                    mds_writer.write(sample)
                except Exception as ex:
                    raise RuntimeError(
                        f'failed to write sample: {sample}') from ex
                    count += 1

        return pd.DataFrame({
            'mds_path_local':
                [os.path.join(partition_path[0], get_index_basename())],
            'mds_path_remote': [
                os.path.join(partition_path[1], get_index_basename())
                if partition_path[1] != '' else ''
            ],
            'fail_count': [0]
        })
        return pdf.drop(cols, axis=1)

    def write_mds_spark(iterator: Iterable):
        """Worker node writes iterable to MDS datasets locally."""
        context = TaskContext.get()

        if context is not None:
            fid = context.taskAttemptId()
        else:
            raise RuntimeError('TaskContext.get() returns None')

        if mds_path[1] == '':  # only local
            output = os.path.join(mds_path[0], f'{fid}')
            partition_path = (output, '')
        else:
            output = (os.path.join(mds_path[0], f'{fid}'),
                      os.path.join(mds_path[1], f'{fid}'))
            partition_path = output

        if mds_kwargs:
            kwargs = mds_kwargs.copy()
            kwargs['out'] = output
        else:
            kwargs = {}

        if merge_index:
            kwargs[
                'keep_local'] = True  # need to keep workers' locals to do merge

        count = 0

        with MDSWriter(**kwargs) as mds_writer:
            for pdf in iterator:
                if udf_iterable is not None:
                    records = udf_iterable(pdf, **udf_kwargs or {})
                else:
                    records = pdf.to_dict('records')
                assert isinstance(records, Iterable), (
                    f'pandas_processing_fn needs to return an iterable instead of a '
                    + f'{type(records)}')

                for sample in records:
                    try:
                        mds_writer.write(sample)
                    except Exception as ex:
                        raise RuntimeError(
                            f'failed to write sample: {sample}') from ex
                        count += 1

        yield pd.concat([
            pd.Series([os.path.join(partition_path[0], get_index_basename())],
                      name='mds_path_local'),
            pd.Series([
                os.path.join(partition_path[1], get_index_basename())
                if partition_path[1] != '' else ''
            ],
                      name='mds_path_remote'),
            pd.Series([count], name='fail_count')
        ],
                        axis=1)

    if dataframe is None:
        raise ValueError(f'Input dataframe is None!')

    if not isSparkDataFrame(dataframe) and not isinstance(
            dataframe, DaskDataFrame):
        raise ValueError(
            f'dataframe_to_mds only takes Spark dataframe or Dask dataframe!')

    if (isSparkDataFrame(dataframe) and
            dataframe.isEmpty()) or (isinstance(dataframe, DaskDataFrame) and
                                     len(dataframe.index) == 0):
        print(f'Return. Input dataframe is Empty! Nothing to be done!')

    if not mds_kwargs:
        mds_kwargs = {}

    if not udf_kwargs:
        udf_kwargs = {}

    if 'out' not in mds_kwargs:
        raise ValueError(
            f'`out` and `columns` need to be specified in `mds_kwargs`')

    if udf_iterable is not None:
        if 'columns' not in mds_kwargs:
            raise ValueError(
                f'If udf_iterable is specified, user must provide correct `columns` in the '
                + f'mds_kwargs')
        logger.warning(
            "With udf_iterable defined, it's up to the user's discretion to provide "
            + "mds_kwargs[columns]'")
    else:
        if 'columns' not in mds_kwargs:
            logger.warning(
                "User's discretion required: columns arg is missing from mds_kwargs. Will be "
                + 'auto-inferred')
            mds_kwargs['columns'] = infer_dataframe_schema(dataframe)
            logger.warning(f"Auto inferred schema: {mds_kwargs['columns']}")
        else:
            infer_dataframe_schema(dataframe, mds_kwargs['columns'])

    out = mds_kwargs['out']
    keep_local = False if 'keep_local' not in mds_kwargs else mds_kwargs[
        'keep_local']
    cu = CloudUploader.get(out, keep_local=keep_local)

    # Fix output format as mds_path: Tuple(local, remote)
    if cu.remote is None:
        mds_path = (cu.local, '')
    else:
        mds_path = (cu.local, cu.remote)

    if isSparkDataFrame(dataframe):
        # Prepare partition schema
        result_schema = StructType([
            StructField('mds_path_local', StringType(), False),
            StructField('mds_path_remote', StringType(), False),
            StructField('fail_count', IntegerType(), False)
        ])
        partitions = dataframe.mapInPandas(func=write_mds_spark,
                                           schema=result_schema).collect()
    else:
        cluster = LocalCluster(processes=False)
        client = Client(cluster)
        partitions = dataframe.map_partitions(write_mds_dask,
                                              meta=pd.DataFrame(
                                                  {
                                                      'mds_path_local': str,
                                                      'mds_path_remote': str,
                                                      'fail_count': int
                                                  },
                                                  index=[0])).compute()

    keep_local_files = True
    # If there are no remote part, we always keep the local
    # In case user forgot to set keep_local and set out to be a local path
    if cu.remote is not None:  # If there are no remote
        if 'keep_local' in mds_kwargs and mds_kwargs['keep_local'] == False:
            keep_local_files = False

    if merge_index:
        if isSparkDataFrame(dataframe):
            index_files = list(
                set([(row['mds_path_local'], row['mds_path_remote'])
                     for row in partitions]))
        else:
            index_files = list(
                set([(row[1]['mds_path_local'], row[1]['mds_path_remote'])
                     for row in partitions.iterrows()]))

        do_merge_index(index_files,
                       out,
                       keep_local=keep_local_files,
                       download_timeout=60)

    if not keep_local_files:
        shutil.rmtree(cu.local, ignore_errors=True)

    sum_fail_count = 0
    if isSparkDataFrame(dataframe):
        for row in partitions:
            sum_fail_count += row['fail_count']

        if sum_fail_count > 0:
            logger.warning(
                f'Total failed records = {sum_fail_count}\nOverall records {dataframe.count()}'
            )
    return mds_path, sum_fail_count
